# GPU web solver — native Blender GPU backend only (see gpu_native.py for
# the GLSL kernels and the Pixar/Kole physics notes). A frame-change
# handler steps the simulation; results are written into mesh attributes
# and a small "SWF GPU Apply" Geometry Nodes modifier applies positions
# and deletes torn edges, so Strandify and the tension heatmap consume
# them unchanged. The base mesh is never modified.

import numpy as np

import bpy
from bpy.app.handlers import persistent
from bpy.props import (
    BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty,
    IntProperty, PointerProperty,
)
from bpy.types import Operator, PropertyGroup

from .constants import (
    GROUP_GPU_APPLY, A_PIN, A_GPU_POS, A_BROKEN, A_TENSION,
)
from .nodeutils import H

_STATES = {}


def gpu_backend_available():
    from . import gpu_native
    return gpu_native.native_available() and not gpu_native.native_broken()


def _ensure_attr(me, name, dtype, domain):
    a = me.attributes.get(name)
    if a is None or a.data_type != dtype or a.domain != domain:
        if a is not None:
            me.attributes.remove(a)
        me.attributes.new(name, dtype, domain)


# ---------------------------------------------------------------------------
#  Frame handler
# ---------------------------------------------------------------------------

# GPU compute is unavailable while a render job owns the GPU: frame-change
# handlers then run on the render thread, where the window's GPU context is
# not active, and any gpu.* call (dispatch/texture read) crashes the Vulkan
# backend outright. Instead, every frame simulated in the viewport is cached
# (positions / broken edges / tension per frame), and during renders the
# handler replays the cache with pure-CPU attribute writes. Playing through
# the frame range once in the viewport is the "bake".
_CACHE = {}   # obj name -> {frame: (pos, broken, tension) float32/bool}
_RENDERING = False
_RENDER_WARNED = False


def _cache_store(obj, frame, arrays):
    _CACHE.setdefault(obj.name, {})[frame] = arrays


def _cache_apply(obj, frame):
    """Replay a cached frame into mesh attributes (render thread safe).
    Falls back to the nearest earlier cached frame; holds last written
    state when nothing is cached yet."""
    from .gpu_native import apply_arrays
    cache = _CACHE.get(obj.name)
    if not cache:
        return
    entry = cache.get(frame)
    if entry is None:
        earlier = [f for f in cache if f <= frame]
        if not earlier:
            return
        entry = cache[max(earlier)]
    pos, brk, tens = entry
    me = obj.data
    if (pos.size != len(me.vertices) * 3
            or brk.size != len(me.edges)):
        return   # web was regenerated since the cache was recorded
    apply_arrays(obj, pos, brk, tens)


@persistent
def _on_render_begin(scene, depsgraph=None):
    global _RENDERING, _RENDER_WARNED
    _RENDERING = True
    if not _RENDER_WARNED:
        _RENDER_WARNED = True
        print("SWF: render detected — replaying the cached web sim (GPU "
              "compute can't run on the render thread). Play through the "
              "frame range once in the viewport to fill the cache.")


@persistent
def _on_render_end(scene, depsgraph=None):
    global _RENDERING
    _RENDERING = False


def _render_active():
    if _RENDERING:
        return True
    try:
        return bpy.app.is_job_running('RENDER')
    except Exception:
        return False


def _reset_state(obj, g, dt):
    from .gpu_native import NativeState
    st = NativeState(obj, g)
    for _ in range(g.pre_warm):      # settle before frame 1 (Kole)
        st.step(obj, g, dt)
    _STATES[obj.name] = st
    return st


@persistent
def _on_frame(scene, depsgraph=None):
    from . import gpu_native
    if gpu_native.native_broken():
        return
    rendering = _render_active()
    fps = scene.render.fps / scene.render.fps_base
    dt = 1.0 / max(fps, 1.0)
    frame = scene.frame_current
    for obj in scene.objects:
        g = getattr(obj, "swf_gpu", None)
        if g is None or not g.enabled or obj.type != 'MESH':
            continue
        if rendering:
            # no GPU access on the render thread — replay the viewport cache
            try:
                _cache_apply(obj, frame)
            except Exception:
                pass
            continue
        try:
            st = _STATES.get(obj.name)
            if (st is None or st.n != len(obj.data.vertices)
                    or frame <= scene.frame_start):
                st = _reset_state(obj, g, dt)
                st.last_frame = frame
                _cache_store(obj, frame, st.write_back(obj))
            elif st.last_frame is not None and frame == st.last_frame + 1:
                st.step(obj, g, dt)
                st.last_frame = frame
                _cache_store(obj, frame, st.write_back(obj))
            else:
                st.last_frame = frame
                st.write_back(obj)   # hold current state while scrubbing
        except Exception as ex:      # never break playback
            gpu_native._mark_broken(ex)


# ---------------------------------------------------------------------------
#  GN apply group (positions + broken-edge deletion, feeds Strandify)
# ---------------------------------------------------------------------------

GPU_APPLY_VERSION = 1


def _build_apply_group():
    nt = bpy.data.node_groups.new(GROUP_GPU_APPLY, "GeometryNodeTree")
    nt.interface.new_socket(name="Geometry", in_out='INPUT',
                            socket_type='NodeSocketGeometry')
    nt.interface.new_socket(name="Geometry", in_out='OUTPUT',
                            socket_type='NodeSocketGeometry')
    h = H(nt)
    gi = h.n("NodeGroupInput", -600, 0)
    go = h.n("NodeGroupOutput", 600, 0)

    gpos = h.named('FLOAT_VECTOR', A_GPU_POS, -600, -300)
    sp = h.n("GeometryNodeSetPosition", -300, 0, label="GPU positions")
    h.lk(gi.outputs["Geometry"], sp.inputs["Geometry"])
    h.lk(gpos.outputs["Attribute"], sp.inputs["Position"])

    brk = h.named('BOOLEAN', A_BROKEN, -300, -300)
    de = h.n("GeometryNodeDeleteGeometry", 0, 0, label="broken edges",
             domain='EDGE', mode='EDGE_FACE')
    h.lk(sp.outputs["Geometry"], de.inputs["Geometry"])
    h.lk(brk.outputs["Attribute"], de.inputs["Selection"])

    eov = h.n("GeometryNodeEdgesOfVertex", 0, -300)
    orphan = h.cmp('INT', 'EQUAL', 200, -300, eov.outputs["Total"], 0)
    dp = h.n("GeometryNodeDeleteGeometry", 300, 0, label="orphans",
             domain='POINT')
    h.lk(de.outputs["Geometry"], dp.inputs["Geometry"])
    h.lk(orphan.outputs["Result"], dp.inputs["Selection"])

    h.lk(dp.outputs["Geometry"], go.inputs["Geometry"])
    return nt


def _ensure_apply_group():
    nt = bpy.data.node_groups.get(GROUP_GPU_APPLY)
    if nt is not None:
        if nt.get("swf_version", 0) >= GPU_APPLY_VERSION:
            return nt
        nt.name = GROUP_GPU_APPLY + ".old"
    nt = _build_apply_group()
    nt["swf_version"] = GPU_APPLY_VERSION
    return nt


def enable_gpu_solver(context, obj, collider=None):
    """Create attributes, add the apply modifier, enable the solver."""
    me = obj.data
    _ensure_attr(me, A_GPU_POS, 'FLOAT_VECTOR', 'POINT')
    _ensure_attr(me, A_BROKEN, 'BOOLEAN', 'EDGE')
    _ensure_attr(me, A_TENSION, 'FLOAT', 'POINT')
    # seed positions so the web renders before first playback
    n = len(me.vertices)
    co = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", co)
    me.attributes[A_GPU_POS].data.foreach_set("vector", co)
    me.update_tag()

    if not any(m.type == 'NODES' and m.node_group
               and m.node_group.name.startswith(GROUP_GPU_APPLY)
               for m in obj.modifiers):
        mod = obj.modifiers.new("SWF GPU Apply", 'NODES')
        mod.node_group = _ensure_apply_group()
        try:  # apply modifier must precede strandify
            with context.temp_override(object=obj):
                bpy.ops.object.modifier_move_to_index(
                    modifier=mod.name, index=0)
        except RuntimeError:
            pass

    g = obj.swf_gpu
    if collider is not None:
        g.collider = collider
    g.enabled = True
    _STATES.pop(obj.name, None)
    _CACHE.pop(obj.name, None)


# ---------------------------------------------------------------------------
#  Properties + operators
# ---------------------------------------------------------------------------

class SWF_GPUProps(PropertyGroup):
    enabled: BoolProperty(name="Enabled", default=False)
    tension: FloatProperty(
        name="Tension", default=0.8, min=0.0, max=1.0,
        description="1 = taut threads, lower = slack that droops into "
                    "catenaries (rest lengths carry built-in slack)")
    resist_compression: BoolProperty(
        name="Resist Compression", default=False,
        description="Off = silk-like unilateral constraints (threads pull "
                    "but never push)")
    gravity: FloatVectorProperty(
        name="Gravity", default=(0.0, 0.0, -9.81), subtype='ACCELERATION')
    wind: FloatVectorProperty(name="Wind", default=(0.0, 0.0, 0.0))
    turbulence: FloatProperty(name="Turbulence", default=0.5, min=0.0,
                              max=50.0)
    damping: FloatProperty(
        name="Dampen", default=0.99, min=0.0, max=1.0,
        description="Lower = motion dies quickly, higher = bouncier")
    stiffness: FloatProperty(name="Stiffness", default=0.8, min=0.0, max=1.0)
    iterations: IntProperty(name="Iterations", default=16, min=1, max=128)
    substeps: IntProperty(name="Substeps", default=4, min=1, max=32)
    pre_warm: IntProperty(
        name="Pre-warm Frames", default=25, min=0, max=500,
        description="Physics steps run before frame 1 so the web starts "
                    "settled")
    deteriorate: FloatProperty(
        name="Deteriorate", default=0.0, min=0.0, max=0.9,
        description="Fraction of threads pre-broken at sim start")
    enable_tearing: BoolProperty(name="Tearing", default=True)
    tear_threshold: FloatProperty(name="Tear Threshold", default=1.5,
                                  min=1.01, max=10.0)
    enable_collision: BoolProperty(name="Collision", default=True)
    collision_shape: EnumProperty(
        name="Shape",
        items=[('SPHERE', "Bounding Sphere",
                "Fast: collider approximated by its bounding sphere"),
               ('MESH_SDF', "Mesh (SDF)",
                "Accurate: the collider mesh is baked into a signed "
                "distance field at sim start. Closed meshes only; "
                "animated location supported, rotation frozen at bake")],
        default='SPHERE')
    sdf_resolution: IntProperty(
        name="SDF Resolution", default=48, min=16, max=96,
        description="Voxel grid resolution of the baked collider field "
                    "(higher = more accurate, slower one-time bake)")
    collider: PointerProperty(
        name="Collider", type=bpy.types.Object,
        description="Collision object (sphere approximation or baked "
                    "mesh SDF depending on Shape)")
    collision_offset: FloatProperty(name="Collision Offset", default=0.01,
                                    min=0.0, max=1.0)
    friction: FloatProperty(name="Friction", default=0.5, min=0.0, max=1.0)
    seed: IntProperty(name="Seed", default=0, min=0)


class SWF_OT_add_gpu_solver(Operator):
    """Enable the GPU solver on the active web.
    If another mesh is selected, it becomes the (sphere) collider"""
    bl_idname = "swf.add_gpu_solver"
    bl_label = "Add GPU Solver"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == 'MESH'

    def execute(self, context):
        if not gpu_backend_available():
            self.report({'ERROR'},
                        "Blender GPU compute unavailable in this build.")
            return {'CANCELLED'}
        obj = context.object
        others = [o for o in context.selected_objects
                  if o is not obj and o.type == 'MESH']
        enable_gpu_solver(context, obj, others[0] if others else None)
        self.report({'INFO'}, "GPU solver active — play from frame 1.")
        return {'FINISHED'}


class SWF_OT_remove_gpu_solver(Operator):
    """Disable the GPU solver and remove its apply modifier"""
    bl_idname = "swf.remove_gpu_solver"
    bl_label = "Remove GPU Solver"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == 'MESH'

    def execute(self, context):
        obj = context.object
        obj.swf_gpu.enabled = False
        _STATES.pop(obj.name, None)
        _CACHE.pop(obj.name, None)
        for m in list(obj.modifiers):
            if (m.type == 'NODES' and m.node_group
                    and m.node_group.name.startswith(GROUP_GPU_APPLY)):
                obj.modifiers.remove(m)
        return {'FINISHED'}


class SWF_OT_reset_gpu(Operator):
    """Rebuild the GPU simulation state (after editing the web,
    changing Tension/Deteriorate, or repinning)"""
    bl_idname = "swf.reset_gpu"
    bl_label = "Reset GPU Sim"

    def execute(self, context):
        _STATES.clear()
        _CACHE.clear()
        from . import gpu_native
        gpu_native._clear_broken()
        return {'FINISHED'}


class SWF_OT_pin_vertices(Operator):
    """Write the current Edit Mode vertex selection into the pin
    attribute the GPU solver anchors on"""
    bl_idname = "swf.pin_vertices"
    bl_label = "Pin Selected"
    bl_options = {'REGISTER', 'UNDO'}

    action: bpy.props.EnumProperty(
        items=[('PIN', "Pin Selected", ""),
               ('UNPIN', "Unpin Selected", ""),
               ('CLEAR', "Clear All Pins", "")],
        default='PIN')

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == 'MESH'

    def execute(self, context):
        obj = context.object
        was_edit = (obj.mode == 'EDIT')
        if was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')
        me = obj.data
        attr = me.attributes.get(A_PIN)
        if (attr is None or attr.domain != 'POINT'
                or attr.data_type != 'BOOLEAN'):
            if attr is not None:
                me.attributes.remove(attr)
            attr = me.attributes.new(A_PIN, 'BOOLEAN', 'POINT')
        if self.action == 'CLEAR':
            for d in attr.data:
                d.value = False
        else:
            val = (self.action == 'PIN')
            for v in me.vertices:
                if v.select:
                    attr.data[v.index].value = val
        if was_edit:
            bpy.ops.object.mode_set(mode='EDIT')
        _STATES.pop(obj.name, None)   # pins bake into the sim state
        _CACHE.pop(obj.name, None)
        return {'FINISHED'}


classes = (SWF_GPUProps, SWF_OT_add_gpu_solver, SWF_OT_remove_gpu_solver,
           SWF_OT_reset_gpu, SWF_OT_pin_vertices)


def _safe_register(cls):
    old = getattr(bpy.types, cls.__name__, None)
    if old is not None:
        try:
            bpy.utils.unregister_class(old)
        except RuntimeError:
            pass
    bpy.utils.register_class(cls)


# (handler list, our function) pairs — render begin/end guard the GPU sim
_HANDLERS = (
    ("frame_change_post", _on_frame),
    ("render_init", _on_render_begin),
    ("render_complete", _on_render_end),
    ("render_cancel", _on_render_end),
)


def _install_handlers():
    _remove_handlers()
    for list_name, fn in _HANDLERS:
        getattr(bpy.app.handlers, list_name).append(fn)


def _remove_handlers():
    for list_name, fn in _HANDLERS:
        handlers = getattr(bpy.app.handlers, list_name)
        for h in [h for h in handlers
                  if getattr(h, "__name__", "") == fn.__name__
                  and "spider_web_forge" in getattr(h, "__module__", "")]:
            handlers.remove(h)


def register():
    if hasattr(bpy.types.Object, "swf_gpu"):
        try:
            del bpy.types.Object.swf_gpu
        except Exception:
            pass
    for c in classes:
        _safe_register(c)
    bpy.types.Object.swf_gpu = PointerProperty(type=SWF_GPUProps)
    _install_handlers()


def unregister():
    _remove_handlers()
    _STATES.clear()
    _CACHE.clear()
    del bpy.types.Object.swf_gpu
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
