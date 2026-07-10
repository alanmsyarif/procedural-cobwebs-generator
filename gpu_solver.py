# GPU web solver (Taichi) — runs the verlet/PBD/tearing simulation on the
# GPU (Vulkan / CUDA / Metal via Taichi, CPU fallback), driven by a
# frame-change handler.
#
# Physics adapted from:
#   * Chang & Luoh, "Dust and Cobwebs for Toy Story 4" (SIGGRAPH Talks 2019)
#   * Thomas Kole, "Geometry Nodes Cobwebs" (gitlab.com/thomaskole)
#
# Adaptations:
#   * TENSION: rest lengths are scaled by 1/tension-style slack so threads
#     droop into catenaries at rest (Kole's Tension; Pixar achieves the same
#     read with a smoothing post-process).
#   * UNILATERAL CONSTRAINTS: silk resists stretch, not compression. Slack
#     threads drape and fold instead of pushing back — this is what makes
#     low-tension webs look right.
#   * PRE-WARM: N solver steps run before frame 1 so the web starts settled.
#   * DETERIORATE: a random fraction of edges is pre-broken at sim start
#     for an aged look (independent of dynamic tearing).
#   * DAMPEN / GRAVITY as first-class controls, per the references.
#
# Data flow: the base mesh is never touched. Each frame the GPU writes
#   swf_gpu_pos  (POINT vector)  — simulated positions
#   swf_broken   (EDGE bool)     — torn / deteriorated edges
#   swf_tension  (POINT float)   — normalized stretch for the heatmap
# and a small "SWF GPU Apply" Geometry Nodes modifier moves the points and
# deletes broken edges, so Strandify and the tension material work unchanged.
#
# Collision: the collider object is approximated as its bounding sphere
# (animated transforms supported). Good for the ball-through-web shot;
# arbitrary-mesh GPU collision would need a BVH and is out of scope here.

import importlib.util
import random
import subprocess
import sys

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

# ---------------------------------------------------------------------------
#  Taichi bootstrap (lazy — the addon must load without taichi installed)
# ---------------------------------------------------------------------------

_TI = {}          # populated by _ensure_taichi(): {'ti': module, kernels...}
_STATES = {}      # object name -> _State
_HANDLER_KEY = "swf_gpu_frame_handler"


def taichi_available():
    return importlib.util.find_spec("taichi") is not None


def _gpu_backend_safe():
    """Probe the GPU backend in a throwaway subprocess: a broken driver
    segfaults the probe instead of crashing Blender."""
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "import taichi as ti; ti.init(arch=ti.gpu, log_level=ti.ERROR)"],
            capture_output=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


def _ensure_taichi():
    if _TI:
        return _TI
    import taichi as ti
    if _gpu_backend_safe():
        try:
            ti.init(arch=ti.gpu, log_level=ti.ERROR)
        except Exception:
            ti.init(arch=ti.cpu, log_level=ti.ERROR)
    else:
        print("SWF GPU solver: no usable GPU backend, using Taichi CPU.")
        ti.init(arch=ti.cpu, log_level=ti.ERROR)
    _TI['ti'] = ti

    @ti.func
    def _n1(p, t, seed):
        s = ti.sin(p[0] * 12.9898 + p[1] * 78.233 + p[2] * 37.719
                   + t * 2.1 + seed) * 43758.5453
        return (s - ti.floor(s)) * 2.0 - 1.0

    @ti.kernel
    def k_integrate(pos: ti.template(), prev: ti.template(),
                    pin: ti.template(),
                    gx: ti.f32, gy: ti.f32, gz: ti.f32,
                    wx: ti.f32, wy: ti.f32, wz: ti.f32,
                    turb: ti.f32, damping: ti.f32, dt2: ti.f32, t: ti.f32):
        for i in pos:
            if pin[i] == 0:
                p = pos[i]
                v = (p - prev[i]) * damping
                nse = ti.Vector([_n1(p, t, 0.0), _n1(p, t, 17.0),
                                 _n1(p, t, 39.0)])
                f = ti.Vector([gx, gy, gz]) + ti.Vector([wx, wy, wz]) \
                    + nse * turb
                prev[i] = p
                pos[i] = p + v + f * dt2
            else:
                prev[i] = pos[i]

    @ti.kernel
    def k_zero_vec(acc: ti.template()):
        for i in acc:
            acc[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def k_zero_f(f: ti.template()):
        for i in f:
            f[i] = 0.0

    @ti.kernel
    def k_edges(pos: ti.template(), acc: ti.template(),
                edges: ti.template(), rest: ti.template(),
                broken: ti.template(), resist_comp: ti.i32):
        for e in range(edges.shape[0]):
            if broken[e] == 0:
                i, j = edges[e][0], edges[e][1]
                d = pos[j] - pos[i]
                length = d.norm()
                if length > 1e-9:
                    stretch = length - rest[e]
                    # unilateral: silk pulls when stretched, never pushes
                    if stretch > 0.0 or resist_comp == 1:
                        c = d * (stretch / length * 0.5)
                        acc[i] += c
                        acc[j] -= c

    @ti.kernel
    def k_apply(pos: ti.template(), acc: ti.template(), pin: ti.template(),
                valence: ti.template(), sor: ti.f32):
        for i in pos:
            if pin[i] == 0 and valence[i] > 0:
                pos[i] += acc[i] / valence[i] * sor

    @ti.kernel
    def k_collide(pos: ti.template(), prev: ti.template(),
                  pin: ti.template(),
                  cx: ti.f32, cy: ti.f32, cz: ti.f32, r: ti.f32,
                  off: ti.f32, fric: ti.f32):
        for i in pos:
            if pin[i] == 0 and r > 0.0:
                c = ti.Vector([cx, cy, cz])
                d = pos[i] - c
                length = d.norm()
                if length < r + off:
                    nrm = d / ti.max(length, 1e-9)
                    target = c + nrm * (r + off)
                    prev[i] = prev[i] + (target - prev[i]) * fric
                    pos[i] = target

    @ti.kernel
    def k_tear(pos: ti.template(), edges: ti.template(),
               rest: ti.template(), broken: ti.template(),
               valence: ti.template(), tension: ti.template(),
               threshold: ti.f32, enable: ti.i32):
        for e in range(edges.shape[0]):
            if broken[e] == 0:
                i, j = edges[e][0], edges[e][1]
                length = (pos[j] - pos[i]).norm()
                ratio = length / ti.max(rest[e], 1e-8)
                t = (ratio - 1.0) / ti.max(threshold - 1.0, 0.01)
                t = ti.min(ti.max(t, 0.0), 1.0)
                ti.atomic_max(tension[i], t)
                ti.atomic_max(tension[j], t)
                if enable == 1 and length > rest[e] * threshold:
                    broken[e] = 1
                    ti.atomic_sub(valence[i], 1)
                    ti.atomic_sub(valence[j], 1)

    _TI.update(k_integrate=k_integrate, k_zero_vec=k_zero_vec,
               k_zero_f=k_zero_f, k_edges=k_edges, k_apply=k_apply,
               k_collide=k_collide, k_tear=k_tear)
    return _TI


# ---------------------------------------------------------------------------
#  Per-object simulation state
# ---------------------------------------------------------------------------

class _State:
    def __init__(self, obj, g):
        T = _ensure_taichi()
        ti = T['ti']
        me = obj.data
        n, m = len(me.vertices), len(me.edges)
        self.n, self.m = n, m
        self.last_frame = None

        pos = np.empty(n * 3, np.float32)
        me.vertices.foreach_get("co", pos)
        pos = pos.reshape(n, 3)

        edges = np.empty(m * 2, np.int32)
        me.edges.foreach_get("vertices", edges)
        edges = edges.reshape(m, 2)

        pin = np.zeros(n, np.int32)
        a = me.attributes.get(A_PIN)
        if a is not None and a.domain == 'POINT':
            tmp = np.zeros(n, np.bool_)
            a.data.foreach_get("value", tmp)
            pin = tmp.astype(np.int32)

        # TENSION (Kole): scale rest lengths so threads carry slack.
        # tension 1.0 -> taut; 0.0 -> rest lengths 2.5x built length.
        slack = 1.0 + (1.0 - g.tension) * 1.5
        init_len = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]],
                                  axis=1).astype(np.float32)
        rest = init_len * slack

        # DETERIORATE (Kole/Pixar): pre-break a random fraction of edges
        broken = np.zeros(m, np.int32)
        if g.deteriorate > 0.0 and m:
            rnd = random.Random(g.seed)
            broken[[e for e in range(m)
                    if rnd.random() < g.deteriorate]] = 1

        valence = np.zeros(n, np.int32)
        alive = broken == 0
        np.add.at(valence, edges[alive, 0], 1)
        np.add.at(valence, edges[alive, 1], 1)

        self.pos = ti.Vector.field(3, ti.f32, shape=max(n, 1))
        self.prev = ti.Vector.field(3, ti.f32, shape=max(n, 1))
        self.acc = ti.Vector.field(3, ti.f32, shape=max(n, 1))
        self.pin = ti.field(ti.i32, shape=max(n, 1))
        self.tension = ti.field(ti.f32, shape=max(n, 1))
        self.edges = ti.Vector.field(2, ti.i32, shape=max(m, 1))
        self.rest = ti.field(ti.f32, shape=max(m, 1))
        self.broken = ti.field(ti.i32, shape=max(m, 1))
        self.valence = ti.field(ti.i32, shape=max(n, 1))

        self.pos.from_numpy(pos)
        self.prev.from_numpy(pos)
        self.pin.from_numpy(pin)
        self.edges.from_numpy(edges)
        self.rest.from_numpy(rest)
        self.broken.from_numpy(broken)
        self.valence.from_numpy(valence)

    def step(self, obj, g, dt):
        T = _TI
        sub = max(g.substeps, 1)
        dt2 = (dt / sub) ** 2
        cx = cy = cz = 0.0
        r = 0.0
        coll = g.collider
        if g.enable_collision and coll is not None:
            loc = obj.matrix_world.inverted() @ coll.matrix_world.translation
            cx, cy, cz = loc.x, loc.y, loc.z
            r = max(coll.dimensions) * 0.5
        t_now = bpy.context.scene.frame_current / max(
            bpy.context.scene.render.fps, 1)
        sor = 1.0 + g.stiffness
        for _ in range(sub):
            T['k_integrate'](self.pos, self.prev, self.pin,
                             g.gravity[0], g.gravity[1], g.gravity[2],
                             g.wind[0], g.wind[1], g.wind[2],
                             g.turbulence, g.damping, dt2, t_now)
            for _i in range(max(g.iterations, 1)):
                T['k_zero_vec'](self.acc)
                T['k_edges'](self.pos, self.acc, self.edges, self.rest,
                             self.broken, 1 if g.resist_compression else 0)
                T['k_apply'](self.pos, self.acc, self.pin,
                             self.valence, sor)
                T['k_collide'](self.pos, self.prev, self.pin,
                               cx, cy, cz, r, g.collision_offset,
                               g.friction)
        T['k_zero_f'](self.tension)
        T['k_tear'](self.pos, self.edges, self.rest, self.broken,
                    self.valence, self.tension, g.tear_threshold,
                    1 if g.enable_tearing else 0)

    def write_back(self, obj):
        me = obj.data
        _ensure_attr(me, A_GPU_POS, 'FLOAT_VECTOR', 'POINT')
        _ensure_attr(me, A_BROKEN, 'BOOLEAN', 'EDGE')
        _ensure_attr(me, A_TENSION, 'FLOAT', 'POINT')
        me.attributes[A_GPU_POS].data.foreach_set(
            "vector", self.pos.to_numpy()[:self.n].ravel())
        me.attributes[A_BROKEN].data.foreach_set(
            "value", self.broken.to_numpy()[:self.m].astype(np.bool_))
        me.attributes[A_TENSION].data.foreach_set(
            "value", self.tension.to_numpy()[:self.n])
        me.update_tag()


def _ensure_attr(me, name, dtype, domain):
    a = me.attributes.get(name)
    if a is None or a.data_type != dtype or a.domain != domain:
        if a is not None:
            me.attributes.remove(a)
        me.attributes.new(name, dtype, domain)


# ---------------------------------------------------------------------------
#  Frame handler
# ---------------------------------------------------------------------------

def _make_state(obj, g):
    from . import gpu_native
    if (g.backend == 'NATIVE' and gpu_native.native_available()
            and not gpu_native.native_broken()):
        try:
            return gpu_native.NativeState(obj, g)
        except Exception as ex:
            gpu_native._mark_broken(ex)
    if not taichi_available():
        raise RuntimeError(
            "Native GPU backend unavailable and Taichi not installed")
    return _State(obj, g)


def _reset_state(obj, g, dt):
    st = _make_state(obj, g)
    # PRE-WARM (Kole): settle the physics before frame 1
    for _ in range(g.pre_warm):
        st.step(obj, g, dt)
    _STATES[obj.name] = st
    return st


@persistent
def _on_frame(scene, depsgraph=None):
    fps = scene.render.fps / scene.render.fps_base
    dt = 1.0 / max(fps, 1.0)
    frame = scene.frame_current
    for obj in scene.objects:
        g = getattr(obj, "swf_gpu", None)
        if g is None or not g.enabled or obj.type != 'MESH':
            continue
        try:
            st = _STATES.get(obj.name)
            if (st is None or st.n != len(obj.data.vertices)
                    or frame <= scene.frame_start):
                st = _reset_state(obj, g, dt)
                st.last_frame = frame
                st.write_back(obj)
            elif st.last_frame is not None and frame == st.last_frame + 1:
                st.step(obj, g, dt)
                st.last_frame = frame
                st.write_back(obj)
            else:
                st.last_frame = frame
                st.write_back(obj)  # hold current state while scrubbing
        except Exception as ex:  # never break playback
            print("SWF GPU solver:", ex)


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


# ---------------------------------------------------------------------------
#  Properties, operators, registration
# ---------------------------------------------------------------------------

class SWF_GPUProps(PropertyGroup):
    enabled: BoolProperty(name="Enabled", default=False)
    backend: EnumProperty(
        name="Backend",
        items=[('NATIVE', "Blender GPU",
                "Blender's built-in gpu module (GLSL compute) — no "
                "installation, uses Blender's own Vulkan/Metal/OpenGL "
                "backend"),
               ('TAICHI', "Taichi",
                "External Taichi library (pip install) — CUDA/Vulkan/"
                "Metal with CPU fallback")],
        default='NATIVE')
    tension: FloatProperty(
        name="Tension", default=0.8, min=0.0, max=1.0,
        description="1 = taut threads, lower = slack that droops into "
                    "catenaries (rest lengths carry built-in slack)")
    resist_compression: BoolProperty(
        name="Resist Compression", default=False,
        description="Off = silk-like unilateral constraints (threads pull "
                    "but never push); on = rod-like behavior")
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
        description="Fraction of threads pre-broken at sim start for an "
                    "aged look")
    enable_tearing: BoolProperty(name="Tearing", default=True)
    tear_threshold: FloatProperty(name="Tear Threshold", default=1.5,
                                  min=1.01, max=10.0)
    enable_collision: BoolProperty(name="Collision", default=True)
    collider: PointerProperty(
        name="Collider", type=bpy.types.Object,
        description="Approximated as its bounding sphere on the GPU")
    collision_offset: FloatProperty(name="Collision Offset", default=0.01,
                                    min=0.0, max=1.0)
    friction: FloatProperty(name="Friction", default=0.5, min=0.0, max=1.0)
    seed: IntProperty(name="Seed", default=0, min=0)


class SWF_OT_install_taichi(Operator):
    """Install the Taichi GPU compute library into Blender's Python
    (~100 MB download; Blender may freeze for a minute)"""
    bl_idname = "swf.install_taichi"
    bl_label = "Install Taichi (GPU)"

    def execute(self, context):
        try:
            subprocess.run([sys.executable, "-m", "ensurepip"], check=False)
            subprocess.run([sys.executable, "-m", "pip", "install",
                            "--upgrade", "taichi"], check=True)
        except Exception as ex:
            self.report({'ERROR'}, "Install failed: %s" % ex)
            return {'CANCELLED'}
        self.report({'INFO'}, "Taichi installed — restart Blender if the "
                              "GPU solver doesn't activate.")
        return {'FINISHED'}


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
        from . import gpu_native
        if not (gpu_native.native_available() or taichi_available()):
            self.report({'ERROR'}, "No GPU backend available — install "
                                   "Taichi or update Blender.")
            return {'CANCELLED'}
        obj = context.object
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
            try:  # place before strandify / everything else
                bpy.ops.object.modifier_move_to_index(
                    modifier=mod.name, index=0)
            except RuntimeError:
                pass

        # avoid double simulation: mute the CPU (GN) solver if present
        for m in obj.modifiers:
            if (m.type == 'NODES' and m.node_group
                    and m.node_group.name.startswith("SWF Tearing Solver")):
                m.show_viewport = False
                m.show_render = False

        g = obj.swf_gpu
        others = [o for o in context.selected_objects
                  if o is not obj and o.type == 'MESH']
        if others:
            g.collider = others[0]
        g.enabled = True
        _STATES.pop(obj.name, None)
        self.report({'INFO'}, "GPU solver active — play from frame 1.")
        return {'FINISHED'}


class SWF_OT_remove_gpu_solver(Operator):
    """Disable the GPU solver and restore the CPU solver modifiers"""
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
        for m in list(obj.modifiers):
            if (m.type == 'NODES' and m.node_group
                    and m.node_group.name.startswith(GROUP_GPU_APPLY)):
                obj.modifiers.remove(m)
            elif (m.type == 'NODES' and m.node_group
                    and m.node_group.name.startswith("SWF Tearing Solver")):
                m.show_viewport = True
                m.show_render = True
        return {'FINISHED'}


class SWF_OT_reset_gpu(Operator):
    """Rebuild the GPU simulation state (after editing the web,
    changing Tension/Deteriorate, or repinning)"""
    bl_idname = "swf.reset_gpu"
    bl_label = "Reset GPU Sim"

    def execute(self, context):
        _STATES.clear()
        return {'FINISHED'}


classes = (SWF_GPUProps, SWF_OT_install_taichi, SWF_OT_add_gpu_solver,
           SWF_OT_remove_gpu_solver, SWF_OT_reset_gpu)


def _safe_register(cls):
    """Register defensively: if a class with this name survived a failed
    or partial previous enable, evict it first."""
    old = getattr(bpy.types, cls.__name__, None)
    if old is not None:
        try:
            bpy.utils.unregister_class(old)
        except RuntimeError:
            pass
    bpy.utils.register_class(cls)


def register():
    if hasattr(bpy.types.Object, "swf_gpu"):
        try:
            del bpy.types.Object.swf_gpu
        except Exception:
            pass
    for c in classes:
        _safe_register(c)
    bpy.types.Object.swf_gpu = PointerProperty(type=SWF_GPUProps)
    handlers = bpy.app.handlers.frame_change_post
    for h in [h for h in handlers if getattr(h, "__name__", "") == "_on_frame"
              and "spider_web_forge" in getattr(h, "__module__", "")]:
        handlers.remove(h)
    handlers.append(_on_frame)


def unregister():
    handlers = bpy.app.handlers.frame_change_post
    for h in [h for h in handlers if getattr(h, "__name__", "") == "_on_frame"
              and "spider_web_forge" in getattr(h, "__module__", "")]:
        handlers.remove(h)
    _STATES.clear()
    del bpy.types.Object.swf_gpu
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
