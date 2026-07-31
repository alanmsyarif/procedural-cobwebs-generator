# GPU web solver — native Blender GPU backend only (see gpu_native.py for
# the GLSL kernels and the Pixar/Kole physics notes). A frame-change
# handler steps the simulation; results are written into mesh attributes
# and a small "Arachne GPU Apply" Geometry Nodes modifier applies positions
# and deletes torn edges, so Strandify and the tension heatmap consume
# them unchanged. The base mesh is never modified.

import numpy as np

import bpy
from bpy.app.handlers import persistent
from mathutils import Vector
from bpy.props import (
    BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty,
    IntProperty, PointerProperty,
)
from bpy.types import Operator, PropertyGroup

from .constants import (
    GROUP_GPU_APPLY, A_PIN, A_SHOT, A_GPU_POS, A_BROKEN, A_TENSION,
    P_PULL, PULL_FIELD,
)
from .nodeutils import H

_STATES = {}


def gpu_backend_available():
    from . import gpu_native
    return gpu_native.native_available() and not gpu_native.native_broken()


def backend_reason():
    """Why the solver is unavailable, or "" when it is fine. The two causes
    need telling apart: a build without compute shaders can never work,
    while a backend disabled by a caught error is this session only and
    Reset GPU Sim clears it."""
    from . import gpu_native
    if not gpu_native.native_available():
        return "No compute shaders in this Blender build."
    if gpu_native.native_broken():
        msg = gpu_native.broken_reason()
        return "Solver hit an error: %s" % (msg or "see system console")
    return ""


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


def _drop_state(name):
    """Discard a simulation and its cache. Anything the web dragged the
    collider to belongs to that simulation, so it goes back where it
    started — otherwise each replay would begin from the end of the last."""
    _STATES.pop(name, None)
    _CACHE.pop(name, None)


def _drop_all_states():
    for name in list(_STATES):
        _drop_state(name)
    _CACHE.clear()


# ---------------------------------------------------------------------------
#  Web -> rigid body coupling
# ---------------------------------------------------------------------------
#
# Blender exposes no way to apply a force to one rigid body from Python —
# no force API, no velocity, nothing. The only in-scene mechanism is a
# force field, so the web's pull is fed in as a WIND field: uniform, aimed
# down the empty's local +Z, parked on the collider with a max distance
# just large enough to reach it and nothing else in the scene.
#
# Two consequences worth knowing. The field is set from a frame-change
# handler, which runs after the depsgraph has already evaluated Bullet for
# that frame, so the pull lands one frame later — invisible in motion, but
# it is why this wants playing forward from frame 1 rather than scrubbing.
# And Bullet only simulates onto frames its point cache has not reached, so
# a stale rigid body cache means the object will not budge; the setup
# operator clears it.


def _world_radius(o):
    """Radius of the object's world-space bounding box, about its centre."""
    pts = [o.matrix_world @ Vector(c) for c in o.bound_box]
    mid = sum(pts, Vector()) / len(pts)
    return max((p - mid).length for p in pts), mid


def _pull_field(obj):
    """The web's wind-field empty, or None. Lookup only — creating it needs
    operators, which a frame handler must not run (see ARN_OT_setup_pull)."""
    name = obj.get(P_PULL)
    emp = bpy.data.objects.get(name) if name else None
    if emp is None or getattr(emp, "field", None) is None:
        return None
    return emp


def _zero_pull(obj):
    """Switch the pull off. Needed on every frame the web is not stepped —
    a reset or a scrub leaves the field at whatever strength the last
    simulated frame set, and Bullet would go on hauling the object with a
    force from a web that no longer exists."""
    emp = _pull_field(obj)
    if emp is not None:
        emp.field.strength = 0.0


def _update_pull(obj, g, st):
    """Point the pull field along the web's net force on the collider and
    scale it to match. Pure data writes, safe from the frame handler."""
    emp = _pull_field(obj)
    if emp is None:
        return
    coll = g.collider
    rb = getattr(coll, "rigid_body", None) if coll is not None else None
    force = st.pull_force(g) if g.pull_collider else None
    if rb is None or rb.type != 'ACTIVE' or force is None:
        emp.field.strength = 0.0
        return
    mag = float(np.linalg.norm(force))
    if mag < 1e-9:
        emp.field.strength = 0.0
        return

    radius, mid = _world_radius(coll)
    # sit on the collider so it is certainly in range, and keep the reach
    # tight — every other active rigid body inside it would feel this too
    emp.location = mid
    emp.rotation_mode = 'QUATERNION'
    emp.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(
        Vector(force / mag))
    emp.field.distance_max = max(radius * 1.5, 1e-3)
    emp.field.strength = mag


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
        print("Arachne: render detected — replaying the cached web sim (GPU "
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
    st.carry(obj)                    # anchors onto their host before frame 1
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
                # rewinding starts a fresh web: drop the pull the previous
                # run left standing, or the object is already moving on
                # frame 1 under a force nothing is exerting any more
                _zero_pull(obj)
            elif st.last_frame is not None and frame == st.last_frame + 1:
                st.step(obj, g, dt)
                st.last_frame = frame
                arrays = st.write_back(obj)
                # after write_back: it leaves the readbacks pull_force
                # needs. Only on stepped frames — a scrub must not shove
                # the rigid body around without simulating the web.
                # Unconditional: it zeroes the field when pull is off, so
                # switching the toggle mid-playback lets go immediately
                _update_pull(obj, g, st)
                _cache_store(obj, frame, arrays)
            else:
                # scrubbing: the sim only advances on consecutive frames, so
                # the web holds — but anchors are kinematic, so keep them on
                # the collider/emitter they are stuck to instead of leaving
                # them behind wherever the last simulated frame put them
                st.last_frame = frame
                st.carry(obj)
                st.write_back(obj)
                _zero_pull(obj)          # no step, no pull
        except Exception as ex:      # never break playback
            gpu_native._mark_broken(ex)


# Dragging the emitter is not a frame change, so _on_frame never fires and
# the web sits there while the thing it is stuck to walks away. This runs on
# any interactive edit instead: anchors are re-placed on their host, and
# with Live Update on the web is rebuilt outright when the geometry it was
# built from is transformed — where the emitter is decides where every Web
# Shot strand starts, which way it flies and what it hits, and a Chaotic
# Cobweb is spun onto its selected surfaces, so moving or scaling either is
# as much a change to the web as any slider.
_IN_DEPSGRAPH = False
_SEEN = {}          # object name -> world matrix last acted on


def _moved(obj):
    if obj is None:
        return False
    cur = tuple(tuple(r) for r in obj.matrix_world)
    if _SEEN.get(obj.name) == cur:
        return False
    _SEEN[obj.name] = cur
    return True


def _aim_moved(p):
    """Has any aim target been dragged? An Aim Collection means the whole
    pool counts, not just whichever object the single-target slot holds —
    every member decides where one of the bursts lands. Not short-circuited:
    _moved records what it saw, so skipping the rest would leave them stale
    and fire a second rebuild on the next tick."""
    if p.shot_aim_collection is not None:
        return any([_moved(o) for o in p.shot_aim_collection.objects])
    return _moved(p.shot_aim)


def _env_moved(p):
    """Has any of a Chaotic Cobweb's anchor geometry been moved or scaled?
    That web is spun onto those surfaces — every thread is raycast against
    them — so dragging or scaling a wall is as much a change to the web as
    any slider. Same non-short-circuit rule as _aim_moved."""
    from .generator import env_objects
    return any([_moved(o) for o in env_objects(p.live_obj)])


def _hosts_moved(p):
    """Whether the geometry this web is built from has been transformed,
    per mode. Modes with no such geometry never rebuild from a drag."""
    if p.mode == 'SHOT':
        return _moved(p.shot_emitter) | _aim_moved(p)
    if p.mode == 'CHAOS':
        return _env_moved(p)
    return False


def note_hosts(p):
    """Record where a freshly generated web's host geometry currently is,
    so the depsgraph tick the generation itself causes doesn't read as a
    drag and rebuild the web a second time."""
    try:
        _hosts_moved(p)
    except Exception:
        pass


@persistent
def _on_depsgraph(scene, depsgraph=None):
    global _IN_DEPSGRAPH
    # our own attribute writes tag the mesh, which lands us straight back
    # here — and rebuilding a web tags plenty more
    if _IN_DEPSGRAPH or _render_active():
        return
    from . import gpu_native
    if gpu_native.native_broken():
        return
    _IN_DEPSGRAPH = True
    try:
        p = getattr(scene, "swf_web", None)
        if (p is not None and p.live and p.live_obj is not None
                and _hosts_moved(p)):
            from .generator import schedule_live_update
            # declines once the solver owns the mesh — fall through to
            # carrying the anchors instead of restarting the simulation
            if schedule_live_update():
                return
        for obj in scene.objects:
            g = getattr(obj, "swf_gpu", None)
            if g is None or not g.enabled or obj.type != 'MESH':
                continue
            st = _STATES.get(obj.name)
            if st is None or not st.hosts_moved():
                continue
            st.carry(obj)
            st.write_back(obj)
    except Exception as ex:
        gpu_native._mark_broken(ex)
    finally:
        _IN_DEPSGRAPH = False


# ---------------------------------------------------------------------------
#  GN apply group (positions + broken-edge deletion, feeds Strandify)
# ---------------------------------------------------------------------------

GPU_APPLY_VERSION = 2


def _build_apply_group():
    nt = bpy.data.node_groups.new(GROUP_GPU_APPLY, "GeometryNodeTree")
    nt.interface.new_socket(name="Geometry", in_out='INPUT',
                            socket_type='NodeSocketGeometry')
    nt.interface.new_socket(name="Geometry", in_out='OUTPUT',
                            socket_type='NodeSocketGeometry')
    h = H(nt)
    gi = h.n("NodeGroupInput", -600, 0)
    go = h.n("NodeGroupOutput", 600, 0)

    # Web Shot reveal: points the flying tip has not reached yet are
    # dropped, so the strands appear one shot at a time as the frames pass
    # (the solver keeps them frozen meanwhile). swf_shot_t is absent on
    # every other web type and reads 0, which culls nothing.
    stime = h.n("GeometryNodeInputSceneTime", -900, -180)
    shot_t = h.named('FLOAT', A_SHOT, -900, -320)
    unborn = h.cmp('FLOAT', 'GREATER_THAN', -750, -250,
                   shot_t.outputs["Attribute"], stime.outputs["Frame"],
                   label="not fired yet")
    reveal = h.n("GeometryNodeDeleteGeometry", -750, 0, label="shot reveal",
                 domain='POINT', mode='ALL')
    h.lk(gi.outputs["Geometry"], reveal.inputs["Geometry"])
    h.lk(unborn.outputs["Result"], reveal.inputs["Selection"])

    gpos = h.named('FLOAT_VECTOR', A_GPU_POS, -600, -300)
    sp = h.n("GeometryNodeSetPosition", -300, 0, label="GPU positions")
    h.lk(reveal.outputs["Geometry"], sp.inputs["Geometry"])
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


def apply_modifier(obj):
    """The object's Arachne GPU Apply modifier, or None. Its viewport
    toggle is what decides whether you are looking at the simulation or at
    the built mesh underneath it."""
    for m in getattr(obj, "modifiers", ()):
        if (m.type == 'NODES' and m.node_group
                and m.node_group.name.startswith(GROUP_GPU_APPLY)):
            return m
    return None


def invalidate_state(obj):
    """Throw away the simulation bound to `obj`'s previous mesh.

    Live Update swaps a freshly built mesh into the object. The solver only
    notices by itself when the vertex count changes, so any edit that keeps
    the topology (Clot, Arc, Slack, Stick To Emitter...) would otherwise go
    on being simulated — and displayed — from the old bind, and the change
    would look like it did nothing. Rebinding happens on the next frame."""
    _drop_state(obj.name)
    g = getattr(obj, "swf_gpu", None)
    if g is None or not g.enabled or obj.type != 'MESH':
        return
    # seed the new mesh's attributes, or the apply modifier reads a missing
    # position attribute (and collapses the web onto the origin) until the
    # first simulated frame lands
    me = obj.data
    _ensure_attr(me, A_GPU_POS, 'FLOAT_VECTOR', 'POINT')
    _ensure_attr(me, A_BROKEN, 'BOOLEAN', 'EDGE')
    _ensure_attr(me, A_TENSION, 'FLOAT', 'POINT')
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    me.attributes[A_GPU_POS].data.foreach_set("vector", co)


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
        mod = obj.modifiers.new("Arachne GPU Apply", 'NODES')
        mod.node_group = _ensure_apply_group()
        try:  # apply modifier must precede strandify
            with context.temp_override(object=obj):
                bpy.ops.object.modifier_move_to_index(
                    modifier=mod.name, index=0)
        except RuntimeError:
            pass

    g = obj.swf_gpu
    if me.attributes.get(A_SHOT) is not None:
        # a shot must be mid-flight at its fire frame, not pre-settled, and
        # fresh silk is taut — slack rest lengths make an impact splat sag
        # off the wall it just stuck to
        g.pre_warm = 0
        g.tension = 0.95
    if collider is not None:
        g.collider = collider
    g.enabled = True
    _drop_state(obj.name)


# ---------------------------------------------------------------------------
#  Properties + operators
# ---------------------------------------------------------------------------

def _static_update(self, context):
    """Keep the Static toggle and the collider's rigid body type in step —
    Passive is Bullet's own word for 'holds things but never moves'."""
    rb = getattr(self.collider, "rigid_body", None) if self.collider else None
    if rb is not None:
        rb.type = 'PASSIVE' if self.collider_static else 'ACTIVE'


class ARN_GPUProps(PropertyGroup):
    enabled: BoolProperty(name="Enabled", default=False)
    tension: FloatProperty(
        name="Tension", default=0.8, min=0.0, max=2.0,
        description="1 = taut threads, rest lengths exactly as built; "
                    "lower = slack that droops into catenaries. Above 1 "
                    "pre-tensions the silk — rest lengths shorter than "
                    "built, so the threads keep pulling and the last of the "
                    "gravity droop comes out (2 = 15% shorter). Tearing "
                    "measures strain against rest, so pre-tensioned threads "
                    "start out closer to snapping — the panel prints what "
                    "is left of the Tear Threshold")
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
    pull_collider: BoolProperty(
        name="Web Pulls Collider", default=False,
        description="Two-way coupling: threads stuck to the Collider haul "
                    "it around instead of only being held by it — fire a "
                    "web shot at a prop and drag it over. The object is "
                    "moved by Blender's rigid body sim, so its mass, "
                    "friction, gravity and tumble come from the Physics "
                    "tab. Single Collider object, not collections")
    pull_strength: FloatProperty(
        name="Pull Strength", default=200.0, min=0.0, max=100000.0,
        description="Scales thread stretch into force field strength. The "
                    "main dial: raise it if the web tugs but nothing moves. "
                    "Not newtons — rigid body force fields have no physical "
                    "unit, so tune it by eye")
    collider_static: BoolProperty(
        name="Static", default=False, update=_static_update,
        description="Make the collider a Passive rigid body — it holds the "
                    "web but the web can never move it. Off = Active, free "
                    "to be pulled")
    collider_collection: PointerProperty(
        name="Collider Collection", type=bpy.types.Collection,
        description="Collide against every mesh in this collection "
                    "(overrides the single Collider). Always baked as a "
                    "merged mesh SDF; treated as static — for animated "
                    "collision use a single Collider object instead")
    collision_offset: FloatProperty(name="Collision Offset", default=0.01,
                                    min=0.0, max=1.0)
    friction: FloatProperty(name="Friction", default=0.5, min=0.0, max=1.0)
    stickiness: FloatProperty(
        name="Stickiness", default=0.0, min=0.0, max=1.0,
        description="How readily threads adhere to the collider on "
                    "contact: a fraction of contacting points latch to "
                    "the surface and stay stuck (0 = slide off freely, "
                    "1 = every contact sticks — the web drapes and clings)")
    stick_follow: BoolProperty(
        name="Stuck Follows Collider", default=True,
        description="Threads anchored on the collider — a web shot's impact "
                    "points, or contacts latched by Stickiness — are carried "
                    "along as it moves and turns, instead of staying behind "
                    "in mid-air. Single collider object only")
    seed: IntProperty(name="Seed", default=0, min=0)


class ARN_OT_add_gpu_solver(Operator):
    """Enable the GPU solver on the active web.
    If another mesh is selected, it becomes the (sphere) collider"""
    bl_idname = "arachne.add_gpu_solver"
    bl_label = "Add GPU Solver"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == 'MESH'

    def execute(self, context):
        if not gpu_backend_available():
            self.report({'ERROR'}, backend_reason())
            return {'CANCELLED'}
        obj = context.object
        others = [o for o in context.selected_objects
                  if o is not obj and o.type == 'MESH']
        enable_gpu_solver(context, obj, others[0] if others else None)
        self.report({'INFO'}, "GPU solver active — play from frame 1.")
        return {'FINISHED'}


class ARN_OT_setup_pull(Operator):
    """Set up rigid body pulling: makes the Collider an Active rigid body
    and builds the force field the web drives it with. Run once, then play
    from frame 1"""
    bl_idname = "arachne.setup_pull"
    bl_label = "Set Up Rigid Body Pull"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return (obj is not None and obj.type == 'MESH'
                and getattr(obj, "swf_gpu", None) is not None
                and obj.swf_gpu.collider is not None)

    def execute(self, context):
        obj = context.object
        g = obj.swf_gpu
        coll = g.collider
        scene = context.scene

        if g.collider_collection is not None:
            self.report({'ERROR'},
                        "Rigid body pull works on a single Collider object, "
                        "not a collection — clear Collider Collection.")
            return {'CANCELLED'}

        if scene.rigidbody_world is None:
            bpy.ops.rigidbody.world_add()
        rbw = scene.rigidbody_world

        # the collider becomes an Active body so Bullet can move it
        if coll.rigid_body is None:
            with context.temp_override(object=coll, active_object=coll,
                                       selected_objects=[coll]):
                bpy.ops.rigidbody.object_add()
        coll.rigid_body.type = 'PASSIVE' if g.collider_static else 'ACTIVE'

        emp = _pull_field(obj)
        if emp is None:
            emp = bpy.data.objects.new(PULL_FIELD % obj.name, None)
            emp.empty_display_type = 'SINGLE_ARROW'
            emp.empty_display_size = 0.35
            emp.hide_render = True
            scene.collection.objects.link(emp)
            with context.temp_override(object=emp, active_object=emp,
                                       selected_objects=[emp]):
                bpy.ops.object.forcefield_toggle()
            obj[P_PULL] = emp.name
        if emp.field is None:
            self.report({'ERROR'}, "Could not add a force field.")
            return {'CANCELLED'}
        # WIND is the only uniform, directional field — everything else
        # falls off or swirls, and the web pulls in a straight line
        emp.field.type = 'WIND'
        emp.field.falloff_power = 0.0
        emp.field.use_max_distance = True
        emp.field.strength = 0.0

        # a scene that limits which effectors reach its rigid bodies would
        # otherwise ignore ours
        eff = rbw.effector_weights.collection
        if eff is not None and emp.name not in eff.objects:
            eff.objects.link(emp)

        # Bullet will not re-simulate frames its cache already holds
        try:
            with context.temp_override(point_cache=rbw.point_cache):
                bpy.ops.ptcache.free_bake()
        except Exception:
            pass

        g.pull_collider = True
        self.report({'INFO'},
                    "Rigid body pull ready — play from frame 1.")
        return {'FINISHED'}


class ARN_OT_remove_gpu_solver(Operator):
    """Disable the GPU solver and remove its apply modifier"""
    bl_idname = "arachne.remove_gpu_solver"
    bl_label = "Remove GPU Solver"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == 'MESH'

    def execute(self, context):
        obj = context.object
        obj.swf_gpu.enabled = False
        _drop_state(obj.name)
        for m in list(obj.modifiers):
            if (m.type == 'NODES' and m.node_group
                    and m.node_group.name.startswith(GROUP_GPU_APPLY)):
                obj.modifiers.remove(m)
        return {'FINISHED'}


class ARN_OT_reset_gpu(Operator):
    """Rebuild the GPU simulation state (after editing the web,
    changing Tension/Deteriorate, or repinning)"""
    bl_idname = "arachne.reset_gpu"
    bl_label = "Reset GPU Sim"

    def execute(self, context):
        _drop_all_states()
        from . import gpu_native
        gpu_native._clear_broken()
        return {'FINISHED'}


class ARN_OT_pin_vertices(Operator):
    """Write the current Edit Mode vertex selection into the pin
    attribute the GPU solver anchors on"""
    bl_idname = "arachne.pin_vertices"
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
        _drop_state(obj.name)         # pins bake into the sim state
        return {'FINISHED'}


classes = (ARN_GPUProps, ARN_OT_add_gpu_solver, ARN_OT_setup_pull,
           ARN_OT_remove_gpu_solver, ARN_OT_reset_gpu, ARN_OT_pin_vertices)


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
    ("depsgraph_update_post", _on_depsgraph),
    ("render_init", _on_render_begin),
    ("render_complete", _on_render_end),
    ("render_cancel", _on_render_end),
)


def _install_handlers():
    _remove_handlers()
    for list_name, fn in _HANDLERS:
        getattr(bpy.app.handlers, list_name).append(fn)


def _remove_handlers():
    # match on this add-on's own package name so renaming the installed
    # folder never orphans handlers (which would double up on re-enable)
    root = (__package__ or __name__).split(".")[0]
    for list_name, fn in _HANDLERS:
        handlers = getattr(bpy.app.handlers, list_name)
        for h in [h for h in handlers
                  if getattr(h, "__name__", "") == fn.__name__
                  and root in getattr(h, "__module__", "")]:
            handlers.remove(h)


def register():
    if hasattr(bpy.types.Object, "swf_gpu"):
        try:
            del bpy.types.Object.swf_gpu
        except Exception:
            pass
    for c in classes:
        _safe_register(c)
    bpy.types.Object.swf_gpu = PointerProperty(type=ARN_GPUProps)
    _install_handlers()


def unregister():
    _remove_handlers()
    _STATES.clear()
    _CACHE.clear()
    del bpy.types.Object.swf_gpu
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
