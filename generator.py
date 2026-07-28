# Procedural orb-web generator — "natural" edition.
#
# Matches the look of real aged webs:
#   * spiral threads sag into scallops between radials
#   * uneven spacing between spiral turns
#   * damage: randomly missing spiral segments (and a few radial spans)
#   * asymmetry: web radius varies smoothly around the circle
#   * wavy radials via per-ring angular drift
#   * a spiral-free zone around the hub (real orb webs have one)
#   * slack tangle threads drooping across the structure
#
# Anchor thread endpoints are written into the `swf_pin` attribute that the
# tearing solver binds automatically. No bmesh operators are used, so vertex
# references stay valid throughout.

import json
import math
import random

import numpy as np

import bpy
import bmesh
from mathutils.bvhtree import BVHTree
from bpy.props import (
    IntProperty, FloatProperty, EnumProperty, BoolProperty,
    PointerProperty,
)
from bpy.types import Operator, PropertyGroup

from .constants import A_PIN, A_SHOT, A_NOTEAR, A_EMIT, P_EMITTER


# A property `update` callback runs in a restricted context where editing
# bpy.data (creating meshes, swapping obj.data) is unsafe, so the rebuild
# is deferred to a one-shot timer. The pending flag also coalesces the
# flurry of updates from dragging a slider into a single regeneration.
_LIVE_PENDING = False


def _live_update(self, context):
    """Property update hook: schedule a live rebuild if Live Update is on
    and a generated web is being tracked."""
    global _LIVE_PENDING
    if _LIVE_PENDING:
        return
    p = getattr(context.scene, "swf_web", None)
    if p is None or not p.live or p.live_obj is None:
        return
    _LIVE_PENDING = True
    bpy.app.timers.register(_live_timer, first_interval=0.0)


def schedule_live_update():
    """Ask for a live rebuild from outside a property callback — moving the
    emitter or the aim target changes the web just as much as any slider
    does. Coalesced through the same one-shot timer."""
    global _LIVE_PENDING
    if _LIVE_PENDING:
        return
    p = getattr(bpy.context.scene, "swf_web", None)
    if p is None or not p.live or p.live_obj is None:
        return
    _LIVE_PENDING = True
    bpy.app.timers.register(_live_timer, first_interval=0.0)


def _live_timer():
    """Runs just after a parameter change, in a context where rebuilding
    the mesh is safe. Returns None so it fires only once."""
    global _LIVE_PENDING
    _LIVE_PENDING = False
    context = bpy.context
    p = getattr(context.scene, "swf_web", None)
    if p is None or not p.live or p.live_obj is None:
        return None
    try:
        regenerate_live(context, p)
    except Exception as ex:                     # never break the UI
        print("Arachne live update failed:", ex)
    return None


class ARN_WebProps(PropertyGroup):
    mode: EnumProperty(
        name="Mode",
        items=[('ORB', "Orb Web", "Classic radial/spiral orb web"),
               ('CHAOS', "Chaotic Cobweb",
                "Spider-spun corner cobweb anchored to selected meshes "
                "(Pixar / Thomas Kole construction)"),
               ('SHOT', "Web Shot",
                "Strands fired from an emitter over time, sticking where "
                "they hit the selected geometry")],
        default='ORB', update=_live_update)
    cobweb_initial: IntProperty(
        name="Initial Lines", default=36, min=2, max=200,
        description="Anchor threads cast between the selected surfaces "
                    "before spinning starts", update=_live_update)
    cobweb_spiders: IntProperty(
        name="Spiders", default=6, min=1, max=32,
        description="Concurrent spinners (Pixar used 5-10)",
        update=_live_update)
    cobweb_steps: IntProperty(
        name="Spin Steps", default=600, min=10, max=3000,
        description="Total threads spun (Pixar used 50-1000)",
        update=_live_update)
    cobweb_spread: FloatProperty(
        name="Spread", default=0.6, min=0.0, max=1.0,
        description="0 = spiders knit dense local clumps, 1 = spinning "
                    "distributes uniformly across the whole volume "
                    "(spiders relocate often and take long bridging jumps)",
        update=_live_update)
    cobweb_jump: FloatProperty(
        name="Jump Distance", default=0.4, min=0.01, max=10.0,
        subtype='DISTANCE',
        description="Max distance a spider jumps per step — larger is "
                    "more chaotic, smaller is denser", update=_live_update)
    cobweb_bridge: FloatProperty(
        name="Bridge Bias", default=0.5, min=0.0, max=1.0,
        description="How much the anchor threads commit to spanning "
                    "gaps: 0 = local/corner webbing hugging each surface, "
                    "1 = long cables strung between separate pieces of "
                    "geometry (a bridge). Raise it when the web should "
                    "hang between floating objects",
        update=_live_update)
    cobweb_clump: FloatProperty(
        name="Clumping", default=0.0, min=0.0, max=1.0,
        description="Random clumping: spiders are drawn toward a few "
                    "random attractor spots, knitting dense knots there "
                    "with sparser spans between — the uneven density of "
                    "real cobwebs (0 = even, 1 = strong clumps)",
        update=_live_update)
    # ---- web shot ---------------------------------------------------------
    shot_emitter: PointerProperty(
        name="Emitter", type=bpy.types.Object,
        description="Object the strands are fired from (a hand bone's "
                    "empty, for instance). Its animated location is "
                    "sampled per shot, so a moving emitter leaves each "
                    "strand anchored where it fired. Unset = 3D cursor",
        update=_live_update)
    shot_aim: PointerProperty(
        name="Aim Target", type=bpy.types.Object,
        description="Fire toward this object. Unset = the emitter's local "
                    "-Z axis; with no emitter either, shots spray in all "
                    "directions", update=_live_update)
    shot_stick_emitter: BoolProperty(
        name="Stick To Emitter", default=True,
        description="The muzzle end stays attached to the emitter and is "
                    "carried along as it moves and turns — the same way an "
                    "impact point sticks to the geometry it hit — so the "
                    "web trails a moving hand instead of hanging in the air "
                    "where it fired. GPU solver only",
        update=_live_update)
    shot_count: IntProperty(
        name="Shots", default=14, min=1, max=400,
        description="Number of strands fired per burst", update=_live_update)
    shot_bursts: IntProperty(
        name="Bursts", default=1, min=1, max=50,
        description="How many times the emitter fires. Each burst is a "
                    "fresh volley of Shots, sampled at the emitter's "
                    "location at that moment", update=_live_update)
    shot_burst_gap: FloatProperty(
        name="Burst Gap", default=24.0, min=0.0, max=2000.0,
        description="Frames between the start of one burst and the next",
        update=_live_update)
    shot_start: IntProperty(
        name="First Shot Frame", default=1, min=0,
        description="Frame the first strand leaves the emitter",
        update=_live_update)
    shot_interval: FloatProperty(
        name="Shot Interval", default=1.5, min=0.0, max=100.0,
        description="Frames between consecutive shots (0 = the whole "
                    "burst fires at once)", update=_live_update)
    shot_speed: FloatProperty(
        name="Shot Speed", default=60.0, min=0.5, max=2000.0,
        description="Travel speed of the flying tip in metres per second "
                    "— sets how many frames a strand takes to reach its "
                    "impact point", update=_live_update)
    shot_range: FloatProperty(
        name="Range", default=12.0, min=0.1, max=500.0, subtype='DISTANCE',
        description="How far a strand flies before giving up. A shot that "
                    "hits nothing within range keeps a free, unpinned end. "
                    "Ignored while an Aim Target is set: the range then "
                    "follows the distance to the target, so shots reach it "
                    "however far it is and the ones the spread throws wide "
                    "stop level with it instead of sailing past",
        update=_live_update)
    shot_spread: FloatProperty(
        # ANGLE properties are stored in radians and only displayed in
        # degrees — bounds must be radians too
        name="Spread", default=math.radians(35.0), min=0.0, max=math.pi,
        subtype='ANGLE',
        description="Cone angle the burst fans out over. 0 = every shot on "
                    "the same line, 180° = spray in all directions",
        update=_live_update)
    shot_whip: FloatProperty(
        name="Whip", default=0.35, min=0.0, max=2.0,
        description="Sideways wander the strand carries in flight — 0 = "
                    "dead straight lines, high = lashing arcs. Several "
                    "waves per strand, so no two bend alike",
        update=_live_update)
    shot_clot: FloatProperty(
        name="Clot", default=0.5, min=0.0, max=0.95,
        description="How far the burst travels as a single clot of web "
                    "fluid before it opens into separate strands, as a "
                    "fraction of the flight. 0 = strands fan out straight "
                    "from the muzzle, 0.8 = a tight blob that only bursts "
                    "open just before it lands", update=_live_update)
    shot_clot_size: FloatProperty(
        name="Clot Thickness", default=0.03, min=0.001, max=2.0,
        subtype='DISTANCE',
        description="Radius of the travelling clot. Keep it small — a thin "
                    "cord reads as one rope, and the threads binding the "
                    "fibres together stay invisible inside it",
        update=_live_update)
    shot_clot_twist: FloatProperty(
        name="Clot Twist", default=1.5, min=0.0, max=12.0,
        description="Turns the strands braid around the clot's axis before "
                    "it opens. 0 = parallel cables, higher = a twisted, "
                    "knotted rope", update=_live_update)
    shot_arc: FloatProperty(
        name="Arc", default=0.5, min=-2.0, max=2.0,
        description="Lob: the strand rides up over the straight line and "
                    "drops onto the impact point, peaking about three "
                    "quarters of the way there — silk thrown rather than "
                    "fired. Height as a fraction of the shot length; "
                    "negative arcs under the line", update=_live_update)
    shot_slack: FloatProperty(
        name="Slack", default=0.08, min=0.0, max=1.0,
        description="Droop built into a landed strand, as a fraction of "
                    "its length (the solver takes it from there)",
        update=_live_update)
    shot_splat: IntProperty(
        name="Splat Strands", default=26, min=0, max=200,
        description="Silk sprayed across the surface where a shot lands: "
                    "radial strands running out from the impact point with "
                    "their tips stuck to the wall", update=_live_update)
    shot_splat_size: FloatProperty(
        name="Splat Size", default=0.35, min=0.0, max=20.0,
        subtype='DISTANCE',
        description="How far the splat spreads across the surface",
        update=_live_update)
    shot_splat_web: IntProperty(
        name="Splat Web", default=16, min=0, max=200,
        description="Chords knitted between neighbouring splat strands, "
                    "the webbing that fills the splat in", update=_live_update)
    shot_tangle: IntProperty(
        name="Cross Threads", default=6, min=0, max=200,
        description="Extra strands strung between shots already fired, "
                    "webbing the burst together", update=_live_update)

    radials: IntProperty(
        name="Radials", default=16, min=3, max=64,
        description="Number of radial threads", update=_live_update)
    rings: IntProperty(
        name="Spiral Turns", default=14, min=2, max=60,
        description="Number of turns in the capture spiral",
        update=_live_update)
    radius: FloatProperty(
        name="Radius", default=1.0, min=0.05, max=50.0,
        subtype='DISTANCE', description="Web radius", update=_live_update)
    hub_factor: FloatProperty(
        name="Hub Size", default=0.08, min=0.01, max=0.5,
        description="Hub radius as a fraction of the web radius",
        update=_live_update)
    jitter: FloatProperty(
        name="Irregularity", default=0.3, min=0.0, max=1.0,
        description="Angular drift, spacing unevenness and positional noise",
        update=_live_update)
    spiral_sag: FloatProperty(
        name="Spiral Sag", default=0.3, min=0.0, max=1.0,
        description="How much spiral threads droop into scallops "
                    "between radials", update=_live_update)
    damage: FloatProperty(
        name="Damage", default=0.15, min=0.0, max=0.9,
        description="Fraction of spiral segments missing (radials break "
                    "at a lower rate)", update=_live_update)
    asymmetry: FloatProperty(
        name="Asymmetry", default=0.25, min=0.0, max=1.0,
        description="Smooth variation of the web radius around the circle",
        update=_live_update)
    tangles: IntProperty(
        name="Tangle Threads", default=8, min=0, max=40,
        description="Slack chaotic threads drooping across the web",
        update=_live_update)
    detail: IntProperty(
        name="Detail", default=2, min=1, max=5,
        description="Sub-points per thread span (sag resolution for the "
                    "solver and the scallops)", update=_live_update)
    anchors: IntProperty(
        name="Anchor Threads", default=5, min=1, max=16,
        description="Number of anchor threads extended past the rim",
        update=_live_update)
    anchor_extend: FloatProperty(
        name="Anchor Length", default=0.35, min=0.05, max=3.0,
        description="Anchor thread length as a fraction of the web radius",
        update=_live_update)
    seed: IntProperty(name="Seed", default=0, min=0, update=_live_update)
    plane: EnumProperty(
        name="Plane",
        items=[('XZ', "XZ (vertical)",
                "Vertical web — sags naturally under -Z gravity"),
               ('XY', "XY (horizontal)", "Flat web in the ground plane")],
        default='XZ', update=_live_update)

    # ---- realtime tweaking ------------------------------------------------
    live: BoolProperty(
        name="Live Update", default=False,
        description="Rebuild the last generated web instantly as you tweak "
                    "the parameters above. Do this before adding the "
                    "solver — regenerating changes the topology",
        update=_live_update)
    live_obj: PointerProperty(
        type=bpy.types.Object,
        description="The generated web currently being tweaked live")


def build_web_data(context, p, env_objs=None):
    """Build the web mesh datablock from properties. Returns
    (mesh, name), or None for invalid setups (e.g. Chaotic with no
    anchor geometry)."""
    if p.mode == 'CHAOS':
        return _build_cobweb(context, p, env_objs or [])
    if p.mode == 'SHOT':
        return _build_shot(context, p, env_objs or [])
    return _build_orb(context, p)


def build_web_object(context, p, env_objs=None):
    """Create the web object from properties. Returns the object,
    or None (with a reason string) for invalid setups."""
    res = build_web_data(context, p, env_objs)
    if res is None:
        return None
    me, name = res
    return _finalize(context, me, name)


def regenerate_live(context, p):
    """Rebuild the tracked live web in place: swap fresh mesh data into
    the existing object so its transform, name and any downstream edits
    of the object (not the mesh) are preserved. Anchor geometry for
    Chaotic mode is restored from names stored at generation time."""
    obj = p.live_obj
    if obj is None or obj.name not in bpy.data.objects:
        return
    if obj.type != 'MESH' or obj.mode != 'OBJECT':
        return
    env = []
    raw = obj.get("swf_env")
    if raw:
        try:
            env = [bpy.data.objects[n] for n in json.loads(raw)
                   if n in bpy.data.objects]
        except Exception:
            env = []
    res = build_web_data(context, p, env)
    if res is None:                              # invalid — keep old data
        return
    me, name = res
    old = obj.data
    obj.data = me
    me.name = name
    # the running simulation belongs to the mesh that was just replaced
    from .gpu_solver import invalidate_state
    invalidate_state(obj)
    if old.users == 0:
        try:
            bpy.data.meshes.remove(old)
        except Exception:
            pass


def _build_orb(context, p):
    rnd = random.Random(p.seed)
    bm = bmesh.new()

    R, N = p.radials, p.rings
    hub = p.radius * p.hub_factor
    two_pi = 2.0 * math.pi

    # smooth asymmetric radius multiplier around the circle
    ph1 = rnd.uniform(0.0, two_pi)
    ph2 = rnd.uniform(0.0, two_pi)

    def rad_mult(j):
        a = two_pi * j / R
        return 1.0 + p.asymmetry * (0.6 * math.sin(a + ph1)
                                    + 0.4 * math.sin(2.0 * a + ph2))

    # base angles with jitter, plus per-ring angular drift (wavy radials)
    base_ang = [
        two_pi * j / R + rnd.uniform(-0.5, 0.5) * p.jitter * two_pi / R
        for j in range(R)
    ]
    drift = [[0.0] * R]
    for i in range(1, N):
        drift.append([
            drift[i - 1][j]
            + rnd.uniform(-1.0, 1.0) * p.jitter * 0.35 * two_pi / R
            for j in range(R)
        ])

    # non-uniform spacing between spiral turns
    if N > 1:
        w = [rnd.uniform(max(0.15, 1.0 - 0.8 * p.jitter),
                         1.0 + 0.8 * p.jitter) for _ in range(N - 1)]
        csum = [0.0]
        for x in w:
            csum.append(csum[-1] + x)
        ring_r = [hub + (p.radius - hub) * (csum[i] / csum[-1])
                  for i in range(N)]
    else:
        ring_r = [p.radius]

    def place(x, y):
        off = rnd.uniform(-1.0, 1.0) * p.jitter * 0.02 * p.radius
        if p.plane == 'XZ':
            return (x, off, y)
        return (x, y, off)

    # vertex grid
    pts = [[None] * R for _ in range(N)]
    verts = [[None] * R for _ in range(N)]
    for i in range(N):
        for j in range(R):
            a = base_ang[j] + drift[i][j]
            r = ring_r[i] * rad_mult(j) \
                * (1.0 + rnd.uniform(-1.0, 1.0) * p.jitter * 0.06)
            x, y = math.cos(a) * r, math.sin(a) * r
            pts[i][j] = (x, y)
            verts[i][j] = bm.verts.new(place(x, y))
    center = bm.verts.new(place(0.0, 0.0))

    def edge(a, b):
        if a is b:
            return
        try:
            bm.edges.new((a, b))
        except ValueError:  # already exists
            pass

    def perp(pa, pb):
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        length = math.hypot(dx, dy) or 1.0
        s = 1.0 if rnd.random() < 0.5 else -1.0
        return (-dy / length * s, dx / length * s)

    def chain2d(va, pa, vb, pb, segs, sag, sag_dir):
        """Connect va->vb through sub-points drooping toward sag_dir."""
        if segs <= 1:
            edge(va, vb)
            return
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        length = math.hypot(dx, dy)
        prev = va
        for s in range(1, segs):
            t = s / segs
            d = math.sin(math.pi * t) * sag * length
            jx = rnd.uniform(-1.0, 1.0) * p.jitter * 0.05 * length
            jy = rnd.uniform(-1.0, 1.0) * p.jitter * 0.05 * length
            v = bm.verts.new(place(pa[0] + dx * t + sag_dir[0] * d + jx,
                                   pa[1] + dy * t + sag_dir[1] * d + jy))
            edge(prev, v)
            prev = v
        edge(prev, vb)

    segs = p.detail + 1
    free = max(1, int(round(N * 0.12)))  # spiral-free zone near the hub

    # ---- capture spiral: scalloped, with damage gaps ----
    flat = [(i, j) for i in range(N) for j in range(R)]
    for k in range(len(flat) - 1):
        i, j = flat[k]
        i2, j2 = flat[k + 1]
        if i < free:
            continue
        if rnd.random() < p.damage:
            continue
        pa, pb = pts[i][j], pts[i2][j2]
        mx, my = (pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5
        ml = math.hypot(mx, my) or 1.0
        inward = (-mx / ml, -my / ml)  # scallops droop toward the hub
        sag = p.spiral_sag * rnd.uniform(0.4, 1.3)
        chain2d(verts[i][j], pa, verts[i2][j2], pb, segs, sag, inward)

    # ---- radials: taut with tiny wobble, rare breaks + hub spokes ----
    for j in range(R):
        if rnd.random() >= p.damage * 0.3:
            chain2d(center, (0.0, 0.0), verts[0][j], pts[0][j], segs,
                    p.spiral_sag * 0.05 * rnd.random(),
                    perp((0.0, 0.0), pts[0][j]))
        for i in range(N - 1):
            if rnd.random() < p.damage * 0.3:
                continue
            pa, pb = pts[i][j], pts[i + 1][j]
            chain2d(verts[i][j], pa, verts[i + 1][j], pb, segs,
                    p.spiral_sag * 0.08 * rnd.random(), perp(pa, pb))

    # ---- anchor threads past the rim; endpoints pinned ----
    anchor_ends = []
    step = max(1, R // p.anchors)
    for j in range(0, R, step):
        if len(anchor_ends) >= p.anchors:
            break
        a = base_ang[j] + drift[N - 1][j]
        dx, dy = math.cos(a), math.sin(a)
        prev = verts[N - 1][j]
        for s in range(1, 4):
            rr = p.radius * rad_mult(j) * (1.0 + p.anchor_extend * s / 3.0)
            v = bm.verts.new(place(dx * rr, dy * rr))
            edge(prev, v)
            prev = v
        anchor_ends.append(prev)

    # ---- slack tangle threads drooping across the web ----
    for _ in range(p.tangles):
        i1, j1 = rnd.randrange(free, N), rnd.randrange(R)
        i2, j2 = rnd.randrange(free, N), rnd.randrange(R)
        va, vb = verts[i1][j1], verts[i2][j2]
        if va is vb:
            continue
        ca, cb = va.co.copy(), vb.co.copy()
        length = (cb - ca).length
        if length < hub:
            continue
        prev = va
        n = max(4, segs + 2)
        for s in range(1, n):
            t = s / n
            co = ca.lerp(cb, t)
            co.z -= math.sin(math.pi * t) * length \
                * p.spiral_sag * rnd.uniform(0.4, 1.5)
            co.x += rnd.uniform(-1.0, 1.0) * 0.03 * length
            co.y += rnd.uniform(-1.0, 1.0) * 0.03 * length
            v = bm.verts.new(co)
            edge(prev, v)
            prev = v
        edge(prev, vb)

    # ---- finalize: pins straight from live references (no ops were run) ----
    bm.verts.index_update()
    pin_indices = {v.index for v in anchor_ends}

    me = bpy.data.meshes.new("SpiderWeb")
    bm.to_mesh(me)
    bm.free()

    attr = me.attributes.new(A_PIN, 'BOOLEAN', 'POINT')
    for idx in pin_indices:
        attr.data[idx].value = True

    return me, "SpiderWeb"


def _finalize(context, me, name):
    obj = bpy.data.objects.new(name, me)
    obj["swf_web"] = True  # excluded from cobweb environment scans
    context.collection.objects.link(obj)
    for o in context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    return obj


# ============================================================================
#  Chaotic cobweb — spider-spinning construction
#  (Chang & Luoh, "Dust and Cobwebs for Toy Story 4"; Thomas Kole's
#  Geometry Nodes implementation). Initial anchor lines are cast between
#  the selected surfaces; spiders then repeatedly jump toward random
#  nearby points, connect to the nearest existing thread, split it, and
#  land there. Rays blocked by geometry become new surface anchors.
# ============================================================================

def _env_data(context, env_objs):
    deps = context.evaluated_depsgraph_get()
    verts, tris = [], []
    for o in env_objs:
        ob = o.evaluated_get(deps)
        me = ob.to_mesh()
        me.calc_loop_triangles()
        mw = o.matrix_world
        base = len(verts)
        verts.extend([tuple(mw @ v.co) for v in me.vertices])
        tris.extend([tuple(base + i for i in lt.vertices)
                     for lt in me.loop_triangles])
        ob.to_mesh_clear()
    if not tris:
        return None
    bvh = BVHTree.FromPolygons(verts, tris)
    V = np.asarray(verts, dtype=np.float64)
    T = np.asarray(tris, dtype=np.int64)
    e1 = V[T[:, 1]] - V[T[:, 0]]
    e2 = V[T[:, 2]] - V[T[:, 0]]
    area = np.linalg.norm(np.cross(e1, e2), axis=1)
    cum = np.cumsum(area)
    if cum[-1] <= 0.0:
        return None
    cum /= cum[-1]
    return bvh, V, T, cum


def _sample_surface(rnd, V, T, cum):
    k = int(np.searchsorted(cum, rnd.random()))
    u, v = rnd.random(), rnd.random()
    if u + v > 1.0:
        u, v = 1.0 - u, 1.0 - v
    a, b, c = V[T[k, 0]], V[T[k, 1]], V[T[k, 2]]
    pnt = a * (1.0 - u - v) + b * u + c * v
    n = np.cross(b - a, c - a)
    n = n / (np.linalg.norm(n) + 1e-12)
    return pnt, n


def _rand_unit(rnd):
    v = np.array([rnd.gauss(0, 1), rnd.gauss(0, 1), rnd.gauss(0, 1)])
    return v / (np.linalg.norm(v) + 1e-12)


def _build_cobweb(context, p, env_objs):
    env_objs = [o for o in env_objs
                if o.type == 'MESH' and not o.get("swf_web")]
    if not env_objs:
        return None
    env = _env_data(context, env_objs)
    if env is None:
        return None
    bvh, V, T, cum = env
    rnd = random.Random(p.seed)
    from mathutils import Vector

    verts = []          # np arrays, world space
    segs = []           # [ia, ib]
    pinned = set()

    def add_vert(co, pin=False):
        verts.append(np.asarray(co, dtype=np.float64))
        if pin:
            pinned.add(len(verts) - 1)
        return len(verts) - 1

    # ---- initial anchor lines between surfaces ----
    # Anchor Span sets the target line length, but the ray reach extends
    # to the whole selected setup so a web can bridge separate / floating
    # pieces of geometry (e.g. strung between two objects like a bridge),
    # not only facing surfaces close together.
    span = p.radius * 2.0
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    bridge = p.cobweb_bridge
    # bridging wants the ray to cross the whole setup; local webbing
    # keeps its reach near the anchor span so threads stay short
    reach = max(span, diag * 1.05) if bridge > 0.0 else span
    min_len = span * 0.05
    # at high bias, reject short anchors that hug a single surface so
    # the web is carried by long spanning cables
    if bridge > 0.0:
        min_len = max(min_len, diag * 0.35 * bridge)

    attempts = 0
    budget = p.cobweb_initial * 60
    while len(segs) < p.cobweb_initial and attempts < budget:
        attempts += 1
        a, na = _sample_surface(rnd, V, T, cum)
        if rnd.random() >= bridge:
            # (a) outward normal-biased ray — fills corners / concave props
            d = _rand_unit(rnd)
            if d.dot(na) < 0.0:
                d = -d                        # keep in outward hemisphere
            if rnd.random() < 0.5:
                d = d + na * 1.2              # half strongly normal-biased
            d = d / (np.linalg.norm(d) + 1e-12)
            hit = bvh.ray_cast(Vector(a + na * 1e-4), Vector(d), reach)
            if hit[0] is None:
                # normals may face the other way (walls, flipped winding)
                d2 = d - na * (2.0 * d.dot(na))
                hit = bvh.ray_cast(Vector(a - na * 1e-4), Vector(d2),
                                   reach)
        else:
            # (b) span toward another surface sample — bridges the gap
            # between separate pieces of geometry
            b0, _nb = _sample_surface(rnd, V, T, cum)
            dvec = b0 - a
            dl = np.linalg.norm(dvec)
            if dl < 1e-6:
                continue
            dirv = dvec / dl
            hit = bvh.ray_cast(Vector(a + dirv * 1e-4), Vector(dirv),
                               reach)
        if hit[0] is None:
            continue
        b = np.asarray(hit[0], dtype=np.float64)
        if np.linalg.norm(b - a) < min_len:
            continue
        segs.append([add_vert(a, True), add_vert(b, True)])

    # last resort for very sparse or widely separated geometry: connect
    # distant surface-point pairs directly, so a bridge web still forms
    # even when rays through the gap keep missing
    if bridge > 0.0:
        need = max(4, int(p.cobweb_initial * (0.25 + 0.75 * bridge)))
        tries = 0
        while len(segs) < need and tries < p.cobweb_initial * 20:
            tries += 1
            a, _ = _sample_surface(rnd, V, T, cum)
            b, _ = _sample_surface(rnd, V, T, cum)
            if np.linalg.norm(b - a) > max(min_len, diag * 0.25):
                segs.append([add_vert(a, True), add_vert(b, True)])

    # safety net: geometry that no ray could link (fully separate pieces
    # with Bridge Bias at 0) still gets a few anchors rather than failing
    tries = 0
    floor = min(4, p.cobweb_initial)
    while len(segs) < floor and tries < 400:
        tries += 1
        a, _ = _sample_surface(rnd, V, T, cum)
        b, _ = _sample_surface(rnd, V, T, cum)
        if np.linalg.norm(b - a) > span * 0.05:
            segs.append([add_vert(a, True), add_vert(b, True)])

    if not segs:
        return None

    # ---- spiders spin threads ----
    def spawn_on_thread():
        """Split a random segment and return the new junction vertex —
        keeps every spawn topologically attached to the web."""
        k = rnd.randrange(len(segs))
        ia, ib = segs[k]
        t = rnd.uniform(0.15, 0.85)
        ni = add_vert(verts[ia] * (1 - t) + verts[ib] * t)
        segs[k] = [ia, ni]
        segs.append([ni, ib])
        return ni

    spiders = [spawn_on_thread() for _ in range(p.cobweb_spiders)]

    # random clumping: a few attractor centres pull spiders into dense
    # local knots, leaving sparser spans between — the uneven density of
    # real cobwebs. Each spider gravitates toward its assigned centre.
    clump = p.cobweb_clump
    clump_centers = []
    spider_home = [0] * len(spiders)
    if clump > 0.0:
        nc = 3 + int(round(clump * 4))
        for _ in range(nc):
            ia, ib = segs[rnd.randrange(len(segs))]
            t = rnd.uniform(0.2, 0.8)
            clump_centers.append(verts[ia] * (1.0 - t) + verts[ib] * t)
        spider_home = [rnd.randrange(nc) for _ in range(len(spiders))]
    clump_p = 0.9 * clump

    def rehome(sj):
        if clump_centers:
            spider_home[sj] = rnd.randrange(len(clump_centers))

    relocate_every = max(3, int(round(30.0 - 24.0 * p.cobweb_spread)))
    long_jump_p = 0.08 + 0.32 * p.cobweb_spread
    for step in range(p.cobweb_steps):
        si = step % len(spiders)
        # periodic relocation spreads spinning over the whole web instead
        # of letting each spider random-walk a local clump
        if step > 0 and (step // len(spiders)) % relocate_every == 0 \
                and si == 0:
            for sj in range(len(spiders)):
                spiders[sj] = spawn_on_thread()
                rehome(sj)
        pi = spiders[si]
        P = verts[pi]
        placed = False
        for _try in range(6):
            jump = p.cobweb_jump * (2.5 if rnd.random() < long_jump_p
                                    else 1.0)
            if clump_centers and rnd.random() < clump_p:
                # head toward the assigned clump centre (with wander), so
                # spinning knots up there; once close it spins locally
                toward = clump_centers[spider_home[si]] - P
                dc = np.linalg.norm(toward)
                if dc > 1e-6:
                    bdir = toward / dc + _rand_unit(rnd) * 0.6
                    bdir = bdir / (np.linalg.norm(bdir) + 1e-12)
                    step_len = min(jump, dc * rnd.uniform(0.35, 0.95))
                    Q = P + bdir * max(step_len, jump * 0.15)
                else:
                    Q = P + _rand_unit(rnd) * rnd.uniform(0.15, 1.0) * jump
            else:
                Q = P + _rand_unit(rnd) * rnd.uniform(0.15, 1.0) * jump
            # blocked by geometry -> land on the surface (new anchor)
            dvec = Q - P
            dist = np.linalg.norm(dvec)
            if dist < 1e-9:
                continue
            dirv = dvec / dist
            hit = bvh.ray_cast(Vector(P + dirv * 1e-4), Vector(dirv), dist)
            if hit[0] is not None:
                land = (np.asarray(hit[0])
                        + np.asarray(hit[1]) * 2e-3)
                ni = add_vert(land, True)
                segs.append([pi, ni])
                spiders[si] = ni
                placed = True
                break
            # connect to the nearest existing thread
            A = np.stack([verts[s[0]] for s in segs])
            B = np.stack([verts[s[1]] for s in segs])
            D = B - A
            L2 = np.einsum('ij,ij->i', D, D)
            t = np.clip(np.einsum('ij,ij->i', Q - A, D)
                        / np.maximum(L2, 1e-12), 0.0, 1.0)
            C = A + D * t[:, None]
            k = int(np.argmin(np.linalg.norm(C - Q, axis=1)))
            R = C[k]
            if np.linalg.norm(R - P) < p.cobweb_jump * 0.05:
                continue  # degenerate — retry with a new target
            tk = t[k]
            ia, ib = segs[k]
            if tk < 0.08:
                ni = ia
            elif tk > 0.92:
                ni = ib
            else:  # split the thread at the landing point
                ni = add_vert(R)
                segs[k] = [ia, ni]
                segs.append([ni, ib])
            segs.append([pi, ni])
            spiders[si] = ni
            placed = True
            break
        if not placed:
            spiders[si] = spawn_on_thread()  # stuck spider relocates
            rehome(si)

    # ---- to bmesh, with sag + detail subdivision ----
    bm = bmesh.new()
    bverts = [bm.verts.new(tuple(co)) for co in verts]

    def edge(a, b):
        if a is b:
            return
        try:
            bm.edges.new((a, b))
        except ValueError:
            pass

    segn = max(p.detail, 1) + 1
    for ia, ib in segs:
        a, b = verts[ia], verts[ib]
        length = np.linalg.norm(b - a)
        droop = length * p.spiral_sag * rnd.uniform(0.2, 1.0)
        prev = bverts[ia]
        for s in range(1, segn):
            t = s / segn
            co = a + (b - a) * t
            co = co.copy()
            co[2] -= math.sin(math.pi * t) * droop
            co += (np.array([rnd.uniform(-1, 1) for _ in range(3)])
                   * p.jitter * 0.02 * length)
            v = bm.verts.new(tuple(co))
            edge(prev, v)
            prev = v
        edge(prev, bverts[ib])

    bm.verts.index_update()
    pin_indices = {bverts[i].index for i in pinned}

    me = bpy.data.meshes.new("Cobweb")
    bm.to_mesh(me)
    bm.free()

    attr = me.attributes.new(A_PIN, 'BOOLEAN', 'POINT')
    for idx in pin_indices:
        attr.data[idx].value = True

    return me, "Cobweb"


# ============================================================================
#  Web shot — strands fired from an emitter over time.
#
#  Each shot is a polyline built from the emitter's position at its fire
#  frame to whatever it hits. Every point stores the frame the flying tip
#  reaches it (A_SHOT); Strandify's Shot Reveal culls points that haven't
#  been reached yet, so a strand visibly grows out to its impact point at
#  Shot Speed. The muzzle end is pinned where the emitter was when it
#  fired — and, with Stick To Emitter, marked A_EMIT so the GPU solver
#  carries it along with the emitter from that frame on; the tip is pinned
#  only if it hit geometry — a miss leaves a free end that whips under the
#  solver. The whole thing repeats Bursts times, Burst Gap frames apart.
# ============================================================================

def _action_fcurves(obj):
    """F-curves of `obj`'s action, from either a legacy action or a
    slotted (4.4+) one."""
    ad = obj.animation_data
    act = ad.action if ad else None
    if act is None:
        return []
    curves = list(getattr(act, "fcurves", []))
    if curves:
        return curves
    slot = getattr(ad, "action_slot", None)
    for layer in getattr(act, "layers", []):
        for strip in getattr(layer, "strips", []):
            cb = strip.channelbag(slot) if (
                slot is not None and hasattr(strip, "channelbag")) else None
            if cb is not None:
                curves.extend(cb.fcurves)
    return curves


def _obj_loc_at(obj, frame):
    """World location of `obj` at `frame`, evaluated straight off its
    location F-curves. Reading the curves rather than stepping the scene
    frame keeps this callable from the Live Update timer. Rotation,
    parenting and constraints are taken as they stand now."""
    if obj is None:
        return None
    here = np.asarray(obj.matrix_world.translation, dtype=np.float64)
    loc = list(obj.location)
    animated = False
    for fc in _action_fcurves(obj):
        if fc.data_path == "location" and 0 <= fc.array_index < 3:
            loc[fc.array_index] = fc.evaluate(frame)
            animated = True
    if not animated:
        return here
    from mathutils import Vector
    basis = obj.matrix_basis.copy()
    basis.translation = Vector(loc)
    parent = obj.matrix_world @ obj.matrix_basis.inverted_safe()
    return np.asarray((parent @ basis).translation, dtype=np.float64)


def _cone_dir(rnd, axis, half_angle):
    """Random unit vector inside a cone of `half_angle` around `axis`."""
    ca = math.cos(min(max(half_angle, 0.0), math.pi))
    z = rnd.uniform(ca, 1.0)
    r = math.sqrt(max(0.0, 1.0 - z * z))
    ph = rnd.uniform(0.0, 2.0 * math.pi)
    a = np.asarray(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    t = (np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9
         else np.array([1.0, 0.0, 0.0]))
    u = np.cross(t, a)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(a, u)
    return u * (r * math.cos(ph)) + v * (r * math.sin(ph)) + a * z


def _build_shot(context, p, env_objs):
    from mathutils import Vector

    rnd = random.Random(p.seed)
    scene = context.scene
    fps = scene.render.fps / max(scene.render.fps_base, 1e-6)
    speed = max(p.shot_speed, 1e-3)
    emit, aim = p.shot_emitter, p.shot_aim
    cursor = np.asarray(scene.cursor.location, dtype=np.float64)

    # What the shots can hit. The Aim Target joins the selection whether or
    # not it was selected — a shot fired at something has to stop on it
    # rather than pass through — and the emitter is taken out, since silk
    # leaving a hand should not immediately stick to that hand.
    env_objs = [o for o in env_objs
                if o.type == 'MESH' and not o.get("swf_web")]
    if (aim is not None and aim.type == 'MESH' and aim not in env_objs
            and not aim.get("swf_web")):
        env_objs.append(aim)
    env_objs = [o for o in env_objs if o is not emit]
    env = _env_data(context, env_objs) if env_objs else None
    bvh = env[0] if env is not None else None

    # a mesh Aim Target is shot at all over, not just at its origin
    aim_surf = None
    aim_now = np.zeros(3)
    if aim is not None and aim.type == 'MESH':
        aim_surf = _env_data(context, [aim])
        aim_now = np.asarray(aim.matrix_world.translation, dtype=np.float64)

    # only the emitter's location is sampled per shot — its orientation is
    # read once, so an animated aim needs an Aim Target rather than a
    # rotating emitter
    axis0 = None
    if emit is not None:
        mw = emit.matrix_world
        a0 = -np.array([mw[0][2], mw[1][2], mw[2][2]], dtype=np.float64)
        if np.linalg.norm(a0) > 1e-9:
            axis0 = a0 / np.linalg.norm(a0)

    def outside(co, clear=8e-3):
        """Lift a point back out of whatever it landed inside.

        Silk is laid out analytically — lobbed, whipped, sagging — with no
        idea the target is in the way, so spans near an impact and cross
        threads strung between two landed tips end up buried in the surface.
        Nearest-surface tells us how far in; the normal takes it back out."""
        if bvh is None:
            return co
        near = bvh.find_nearest(Vector(co))
        if near[0] is None:
            return co
        surf = np.asarray(near[0], dtype=np.float64)
        nrm = np.asarray(near[1], dtype=np.float64)
        if float(np.dot(co - surf, nrm)) < clear:
            return surf + nrm * clear
        return co

    verts, times, segs, pinned = [], [], [], set()
    muzzles = set()                 # anchors that ride the emitter

    def add_vert(co, t, pin=False, muzzle_end=False):
        verts.append(np.asarray(co, dtype=np.float64))
        times.append(float(t))
        if pin:
            pinned.add(len(verts) - 1)
        if muzzle_end:
            muzzles.add(len(verts) - 1)
        return len(verts) - 1

    def add_splat(hub, center, normal, arrive):
        """The impact splat: silk sprayed across the surface the shot just
        hit — radial strands out from the impact point with their tips
        stuck down, knitted together by a few chords. `hub` is the
        strand's tip vertex, already sitting on the surface."""
        if p.shot_splat <= 0 or p.shot_splat_size <= 0.0:
            return
        nrm = normal / (np.linalg.norm(normal) + 1e-12)
        t0 = np.cross(nrm, _rand_unit(rnd))
        if np.linalg.norm(t0) < 1e-9:
            t0 = np.cross(nrm, np.array([1.0, 0.0, 0.0]))
        t0 = t0 / (np.linalg.norm(t0) + 1e-12)
        t1 = np.cross(nrm, t0)

        sub = max(p.detail, 1) + 2

        def ray(from_idx, start, tang, reach, born):
            """One filament running out across the surface. Returns
            (tip vertex, mid vertex)."""
            end = start + tang * reach
            # drop the tip back onto the surface, so the splat wraps
            # corners and curved props instead of floating off them
            if bvh is not None:
                h2 = bvh.ray_cast(Vector(end + nrm * reach), Vector(-nrm),
                                  reach * 2.0)
                if h2[0] is not None:
                    end = (np.asarray(h2[0], dtype=np.float64)
                           + np.asarray(h2[1], dtype=np.float64) * 1e-3)
            # the spray reaches the rim a beat after the tip lands
            spread_t = 1.5 * reach / max(p.shot_splat_size, 1e-6)
            prev, mid = from_idx, None
            for s in range(1, sub + 1):
                t = s / sub
                co = start + (end - start) * t
                if s < sub:
                    # silk lifts a little off the wall between anchors
                    co = co + nrm * (math.sin(math.pi * t) * reach * 0.12)
                    co += (np.array([rnd.uniform(-1, 1) for _ in range(3)])
                           * p.jitter * 0.08 * reach)
                vi = add_vert(co, born + spread_t * t, s == sub)
                segs.append([prev, vi])
                prev = vi
                if mid is None and t >= 0.55:
                    mid = vi
            return prev, (mid if mid is not None else prev)

        rays = p.shot_splat
        mids = []
        for j in range(rays):
            ang = ((j + rnd.uniform(-0.4, 0.4)) * 2.0 * math.pi) / rays
            # short filaments crowd the core, a few long spikes reach out
            reach = p.shot_splat_size * (0.35 + 0.65
                                         * rnd.random() ** 1.3)
            tang = t0 * math.cos(ang) + t1 * math.sin(ang)
            tip, mid = ray(hub, center, tang, reach, arrive)
            mids.append(mid)
            # a third of them fork part-way out — the frayed look of silk
            # hitting a wall at speed
            if rnd.random() < 0.35:
                fang = ang + rnd.uniform(0.4, 1.1) * rnd.choice((-1.0, 1.0))
                ftang = t0 * math.cos(fang) + t1 * math.sin(fang)
                ray(mid, verts[mid], ftang,
                    reach * rnd.uniform(0.25, 0.6), times[mid])

        for _ in range(p.shot_splat_web if len(mids) > 1 else 0):
            j = rnd.randrange(len(mids))
            segs.append([mids[j], mids[(j + 1) % len(mids)]])

    nseg = max(6, p.detail * 6)
    # clamped: a .blend saved before Spread's bounds were corrected to
    # radians can still hold a huge value
    half = min(max(p.shot_spread, 0.0), math.pi) * 0.5

    # The emitter fires Bursts volleys of Shots each, Burst Gap frames
    # apart. Every volley is built on its own: its own clot, its own
    # lashing and its own cross threads, from the emitter's location at the
    # moment it goes off — so a moving hand leaves a separate web behind
    # per shot rather than one fan smeared along its path.
    for burst in range(max(p.shot_bursts, 1)):
        b_start = float(p.shot_start) + burst * p.shot_burst_gap
        strands = []            # (vertex indices, arrival frames, angle)

        # ---- pass 1: where every shot starts, aims and lands ---------
        fired = []
        for i in range(p.shot_count):
            fire = b_start + i * p.shot_interval
            if p.shot_interval > 0.0:
                fire += rnd.uniform(-0.35, 0.35) * p.shot_interval * p.jitter
                fire = max(fire, b_start)   # never before the burst starts
            muzzle = _obj_loc_at(emit, fire)
            if muzzle is None:
                muzzle = cursor

            reach = p.shot_range
            if aim is not None:
                # spread the burst over the target's surface instead of
                # converging every strand on its origin — a bundle of
                # lines all through one point is what reads as "fired
                # by a machine"
                if aim_surf is not None:
                    pnt, _n = _sample_surface(rnd, aim_surf[1], aim_surf[2],
                                              aim_surf[3])
                    tgt = pnt + (_obj_loc_at(aim, fire) - aim_now)
                else:
                    tgt = _obj_loc_at(aim, fire)
                d = tgt - muzzle
                dl = float(np.linalg.norm(d))
                axis = d / dl if dl > 1e-9 else _rand_unit(rnd)
                # Range adapts to the target: far enough to reach it (Range
                # on its own strands every shot in mid-air whenever the
                # target sits farther away), and no farther, so the ones the
                # spread throws wide stop level with it instead of sailing
                # metres past. `over` covers the target's own depth — the
                # sampled point can be on its near face while the shot flies
                # at the far one.
                over = 1.15 + min(max(p.shot_spread, 0.0), math.pi) * 0.25
                reach = dl * over
            elif axis0 is not None:
                axis = axis0
            else:
                axis = _rand_unit(rnd)   # no emitter, no target: spray outward
            dirv = _cone_dir(rnd, axis, half)

            hit = (bvh.ray_cast(Vector(muzzle + dirv * 1e-4), Vector(dirv),
                                reach)
                   if bvh is not None else (None, None, None, None))
            if hit[0] is not None:
                hit_n = np.asarray(hit[1], dtype=np.float64)
                end = np.asarray(hit[0], dtype=np.float64) + hit_n * 2e-3
                stick = True
            else:
                hit_n = None
                # `reach`, not Range: with an Aim Target the reach is what
                # the target's distance allows, and a miss that ignored it
                # sailed the full Range straight through the target
                end = muzzle + dirv * reach
                stick = False

            L = float(np.linalg.norm(end - muzzle))
            if L < 1e-4:
                continue
            fired.append((fire, muzzle, dirv, end, stick, hit_n, L))

        if not fired:
            continue        # this volley found nowhere to go; try the next

        # ---- the clot ------------------------------------------------
        # The burst leaves as one mass of web fluid and only opens into
        # separate strands partway to the target, so every strand rides a
        # shared path first and blends out to its own line after that.
        clot = min(max(p.shot_clot, 0.0), 0.95)
        mu_c = np.mean([f[1] for f in fired], axis=0)
        ax_c = np.sum([f[2] for f in fired], axis=0)
        nax = float(np.linalg.norm(ax_c))
        ax_c = ax_c / nax if nax > 1e-9 else fired[0][2]
        L_c = float(np.mean([f[6] for f in fired]))
        up_c = np.array([0.0, 0.0, 1.0]) - ax_c * float(ax_c[2])
        nu_c = float(np.linalg.norm(up_c))
        up_c = (up_c / nu_c if nu_c > 1e-6
                else np.cross(ax_c, np.array([1.0, 0.0, 0.0])))
        arc_c = p.shot_arc * L_c * 0.35
        # the clot has volume: each strand sits at its own spot inside the mass
        clot_r = p.shot_clot_size
        e1_c, e2_c = up_c, np.cross(ax_c, up_c)
        # the rope meanders as a unit — shared by every strand, so the bundle
        # snakes instead of running dead straight like a cable
        mean_amp = L_c * (0.015 + 0.03 * p.shot_whip)
        waves_c = [[(rnd.uniform(0.5, 2.2), rnd.uniform(0.0, 2.0 * math.pi),
                     rnd.uniform(-1.0, 1.0)) for _ in range(2)]
                   for _axis in range(2)]
        twist = p.shot_clot_twist * 2.0 * math.pi
        # clot_s = last segment still fully balled up (opened() is 0 there).
        # Lashing stops at it: one segment further the strands have already
        # begun to separate, and a binder thread there would span metres and
        # haul the opening fan back together.
        clot_s = max(0, min(nseg - 1, int(clot * nseg)))
        open_s = min(nseg - 1, clot_s + 1)

        def opened(t):
            """0 while the shots are still balled together, easing to 1 (fully
            on their own lines) by the time they land."""
            if t <= clot:
                return 0.0
            x = (t - clot) / max(1.0 - clot, 1e-6)
            return x * x * (3.0 - 2.0 * x)

        def rope(t, lob):
            """Centre line of the travelling clot."""
            mean = sum(amp * math.sin(math.pi * t * freq + ph)
                       for freq, ph, amp in waves_c[0]) * 0.5
            mean2 = sum(amp * math.sin(math.pi * t * freq + ph)
                        for freq, ph, amp in waves_c[1]) * 0.5
            env = math.sin(math.pi * min(t / max(clot, 1e-6), 1.0) ** 0.7)
            return (mu_c + ax_c * (t * L_c) + up_c * (lob * arc_c)
                    + e1_c * (mean * mean_amp * env)
                    + e2_c * (mean2 * mean_amp * env))

        # ---- pass 2: build the strands -------------------------------
        for fire, muzzle, dirv, end, stick, hit_n, L in fired:
            flight = (L / speed) * fps           # frames the tip is airborne
            # where this strand sits in the rope's cross-section, and how its
            # braid is phased — sqrt keeps the mass evenly filled
            c_rad = clot_r * math.sqrt(rnd.random())
            c_ph = rnd.uniform(0.0, 2.0 * math.pi)
            # fibres in a real bundle wander in and out, touching and parting
            # again along its length instead of holding a fixed spacing
            f_freq = rnd.uniform(1.5, 4.5)
            f_ph = rnd.uniform(0.0, 2.0 * math.pi)

            # the lash: a perpendicular basis per strand carrying a few waves
            # of different frequency, so shots read as thrown silk rather than
            # laser lines. Both ends are held by the sin(pi t) envelope.
            u = np.cross(dirv, _rand_unit(rnd))
            u = (u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-9
                 else _rand_unit(rnd))
            w = np.cross(dirv, u)
            waves = [[(rnd.uniform(0.6, 3.2), rnd.uniform(0.0, 2.0 * math.pi),
                       rnd.uniform(-1.0, 1.0)) for _ in range(3)]
                     for _axis in range(2)]
            # each strand leans its own way, so a burst fans out in flight
            bow_u, bow_w = rnd.uniform(-1.0, 1.0), rnd.uniform(-1.0, 1.0)

            # the lob: world-up component perpendicular to the flight line, so
            # the strand rides over the straight path and drops onto the hit
            upw = np.array([0.0, 0.0, 1.0]) - dirv * float(dirv[2])
            nu = float(np.linalg.norm(upw))
            upw = upw / nu if nu > 1e-6 else u
            arc_h = p.shot_arc * L * 0.35 * rnd.uniform(0.75, 1.25)
            hook = rnd.uniform(-1.0, 1.0)

            def wander(t, axis):
                v = sum(amp * math.sin(math.pi * t * freq + ph)
                        for freq, ph, amp in waves[axis])
                return v / 3.0

            idx, arrive = [], []
            for s in range(nseg + 1):
                t = s / nseg
                # t**1.8 skews the lob's peak to ~0.68 of the way out: shallow
                # off the muzzle, steep climb, then the drop onto the hit
                lob = math.sin(math.pi * t ** 1.8)
                own = muzzle + (end - muzzle) * t
                if 0 < s < nseg:                 # never shift the two anchors
                    # Everything below pushes the strand sideways off the
                    # flight line, and right before the impact point that
                    # buries the last span in the surface it is landing on.
                    # Taper it out over the final stretch: silk straightens
                    # as it lands anyway.
                    land = min(1.0, (1.0 - t) / 0.2) if stick else 1.0
                    own = own + upw * (lob * arc_h * land) \
                              + u * (lob * arc_h * 0.3 * hook * land)
                    env = (math.sin(math.pi * t) * p.shot_whip * L * 0.12
                           * land)
                    own = own + u * (env * (wander(t, 0) + bow_u * 0.6)) \
                              + w * (env * (wander(t, 1) + bow_w * 0.6))
                    own[2] -= math.sin(math.pi * t) * L * p.shot_slack * land
                    own += (np.array([rnd.uniform(-1, 1) for _ in range(3)])
                            * p.jitter * 0.01 * L * land)
                b = 1.0 if s == nseg else opened(t)
                if b < 1.0:
                    # still (partly) inside the travelling clot: braided round
                    # the rope's axis, with the bundle swelling and pinching
                    # along its length so it reads knotted rather than spun
                    aa = c_ph + twist * t
                    bulge = 1.0 + 0.4 * math.sin(math.pi * t * 3.0 + c_ph)
                    fib = 1.0 + 0.55 * math.sin(f_freq * math.pi * t + f_ph)
                    off = (e1_c * math.cos(aa) + e2_c * math.sin(aa)) \
                        * (c_rad * bulge * fib)
                    co = (rope(t, lob) + off) * (1.0 - b) + own * b
                else:
                    co = own
                if 0 < s < nseg:
                    # after the clot blend, not before: the rope's centre
                    # line runs from the burst's mean muzzle to its mean
                    # landing point and thinks nothing of going straight
                    # through the target. Anchors are left alone — the
                    # muzzle belongs to the emitter, the tip is already
                    # sitting on the surface it hit.
                    co = outside(co)
                pin = (s == 0) or (s == nseg and stick)
                at = fire + flight * t
                idx.append(add_vert(co, at, pin, muzzle_end=(s == 0)))
                arrive.append(at)
                if s:
                    segs.append([idx[-2], idx[-1]])
            # angle around the burst axis, so "neighbouring" strands are known
            # and ring threads can be strung between them
            e2 = np.cross(ax_c, up_c)
            ang_c = math.atan2(float(np.dot(dirv, e2)),
                               float(np.dot(dirv, up_c)))
            strands.append((idx, arrive, ang_c))
            if stick:
                add_splat(idx[-1], end, hit_n, arrive[-1])

        # ---- webbing -------------------------------------------------
        # The strands are the radials; this knits them into something that
        # reads as a spider web rather than a fan of parallel curves:
        #   * ring threads run between angularly neighbouring strands at a
        #     similar distance out, the concentric pass of a real orb web
        #   * bridge threads hop to whichever strand is actually nearest,
        #     sometimes chaining on to a third — the irregular part
        # Nothing is strung inside the clot: while the burst is still one mass
        # a cross thread would just be a chord through overlapping strands, and
        # those repeated chords are what made the geometry look stamped out.
        def link(va, vb, extra=0.0):
            """Silk between two existing points, sagging under its own weight.
            Born once both ends exist."""
            A, B = verts[va], verts[vb]
            span = float(np.linalg.norm(B - A))
            if span < 1e-4:
                return
            born = max(times[va], times[vb]) + extra
            sub = max(p.detail, 1) + 1
            droop = span * max(p.shot_slack, 0.04) * rnd.uniform(0.6, 2.2)
            prev = va
            inner = []
            for s in range(1, sub):
                t = s / sub
                co = A + (B - A) * t
                co[2] -= math.sin(math.pi * t) * droop
                co += (np.array([rnd.uniform(-1, 1) for _ in range(3)])
                       * p.jitter * 0.03 * span)
                co = outside(co)
                ni = add_vert(co, born)
                segs.append([prev, ni])
                prev = ni
                inner.append(ni)
            segs.append([prev, vb])
            # loose fibres trailing off the thread — every junction in a real
            # web frays, and it is most of what makes one read as silk
            if inner and rnd.random() < 0.4:
                root = rnd.choice(inner)
                d = _rand_unit(rnd) * span * rnd.uniform(0.06, 0.22)
                prev2 = root
                for k in range(2):
                    co = verts[root] + d * ((k + 1) * 0.5)
                    co[2] -= span * 0.015 * (k + 1)
                    nv = add_vert(outside(co), times[root])
                    segs.append([prev2, nv])
                    prev2 = nv

        # ---- rope: lash the clot together ----------------------------
        # Bunching the strands at build time is not enough — once the
        # solver runs, gravity and the constraints pull them apart into a
        # ribbon. Short binder threads between neighbours at every clot
        # segment (plus a few chords across the bundle) hold the mass
        # together as one rope, and since silk only pulls they never push
        # the strands apart.
        binders = []          # edges the solver must not tear
        if len(strands) > 1 and clot_s > 1:
            ring = sorted(range(len(strands)), key=lambda k: strands[k][2])

            # Whipping: threads spiralling around the outside of the bundle,
            # tied to whichever fibre they pass. This is how silk (and rope)
            # actually holds together, and it reads as one cord. A ladder of
            # straight rungs between neighbours binds just as well but renders
            # as the cross-bars of a cable harness.
            for _wi in range(max(2, len(strands) // 6)):
                ph = rnd.uniform(0.0, 2.0 * math.pi)
                turns = p.shot_clot_twist + rnd.uniform(1.5, 4.0)
                rad_k = rnd.uniform(1.02, 1.3)
                prev = None
                for s in range(0, clot_s + 1):
                    t = s / nseg
                    aa = ph + turns * 2.0 * math.pi * (t / max(clot, 1e-6))
                    lob = math.sin(math.pi * t ** 1.8)
                    co = rope(t, lob) + (
                        e1_c * math.cos(aa)
                        + e2_c * math.sin(aa)) * (clot_r * rad_k)
                    born = times[strands[ring[0]][0][s]]
                    vi = add_vert(co, born, s == 0, muzzle_end=(s == 0))
                    if prev is not None:
                        segs.append([prev, vi])
                    prev = vi
                    # bite onto the nearest fibre — that tie is what binds
                    if s % 2 == 0:
                        best, bd = None, 1e18
                        for a in ring:
                            va = strands[a][0][s]
                            d = float(np.linalg.norm(verts[va] - co))
                            if d < bd:
                                best, bd = va, d
                        if best is not None:
                            segs.append([vi, best])
                            binders.append((vi, best))

            # Ties between neighbouring fibres at every station. These are
            # what actually holds the cord: they are short, so radial
            # separation stretches them immediately. Diagonal ties along the
            # bundle look nicer but barely constrain anything — their length
            # is dominated by the axial span, so fibres drift centimetres
            # before they pull. Keeping Clot Thickness small hides them.
            for pos, a in enumerate(ring):
                b_i = ring[(pos + 1) % len(ring)]
                if a == b_i:
                    continue
                for s in range(1, clot_s + 1):
                    va, vb = strands[a][0][s], strands[b_i][0][s]
                    segs.append([va, vb])
                    binders.append((va, vb))

            # chords straight across the bundle at every station. Neighbour
            # ties alone leave the cross-section free to inflate — each
            # fibre only feels its two neighbours, and 5% of slack per link
            # adds up around the ring. A chord spans the diameter, so it
            # caps the thickness.
            # (`opp`, not `half`: the spread half-angle above is still needed
            # by the next burst)
            opp = max(len(ring) // 2, 1)
            for s in range(1, clot_s + 1):
                for k in rnd.sample(range(len(ring)),
                                    min(len(ring), max(2, len(ring) // 3))):
                    a = ring[k]
                    b_i = ring[(k + opp) % len(ring)]
                    if a == b_i:
                        continue
                    va, vb = strands[a][0][s], strands[b_i][0][s]
                    segs.append([va, vb])
                    binders.append((va, vb))

        if len(strands) > 1 and p.shot_tangle > 0:
            # only the opened-out part of each strand can carry webbing
            cand = list(range(open_s, nseg))
            by_angle = sorted(range(len(strands)), key=lambda k: strands[k][2])

            if cand:
                # pool of attachable points for the nearest-neighbour search
                pool_v, pool_owner = [], []
                for si, (idx_s, _arr, _ang) in enumerate(strands):
                    for s in cand:
                        pool_v.append(idx_s[s])
                        pool_owner.append(si)
                P = np.stack([verts[v] for v in pool_v])
                owner = np.asarray(pool_owner)

                rings = int(round(p.shot_tangle * 0.55))
                for k in range(p.shot_tangle):
                    if k < rings:
                        # concentric: neighbour strand, similar distance out
                        j = rnd.randrange(len(by_angle))
                        a = by_angle[j]
                        b = by_angle[(j + 1) % len(by_angle)]
                        if a == b:
                            continue
                        sa = rnd.choice(cand)
                        sb = min(max(sa + rnd.randint(-1, 1), cand[0]),
                                 cand[-1])
                        link(strands[a][0][sa], strands[b][0][sb],
                             rnd.uniform(0.0, 1.5))
                    else:
                        # bridge: reach for the nearest point on another strand
                        src = rnd.randrange(len(pool_v))
                        d = np.linalg.norm(P - P[src], axis=1)
                        d[owner == owner[src]] = np.inf
                        if not np.isfinite(d).any():
                            continue
                        near = np.argsort(d)[:3]
                        dst = int(rnd.choice(near))
                        link(pool_v[src], pool_v[dst], rnd.uniform(0.0, 1.5))
                        # a third of them carry on to yet another strand, which
                        # is what builds junctions instead of isolated chords
                        if rnd.random() < 0.35:
                            d2 = np.linalg.norm(P - P[dst], axis=1)
                            d2[owner == owner[dst]] = np.inf
                            d2[owner == owner[src]] = np.inf
                            if np.isfinite(d2).any():
                                link(pool_v[dst],
                                     pool_v[int(np.argmin(d2))],
                                     rnd.uniform(0.0, 2.0))

    if not segs:
        return None

    bm = bmesh.new()
    bverts = [bm.verts.new(tuple(co)) for co in verts]
    for ia, ib in segs:
        if ia == ib:
            continue
        try:
            bm.edges.new((bverts[ia], bverts[ib]))
        except ValueError:
            pass
    bm.verts.index_update()
    mesh_idx = [v.index for v in bverts]

    me = bpy.data.meshes.new("Web Shot")
    bm.to_mesh(me)
    bm.free()

    pin_attr = me.attributes.new(A_PIN, 'BOOLEAN', 'POINT')
    for i in pinned:
        pin_attr.data[mesh_idx[i]].value = True
    tarr = np.zeros(len(verts), dtype=np.float32)
    for i, mi in enumerate(mesh_idx):
        tarr[mi] = times[i]
    me.attributes.new(A_SHOT, 'FLOAT', 'POINT').data.foreach_set(
        "value", tarr)

    # Muzzle anchors ride the emitter. The GPU solver carries them with its
    # transform from the frame they fire onward — the same mechanism that
    # keeps an impact point stuck to the geometry it hit — so the strands
    # trail a moving emitter instead of hanging where they left it.
    if emit is not None and p.shot_stick_emitter and muzzles:
        marr = np.zeros(len(verts), dtype=np.bool_)
        for i in muzzles:
            marr[mesh_idx[i]] = True
        me.attributes.new(A_EMIT, 'BOOLEAN', 'POINT').data.foreach_set(
            "value", marr)
        me[P_EMITTER] = emit.name

    # The threads binding the clot's fibres together are only millimetres
    # long, so any jostle passes the tear threshold in relative terms and
    # the cord bursts apart. Flag them unbreakable.
    if binders:
        want = {frozenset((mesh_idx[a], mesh_idx[b])) for a, b in binders}
        flags = np.zeros(len(me.edges), dtype=np.bool_)
        for ei, e in enumerate(me.edges):
            if frozenset((e.vertices[0], e.vertices[1])) in want:
                flags[ei] = True
        me.attributes.new(A_NOTEAR, 'BOOLEAN', 'EDGE').data.foreach_set(
            "value", flags)

    return me, "Web Shot"


class ARN_OT_generate_web(Operator):
    """Generate a natural orb web (anchor endpoints pre-pinned)"""
    bl_idname = "arachne.generate_web"
    bl_label = "Generate Web"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.swf_web
        env = [o for o in context.selected_objects
               if o.type == 'MESH' and not o.get("swf_web")]
        obj = build_web_object(context, p, env)
        if obj is None:
            self.report({'ERROR'},
                        "Nothing to build — Chaotic Cobweb needs selected "
                        "mesh geometry to anchor to (select a corner/prop, "
                        "then generate); Web Shot needs at least one shot "
                        "that travels a nonzero distance.")
            return {'CANCELLED'}
        # remember the anchor geometry and track this web so Live Update
        # can rebuild it in place as parameters change
        obj["swf_env"] = json.dumps([o.name for o in env])
        p.live_obj = obj
        return {'FINISHED'}


classes = (ARN_WebProps, ARN_OT_generate_web)


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
    # a pointer property referencing a ghost class blocks its eviction —
    # clear the pointer first, then classes can be safely re-registered
    if hasattr(bpy.types.Scene, "swf_web"):
        try:
            del bpy.types.Scene.swf_web
        except Exception:
            pass
    for c in classes:
        _safe_register(c)
    bpy.types.Scene.swf_web = bpy.props.PointerProperty(type=ARN_WebProps)


def unregister():
    del bpy.types.Scene.swf_web
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
