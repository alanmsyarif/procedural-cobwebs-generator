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

import math
import random

import numpy as np

import bpy
import bmesh
from mathutils.bvhtree import BVHTree
from bpy.props import (
    IntProperty, FloatProperty, EnumProperty,
)
from bpy.types import Operator, PropertyGroup

from .constants import A_PIN


class SWF_WebProps(PropertyGroup):
    mode: EnumProperty(
        name="Mode",
        items=[('ORB', "Orb Web", "Classic radial/spiral orb web"),
               ('CHAOS', "Chaotic Cobweb",
                "Spider-spun corner cobweb anchored to selected meshes "
                "(Pixar / Thomas Kole construction)")],
        default='ORB')
    cobweb_initial: IntProperty(
        name="Initial Lines", default=36, min=2, max=200,
        description="Anchor threads cast between the selected surfaces "
                    "before spinning starts")
    cobweb_spiders: IntProperty(
        name="Spiders", default=6, min=1, max=32,
        description="Concurrent spinners (Pixar used 5-10)")
    cobweb_steps: IntProperty(
        name="Spin Steps", default=600, min=10, max=3000,
        description="Total threads spun (Pixar used 50-1000)")
    cobweb_spread: FloatProperty(
        name="Spread", default=0.6, min=0.0, max=1.0,
        description="0 = spiders knit dense local clumps, 1 = spinning "
                    "distributes uniformly across the whole volume "
                    "(spiders relocate often and take long bridging jumps)")
    cobweb_jump: FloatProperty(
        name="Jump Distance", default=0.4, min=0.01, max=10.0,
        subtype='DISTANCE',
        description="Max distance a spider jumps per step — larger is "
                    "more chaotic, smaller is denser")
    radials: IntProperty(
        name="Radials", default=16, min=3, max=64,
        description="Number of radial threads")
    rings: IntProperty(
        name="Spiral Turns", default=14, min=2, max=60,
        description="Number of turns in the capture spiral")
    radius: FloatProperty(
        name="Radius", default=1.0, min=0.05, max=50.0,
        subtype='DISTANCE', description="Web radius")
    hub_factor: FloatProperty(
        name="Hub Size", default=0.08, min=0.01, max=0.5,
        description="Hub radius as a fraction of the web radius")
    jitter: FloatProperty(
        name="Irregularity", default=0.3, min=0.0, max=1.0,
        description="Angular drift, spacing unevenness and positional noise")
    spiral_sag: FloatProperty(
        name="Spiral Sag", default=0.3, min=0.0, max=1.0,
        description="How much spiral threads droop into scallops "
                    "between radials")
    damage: FloatProperty(
        name="Damage", default=0.15, min=0.0, max=0.9,
        description="Fraction of spiral segments missing (radials break "
                    "at a lower rate)")
    asymmetry: FloatProperty(
        name="Asymmetry", default=0.25, min=0.0, max=1.0,
        description="Smooth variation of the web radius around the circle")
    tangles: IntProperty(
        name="Tangle Threads", default=8, min=0, max=40,
        description="Slack chaotic threads drooping across the web")
    detail: IntProperty(
        name="Detail", default=2, min=1, max=5,
        description="Sub-points per thread span (sag resolution for the "
                    "solver and the scallops)")
    anchors: IntProperty(
        name="Anchor Threads", default=5, min=1, max=16,
        description="Number of anchor threads extended past the rim")
    anchor_extend: FloatProperty(
        name="Anchor Length", default=0.35, min=0.05, max=3.0,
        description="Anchor thread length as a fraction of the web radius")
    seed: IntProperty(name="Seed", default=0, min=0)
    plane: EnumProperty(
        name="Plane",
        items=[('XZ', "XZ (vertical)",
                "Vertical web — sags naturally under -Z gravity"),
               ('XY', "XY (horizontal)", "Flat web in the ground plane")],
        default='XZ')


def build_web_object(context, p, env_objs=None):
    """Create the web object from properties. Returns the object,
    or None (with a reason string) for invalid setups."""
    if p.mode == 'CHAOS':
        return _build_cobweb(context, p, env_objs or [])
    return _build_orb(context, p)


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

    return _finalize(context, me, "SpiderWeb")


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
    span = p.radius * 2.0
    attempts = 0
    while (sum(1 for _ in segs) < p.cobweb_initial
           and attempts < p.cobweb_initial * 40):
        attempts += 1
        a, na = _sample_surface(rnd, V, T, cum)
        d = _rand_unit(rnd)
        if d.dot(na) < 0.0:
            d = -d                            # keep in outward hemisphere
        if rnd.random() < 0.5:
            d = d + na * 1.2                  # half strongly normal-biased
        d = d / (np.linalg.norm(d) + 1e-12)
        hit = bvh.ray_cast(Vector(a + na * 1e-4), Vector(d), span)
        if hit[0] is None:
            # normals may face the other way (walls, flipped winding):
            # retry with the normal component of the direction flipped
            d2 = d - na * (2.0 * d.dot(na))
            hit = bvh.ray_cast(Vector(a - na * 1e-4), Vector(d2), span)
        if hit[0] is None:
            continue
        b = np.asarray(hit[0], dtype=np.float64)
        if np.linalg.norm(b - a) < span * 0.05:
            continue
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
        pi = spiders[si]
        P = verts[pi]
        placed = False
        for _try in range(6):
            jump = p.cobweb_jump * (2.5 if rnd.random() < long_jump_p
                                    else 1.0)
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

    return _finalize(context, me, "Cobweb")


class SWF_OT_generate_web(Operator):
    """Generate a natural orb web (anchor endpoints pre-pinned)"""
    bl_idname = "swf.generate_web"
    bl_label = "Generate Web"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.swf_web
        env = [o for o in context.selected_objects if o.type == 'MESH']
        obj = build_web_object(context, p, env)
        if obj is None:
            self.report({'ERROR'},
                        "Chaotic Cobweb needs selected mesh geometry to "
                        "anchor to (select a corner/prop, then generate).")
            return {'CANCELLED'}
        return {'FINISHED'}


classes = (SWF_WebProps, SWF_OT_generate_web)


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
    bpy.types.Scene.swf_web = bpy.props.PointerProperty(type=SWF_WebProps)


def unregister():
    del bpy.types.Scene.swf_web
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
