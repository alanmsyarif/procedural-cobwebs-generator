# Native Blender GPU backend — GLSL compute via Blender's `gpu` module.
# No external dependencies.
#
# Physics (Pixar "Dust and Cobwebs for Toy Story 4" / Thomas Kole adaptions):
# verlet + world-space gravity/wind + turbulence, unilateral silk
# constraints (gather architecture, valence-averaged, SOR), tension slack,
# deteriorate, pre-warm, threshold tearing.
#
# Collision, two shapes:
#   * SPHERE   — collider approximated by its bounding sphere (fast).
#   * MESH SDF — the collider mesh is baked into a signed-distance-field
#     3D texture at sim start (BVH nearest queries in web-local space).
#     The solve kernel samples it with trilinear filtering; the field
#     gradient supplies the surface normal. Animated collider LOCATION is
#     tracked per frame as an offset; rotation/deformation after the bake
#     is frozen. Collider should be a closed mesh (the inside/outside
#     sign relies on consistent normals).
#
# Push-constant budget: 8 x vec4 = 128 bytes (the guaranteed minimum).
# Point/edge counts and SDF resolution are compile-time #defines.
#   p1 = (dt2, damping, turbulence, time)
#   p2 = (gravity_local.xyz, sdf_delta.z)
#   p3 = (wind_local.xyz, sor)
#   p4 = (sphere.xyz, sphere_radius)
#   p5 = (collision_offset, friction, tear_threshold, tearing_on)
#   p6 = (resist_compression, sdf_on, stickiness, latch_state)
#   p7 = (sdf_box_min.xyz, sdf_delta.x)
#   p8 = (sdf_inv_cell.xyz, sdf_delta.y)
# The follow kernel reuses p1..p3 as the rows of a 4x4 transform.
#
# pos.w is the pin state: 0 free, 1 pinned in place, 2 pinned AND carried
# along by the collider (see _FOLLOW). Every ">0.5" test means "pinned".

import random

import numpy as np

import bpy
from mathutils import Vector

from .constants import (A_PIN, A_SHOT, A_NOTEAR, A_GPU_POS, A_BROKEN,
                        A_TENSION)

_W = 1024
_BROKEN_FLAG = False


def native_available():
    try:
        import gpu
        return hasattr(gpu, "compute") and hasattr(gpu.types,
                                                   "GPUShaderCreateInfo")
    except Exception:
        return False


def native_broken():
    return _BROKEN_FLAG


def _mark_broken(ex):
    global _BROKEN_FLAG
    _BROKEN_FLAG = True
    print("Arachne native GPU backend disabled after error:", ex)


def _clear_broken():
    global _BROKEN_FLAG
    _BROKEN_FLAG = False


# ---------------------------------------------------------------------------
#  GLSL sources
# ---------------------------------------------------------------------------

_COMMON = """
ivec2 texel(int i) { return ivec2(i % WIDTH, i / WIDTH); }

float n1(vec3 p, float t, float seed) {
    float s = sin(dot(p, vec3(12.9898, 78.233, 37.719)) + t * 2.1 + seed)
              * 43758.5453;
    return fract(s) * 2.0 - 1.0;
}
"""

_SDF_FUNCS = """
float sdf_tap(ivec3 c) {
    c = clamp(c, ivec3(0), ivec3(R_SDF - 1));
    return imageLoad(sdf, c).r;
}

float sdf_tri(vec3 cc) {
    vec3 f = floor(cc);
    vec3 t = cc - f;
    ivec3 i0 = ivec3(f);
    float c00 = mix(sdf_tap(i0 + ivec3(0,0,0)), sdf_tap(i0 + ivec3(1,0,0)), t.x);
    float c10 = mix(sdf_tap(i0 + ivec3(0,1,0)), sdf_tap(i0 + ivec3(1,1,0)), t.x);
    float c01 = mix(sdf_tap(i0 + ivec3(0,0,1)), sdf_tap(i0 + ivec3(1,0,1)), t.x);
    float c11 = mix(sdf_tap(i0 + ivec3(0,1,1)), sdf_tap(i0 + ivec3(1,1,1)), t.x);
    return mix(mix(c00, c10, t.y), mix(c01, c11, t.y), t.z);
}
"""

_INTEGRATE = _COMMON + """
void main() {
    int i = int(gl_GlobalInvocationID.y) * WIDTH
          + int(gl_GlobalInvocationID.x);
    if (i >= N_POINTS) { return; }
    vec4 P4 = imageLoad(posA, texel(i));
    vec3 P = P4.xyz;
    /* prevI.w carries the point's birth time in seconds — the moment the
       flying tip of a Web Shot strand reaches it. Every prevI store must
       preserve it. Other web types carry -1e9, i.e. born before the sim
       started. */
    vec4 PV = imageLoad(prevI, texel(i));
    if (P4.w > 0.5) { imageStore(prevI, texel(i), vec4(P, PV.w)); return; }
    /* Unfired: held frozen with zero velocity, so a strand enters the sim
       taut at its fire pose instead of having sagged for all the frames
       it spent waiting to be shot. */
    if (PV.w > p1.w) { imageStore(prevI, texel(i), vec4(P, PV.w)); return; }
    vec3 pv = PV.xyz;
    vec3 vel = (P - pv) * p1.y;
    vec3 nse = vec3(n1(P, p1.w, 0.0), n1(P, p1.w, 17.0), n1(P, p1.w, 39.0));
    vec3 f = p2.xyz + p3.xyz + nse * p1.z;
    imageStore(prevI, texel(i), vec4(P, PV.w));
    imageStore(posA, texel(i), vec4(P + vel + f * p1.x, P4.w));
}
"""

_SOLVE = _COMMON + _SDF_FUNCS + """
void main() {
    int i = int(gl_GlobalInvocationID.y) * WIDTH
          + int(gl_GlobalInvocationID.x);
    if (i >= N_POINTS) { return; }
    vec4 P4 = imageLoad(posIn, texel(i));
    vec3 P = P4.xyz;
    if (P4.w > 0.5) { imageStore(posOut, texel(i), P4); return; }
    /* prevI.w = birth time (see the integrate stage): a Web Shot point the
       flying tip has not reached yet takes no constraint or collision */
    if (imageLoad(prevI, texel(i)).w > p1.w) {
        imageStore(posOut, texel(i), P4);
        return;
    }
    float outw = P4.w;   /* may latch to a pin (1.0) on sticky contact */

    vec2 off2 = imageLoad(incOff, texel(i)).xy;
    int start = int(off2.x); int cnt = int(off2.y);
    vec3 acc = vec3(0.0);
    int alive = 0;
    for (int k = 0; k < cnt; k++) {
        int e = int(imageLoad(incLst, texel(start + k)).x);
        vec4 E = imageLoad(edges, texel(e));
        if (E.w > 0.5) { continue; }
        alive += 1;
        int ia = int(E.x); int ib = int(E.y);
        int other = (ia == i) ? ib : ia;
        vec3 O = imageLoad(posIn, texel(other)).xyz;
        vec3 d = O - P;
        float len = length(d);
        if (len > 1e-9) {
            float stretch = len - E.z;
            if (stretch > 0.0 || p6.x > 0.5) {   /* unilateral silk */
                acc += d * (stretch / len * 0.5);
            }
        }
    }
    if (alive > 0) { P += acc / float(alive) * p3.w; }

    float coff = p5.x;
    if (p6.y > 0.5) {
        /* MESH SDF collision: sample field at the point compensated for
           the collider's motion since the bake */
        vec3 q = P - vec3(p7.w, p8.w, p2.w);
        vec3 cc = (q - p7.xyz) * p8.xyz;
        if (all(greaterThan(cc, vec3(1.0)))
                && all(lessThan(cc, vec3(float(R_SDF) - 2.0)))) {
            float d = sdf_tri(cc);
            if (d < coff) {
                float e = 1.0;
                vec3 grad = vec3(
                    sdf_tri(cc + vec3(e,0,0)) - sdf_tri(cc - vec3(e,0,0)),
                    sdf_tri(cc + vec3(0,e,0)) - sdf_tri(cc - vec3(0,e,0)),
                    sdf_tri(cc + vec3(0,0,e)) - sdf_tri(cc - vec3(0,0,e)));
                vec3 gi = grad * p8.xyz;
                /* a healthy SDF gradient has magnitude ~1; near the
                   field's apex trilinear flattening makes it tiny and
                   direction-noisy — use a deterministic escape push
                   there and let later iterations refine it */
                vec3 n = (dot(gi, gi) > 0.0625) ? normalize(gi)
                                                : vec3(0.0, 0.0, 1.0);
                vec3 target = P + n * (coff - d);
                vec4 pv4 = imageLoad(prevI, texel(i));
                vec3 pv = pv4.xyz;
                /* stickiness p6.z: a per-point fraction of contacts
                   latch to the surface (become pins), the rest just
                   slide/settle with friction — sticky silk catching */
                float adhere = 0.0;
                if (p6.z > 0.0) {
                    float hr = fract(sin(float(i) * 12.9898 + 4.1)
                                     * 43758.5453);
                    if (hr < p6.z) { adhere = 1.0; }
                }
                float pb = mix(p5.y, 1.0, adhere);
                imageStore(prevI, texel(i),
                           vec4(pv + (target - pv) * pb, pv4.w));
                P = target;
                outw = max(outw, adhere * p6.w);
            }
        }
    } else if (p4.w > 0.0) {
        /* bounding-sphere collision */
        vec3 c = p4.xyz;
        vec3 d = P - c;
        float len = length(d);
        float rr = p4.w + coff;
        if (len < rr) {
            vec3 nrm = d / max(len, 1e-9);
            vec3 target = c + nrm * rr;
            vec4 pv4 = imageLoad(prevI, texel(i));
            vec3 pv = pv4.xyz;
            float adhere = 0.0;
            if (p6.z > 0.0) {
                float hr = fract(sin(float(i) * 12.9898 + 4.1)
                                 * 43758.5453);
                if (hr < p6.z) { adhere = 1.0; }
            }
            float pb = mix(p5.y, 1.0, adhere);
            imageStore(prevI, texel(i),
                       vec4(pv + (target - pv) * pb, pv4.w));
            P = target;
            outw = max(outw, adhere * p6.w);
        }
    }
    imageStore(posOut, texel(i), vec4(P, outw));
}
"""

# Stuck points ride the collider: each frame the collider's incremental
# rigid motion (rotation included) is applied to points whose pin state is
# 2, so a web that latched onto a moving object travels with it instead of
# hanging in the air where it stuck. p1..p3 are the rows of that transform,
# expressed in the web's local space.
_FOLLOW = _COMMON + """
void main() {
    int i = int(gl_GlobalInvocationID.y) * WIDTH
          + int(gl_GlobalInvocationID.x);
    if (i >= N_POINTS) { return; }
    vec4 P4 = imageLoad(posA, texel(i));
    /* p4.x selects the group: state 2+k is stuck to collider k, so each
       collection member is dispatched with its own transform */
    if (abs(P4.w - p4.x) > 0.25) { return; }
    vec3 P = P4.xyz;
    imageStore(posA, texel(i), vec4(dot(p1.xyz, P) + p1.w,
                                    dot(p2.xyz, P) + p2.w,
                                    dot(p3.xyz, P) + p3.w, P4.w));
    /* carry the verlet history too, or the point reads as having been
       teleported and fires off at collider speed once released */
    vec4 PV = imageLoad(prevI, texel(i));
    imageStore(prevI, texel(i), vec4(dot(p1.xyz, PV.xyz) + p1.w,
                                     dot(p2.xyz, PV.xyz) + p2.w,
                                     dot(p3.xyz, PV.xyz) + p3.w, PV.w));
}
"""

_TEAR = _COMMON + """
void main() {
    int e = int(gl_GlobalInvocationID.y) * WIDTH
          + int(gl_GlobalInvocationID.x);
    if (e >= M_EDGES) { return; }
    vec4 E = imageLoad(edges, texel(e));
    /* E.w: >0.5 already torn, <-0.5 flagged unbreakable (binder threads
       inside a web-shot clot are millimetres long, so ordinary relative
       stretch snaps them instantly and the bundle falls apart) */
    if (E.w > 0.5 || E.w < -0.5) { return; }
    vec3 A = imageLoad(posA, texel(int(E.x))).xyz;
    vec3 B = imageLoad(posA, texel(int(E.y))).xyz;
    float len = length(B - A);
    if (p5.w > 0.5 && len > E.z * p5.z) {
        imageStore(edges, texel(e), vec4(E.xyz, 1.0));
    }
}
"""

_TENSION = _COMMON + """
void main() {
    int i = int(gl_GlobalInvocationID.y) * WIDTH
          + int(gl_GlobalInvocationID.x);
    if (i >= N_POINTS) { return; }
    vec2 off2 = imageLoad(incOff, texel(i)).xy;
    int start = int(off2.x); int cnt = int(off2.y);
    vec3 P = imageLoad(posA, texel(i)).xyz;
    float tmax = 0.0;
    for (int k = 0; k < cnt; k++) {
        int e = int(imageLoad(incLst, texel(start + k)).x);
        vec4 E = imageLoad(edges, texel(e));
        if (E.w > 0.5) { continue; }
        int ia = int(E.x); int ib = int(E.y);
        int other = (ia == i) ? ib : ia;
        vec3 O = imageLoad(posA, texel(other)).xyz;
        float ratio = length(O - P) / max(E.z, 1e-8);
        float t = (ratio - 1.0) / max(p5.z - 1.0, 0.01);
        tmax = max(tmax, clamp(t, 0.0, 1.0));
    }
    imageStore(tens, texel(i), vec4(tmax, 0.0, 0.0, 0.0));
}
"""

_PUSH_NAMES = ("p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8")


# ---------------------------------------------------------------------------
#  Texture helpers
# ---------------------------------------------------------------------------

def _tex(gpu, count, channels, data=None):
    h = max((count + _W - 1) // _W, 1)
    fmt = {1: 'R32F', 2: 'RG32F', 4: 'RGBA32F'}[channels]
    if data is not None:
        pad = np.zeros((_W * h, channels), np.float32)
        pad[:count] = data.reshape(count, channels)
        buf = gpu.types.Buffer('FLOAT', (h, _W, channels),
                               pad.reshape(h, _W, channels))
        return gpu.types.GPUTexture((_W, h), format=fmt, data=buf)
    return gpu.types.GPUTexture((_W, h), format=fmt)


def _tex3d(gpu, res, data):
    buf = gpu.types.Buffer('FLOAT', (res, res, res),
                           np.ascontiguousarray(data, np.float32))
    return gpu.types.GPUTexture((res, res, res), format='R32F', data=buf)


def _read(tex, count, channels):
    buf = tex.read()
    try:
        arr = np.asarray(buf, dtype=np.float32)
    except Exception:
        arr = np.array(buf.to_list(), dtype=np.float32)
    return arr.reshape(-1, channels)[:count]


def _shader(gpu, source, images, n, m, sdf_res):
    info = gpu.types.GPUShaderCreateInfo()
    info.define("WIDTH", str(_W))
    info.define("N_POINTS", str(max(n, 1)))
    info.define("M_EDGES", str(max(m, 1)))
    info.define("R_SDF", str(max(sdf_res, 1)))
    for slot, (fmt, ttype, name) in enumerate(images):
        info.image(slot, fmt, ttype, name, qualifiers={"READ", "WRITE"})
    for pname in _PUSH_NAMES:
        info.push_constant('VEC4', pname)
    info.local_group_size(8, 8)
    info.compute_source(source)
    return gpu.shader.create_from_info(info)


# ---------------------------------------------------------------------------
#  SDF bake (CPU, once per sim reset)
# ---------------------------------------------------------------------------

def _collect_colliders(g):
    """Mesh objects to collide against: every mesh in the collision
    collection if one is set, otherwise the single collider object."""
    coll = getattr(g, "collider_collection", None)
    if coll is not None:
        return [o for o in coll.all_objects
                if o.type == 'MESH' and not o.get("swf_web")]
    if g.collider is not None and g.collider.type == 'MESH':
        return [g.collider]
    return []


def _assign_collider(web_obj, colliders, pos, mask, offset):
    """For every point flagged in `mask`, which collider it is sitting on
    (index into `colliders`), or -1 for none. Run once at bind time to
    decide which object each anchor rides when things start moving."""
    from mathutils.bvhtree import BVHTree
    out = np.full(len(pos), -1, np.int32)
    if not mask.any() or not colliders:
        return out
    deps = bpy.context.evaluated_depsgraph_get()
    idx = np.flatnonzero(mask)
    best = np.full(len(pos), np.inf)
    for k, coll in enumerate(colliders):
        ob = coll.evaluated_get(deps)
        me = ob.to_mesh()
        me.calc_loop_triangles()
        M = web_obj.matrix_world.inverted_safe() @ coll.matrix_world
        verts = [tuple(M @ v.co) for v in me.vertices]
        tris = [tuple(lt.vertices) for lt in me.loop_triangles]
        ob.to_mesh_clear()
        if not tris:
            continue
        bvh = BVHTree.FromPolygons(verts, tris)
        V = np.asarray(verts)
        diag = float(np.linalg.norm(V.max(0) - V.min(0)))
        thresh = max(offset * 3.0, diag * 0.02, 1e-4)
        for i in idx:
            co, _n, _t, d = bvh.find_nearest(Vector(pos[i]), thresh * 4.0)
            if co is not None and d is not None and d <= thresh                     and d < best[i]:
                best[i] = d
                out[i] = k
    return out


def _bake_sdf(gpu, web_obj, colliders, res):
    """Signed distance field of the collider(s), in web-local space,
    sampled on a res^3 grid via BVH nearest queries. Multiple colliders
    (a collection) are merged into one field."""
    from mathutils.bvhtree import BVHTree
    deps = bpy.context.evaluated_depsgraph_get()
    winv = web_obj.matrix_world.inverted()
    verts = []
    tris = []
    for coll in colliders:
        ob = coll.evaluated_get(deps)
        me = ob.to_mesh()
        me.calc_loop_triangles()
        M = winv @ coll.matrix_world
        base = len(verts)
        verts.extend(tuple(M @ v.co) for v in me.vertices)
        tris.extend(tuple(base + i for i in lt.vertices)
                    for lt in me.loop_triangles)
        ob.to_mesh_clear()
    if not tris:
        return None
    bvh = BVHTree.FromPolygons(verts, tris)

    V = np.asarray(verts)
    bmin = V.min(0)
    bmax = V.max(0)
    pad = float((bmax - bmin).max()) * 0.15 + 0.05
    bmin = bmin - pad
    bmax = bmax + pad

    print("Arachne: baking collider SDF (%d^3)..." % res)
    axes = [np.linspace(bmin[k], bmax[k], res) for k in range(3)]
    dist = np.full((res, res, res), 1e3, np.float32)
    for iz in range(res):
        z = axes[2][iz]
        for iy in range(res):
            y = axes[1][iy]
            for ix in range(res):
                p = Vector((axes[0][ix], y, z))
                co, nrm, _idx, d = bvh.find_nearest(p)
                if co is None:
                    continue
                sgn = 1.0 if (p - co).dot(nrm) >= 0.0 else -1.0
                dist[iz, iy, ix] = sgn * d
    print("Arachne: SDF bake done.")

    inv_cell = (res - 1) / np.maximum(bmax - bmin, 1e-9)
    # location tracking only makes sense for a single rigid collider;
    # a collection is treated as static (baked once)
    bake_loc = np.array(web_obj.matrix_world.inverted()
                        @ colliders[0].matrix_world.translation)
    return {
        "tex": _tex3d(gpu, res, dist),
        "bmin": bmin.astype(np.float64),
        "inv_cell": inv_cell.astype(np.float64),
        "bake_loc": bake_loc,
        "res": res,
    }


# ---------------------------------------------------------------------------
#  Simulation state
# ---------------------------------------------------------------------------

def apply_arrays(obj, pos, brk, tens):
    """Write position/broken/tension arrays into mesh attributes.
    Pure CPU — safe to call from render-thread frame handlers, where no
    GPU context exists (used to replay the cached sim during renders)."""
    from .gpu_solver import _ensure_attr
    me = obj.data
    _ensure_attr(me, A_GPU_POS, 'FLOAT_VECTOR', 'POINT')
    _ensure_attr(me, A_BROKEN, 'BOOLEAN', 'EDGE')
    _ensure_attr(me, A_TENSION, 'FLOAT', 'POINT')
    me.attributes[A_GPU_POS].data.foreach_set("vector", pos)
    me.attributes[A_BROKEN].data.foreach_set("value", brk)
    me.attributes[A_TENSION].data.foreach_set("value", tens)
    me.update_tag()


class NativeState:
    """Simulation state: step() advances physics, write_back() -> mesh."""

    def __init__(self, obj, g):
        import gpu
        self._gpu = gpu
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

        pin = np.zeros(n, np.float32)
        a = me.attributes.get(A_PIN)
        if a is not None and a.domain == 'POINT':
            tmp = np.zeros(n, np.bool_)
            a.data.foreach_get("value", tmp)
            pin = tmp.astype(np.float32)

        # tension slack (Kole): rest lengths longer than built lengths
        slack = 1.0 + (1.0 - g.tension) * 1.5
        rest = (np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]],
                               axis=1) * slack).astype(np.float32)

        broken = np.zeros(m, np.float32)
        if g.deteriorate > 0.0 and m:
            rnd = random.Random(g.seed)
            broken[[e for e in range(m)
                    if rnd.random() < g.deteriorate]] = 1.0
        # -1 marks an edge the tear kernel must leave alone
        a = me.attributes.get(A_NOTEAR)
        if a is not None and a.domain == 'EDGE' and m:
            nt = np.zeros(m, np.bool_)
            a.data.foreach_get("value", nt)
            broken[nt & (broken < 0.5)] = -1.0

        # incidence lists for the gather solve (topology is static)
        counts = np.zeros(n, np.int64)
        np.add.at(counts, edges[:, 0], 1)
        np.add.at(counts, edges[:, 1], 1)
        starts = np.zeros(n, np.int64)
        np.cumsum(counts[:-1], out=starts[1:])
        cursor = starts.copy()
        lst = np.zeros(max(int(counts.sum()), 1), np.float32)
        for e in range(m):
            for v in (edges[e, 0], edges[e, 1]):
                lst[cursor[v]] = e
                cursor[v] += 1
        inc_off = np.stack([starts, counts], 1).astype(np.float32)

        # per-point birth time (Web Shot), in the same seconds the shaders
        # see as p1.w. Absent = -1e9: every point alive from the first step,
        # which is every other web type.
        fps = max(bpy.context.scene.render.fps, 1)
        birth = np.full(n, -1e9, np.float32)
        a = me.attributes.get(A_SHOT)
        if a is not None and a.domain == 'POINT':
            tmp = np.zeros(n, np.float32)
            a.data.foreach_get("value", tmp)
            birth = (tmp / fps).astype(np.float32)

        # position textures are built after the collider is known, since
        # pinned points sitting on it are promoted to "follow" state first
        edge4 = np.concatenate(
            [edges.astype(np.float32), rest[:, None], broken[:, None]],
            1).astype(np.float32)

        self.edges = _tex(gpu, m, 4, edge4)
        self.inc_off = _tex(gpu, n, 2, inc_off)
        self.inc_lst = _tex(gpu, len(lst), 1, lst[:, None])
        self.tens = _tex(gpu, n, 1)

        # SDF collision (optional) + dummy 3D texture for sphere mode.
        # A collision collection (or >1 collider) always uses the SDF —
        # a bounding sphere can't represent multiple/compound shapes.
        self.sdf = None
        self.sdf_track = None    # single rigid collider to offset-track
        sdf_res = 1
        colliders = _collect_colliders(g) if g.enable_collision else []
        self.colliders = colliders
        want_sdf = colliders and (
            g.collision_shape == 'MESH_SDF' or len(colliders) > 1
            or getattr(g, "collider_collection", None) is not None)
        if want_sdf:
            self.sdf = _bake_sdf(gpu, obj, colliders, g.sdf_resolution)
            if self.sdf is not None:
                sdf_res = self.sdf["res"]
                if len(colliders) == 1:
                    self.sdf_track = colliders[0]
            else:
                print("Arachne: SDF bake failed (no faces?) — "
                      "falling back to sphere collision.")
        # Points already pinned onto the collider's surface (a web shot's
        # impact anchors, or hand-pinned ones) are promoted to pin state 2,
        # so they are carried along when it moves. Runtime sticky contacts
        # get the same state from the solve kernel.
        # pin state 2+k means "stuck to colliders[k]", so every member of a
        # collider collection carries its own anchors
        self.follow = bool(getattr(g, "stick_follow", False)) and colliders
        self.stick_objs = list(colliders) if self.follow else []
        if self.follow:
            grp = _assign_collider(obj, self.stick_objs, pos, pin > 0.5,
                                   g.collision_offset)
            hit = grp >= 0
            pin = np.where(hit, 2.0 + grp, pin).astype(np.float32)
            self._coll_prev = [o.matrix_world.copy() for o in self.stick_objs]
            self._assigned = hit.copy()
            self._offset = g.collision_offset
            self.follow = bool(hit.any()) or g.stickiness > 0.0
        else:
            self._coll_prev = []
            self._assigned = np.zeros(n, np.bool_)
            self._offset = g.collision_offset

        pos4 = np.concatenate([pos, pin[:, None]], 1).astype(np.float32)
        # prev carries the birth time in .w — the shaders preserve it
        prev4 = np.concatenate([pos, birth[:, None]], 1).astype(np.float32)
        self.posA = _tex(gpu, n, 4, pos4)
        self.posB = _tex(gpu, n, 4, pos4)
        self.prev = _tex(gpu, n, 4, prev4)

        self._dummy3d = _tex3d(gpu, 1, np.full((1, 1, 1), 1e3, np.float32))

        pt = 'FLOAT_2D'
        self.sh_int = _shader(gpu, _INTEGRATE,
                              [('RGBA32F', pt, 'posA'),
                               ('RGBA32F', pt, 'prevI')], n, m, sdf_res)
        self.sh_solve = _shader(gpu, _SOLVE,
                                [('RGBA32F', pt, 'posIn'),
                                 ('RGBA32F', pt, 'posOut'),
                                 ('RGBA32F', pt, 'prevI'),
                                 ('RGBA32F', pt, 'edges'),
                                 ('RG32F', pt, 'incOff'),
                                 ('R32F', pt, 'incLst'),
                                 ('R32F', 'FLOAT_3D', 'sdf')],
                                n, m, sdf_res)
        self.sh_follow = _shader(gpu, _FOLLOW,
                                 [('RGBA32F', pt, 'posA'),
                                  ('RGBA32F', pt, 'prevI')], n, m, sdf_res)
        self.sh_tear = _shader(gpu, _TEAR,
                               [('RGBA32F', pt, 'posA'),
                                ('RGBA32F', pt, 'edges')], n, m, sdf_res)
        self.sh_tens = _shader(gpu, _TENSION,
                               [('RGBA32F', pt, 'posA'),
                                ('RGBA32F', pt, 'edges'),
                                ('RG32F', pt, 'incOff'),
                                ('R32F', pt, 'incLst'),
                                ('R32F', pt, 'tens')], n, m, sdf_res)

    # -- dispatch helpers ---------------------------------------------------
    def _groups(self, count):
        h = max((count + _W - 1) // _W, 1)
        return (_W + 7) // 8, (h + 7) // 8

    def _push(self, sh, g, dt2, t_now, g_loc, w_loc, sphere,
              sdf_on, sdf_delta, sdf_bmin, sdf_inv):
        sh.uniform_float("p1", (dt2, g.damping, g.turbulence, t_now))
        sh.uniform_float("p2", (g_loc[0], g_loc[1], g_loc[2], sdf_delta[2]))
        sh.uniform_float("p3", (w_loc[0], w_loc[1], w_loc[2],
                                1.0 + g.stiffness))
        sh.uniform_float("p4", sphere)
        sh.uniform_float("p5", (g.collision_offset, g.friction,
                                g.tear_threshold,
                                1.0 if g.enable_tearing else 0.0))
        sh.uniform_float("p6", (1.0 if g.resist_compression else 0.0,
                                sdf_on, g.stickiness,
                                2.0 if self.follow else 1.0))
        sh.uniform_float("p7", (sdf_bmin[0], sdf_bmin[1], sdf_bmin[2],
                                sdf_delta[0]))
        sh.uniform_float("p8", (sdf_inv[0], sdf_inv[1], sdf_inv[2],
                                sdf_delta[1]))

    def step(self, obj, g, dt):
        gpu = self._gpu
        sub = max(g.substeps, 1)
        dt2 = (dt / sub) ** 2

        # world-space gravity/wind -> object local frame (per frame, so
        # rotated or animated web objects still sag toward true down)
        m3inv = obj.matrix_world.to_3x3().inverted_safe()
        g_loc = m3inv @ Vector(g.gravity)
        w_loc = m3inv @ Vector(g.wind)

        sphere = (0.0, 0.0, 0.0, 0.0)
        sdf_on = 0.0
        sdf_delta = (0.0, 0.0, 0.0)
        sdf_bmin = (0.0, 0.0, 0.0)
        sdf_inv = (0.0, 0.0, 0.0)
        if g.enable_collision and (self.sdf is not None or self.colliders):
            if self.sdf is not None:
                if self.sdf_track is not None:
                    cur = np.array(obj.matrix_world.inverted()
                                   @ self.sdf_track.matrix_world.translation)
                    d = cur - self.sdf["bake_loc"]
                    sdf_delta = (float(d[0]), float(d[1]), float(d[2]))
                # else: collection — static, delta stays zero
                sdf_on = 1.0
                sdf_bmin = tuple(float(x) for x in self.sdf["bmin"])
                sdf_inv = tuple(float(x) for x in self.sdf["inv_cell"])
            elif self.colliders:
                coll = self.colliders[0]
                loc = (obj.matrix_world.inverted()
                       @ coll.matrix_world.translation)
                s = obj.matrix_world.to_scale()
                avg_s = max((s.x + s.y + s.z) / 3.0, 1e-6)
                sphere = (loc.x, loc.y, loc.z,
                          max(coll.dimensions) * 0.5 / avg_s)

        t_now = bpy.context.scene.frame_current / max(
            bpy.context.scene.render.fps, 1)
        gx, gy = self._groups(self.n)
        egx, egy = self._groups(self.m)
        sdf_tex = self.sdf["tex"] if self.sdf is not None else self._dummy3d

        def push(sh):
            self._push(sh, g, dt2, t_now, g_loc, w_loc, sphere,
                       sdf_on, sdf_delta, sdf_bmin, sdf_inv)

        # carry stuck points along with the collider first, so the physics
        # substeps see them already in their new place
        if self.follow:
            wi = obj.matrix_world.inverted_safe()
            for k, coll in enumerate(self.stick_objs):
                cur = coll.matrix_world
                if cur == self._coll_prev[k]:
                    continue
                M = wi @ cur @ self._coll_prev[k].inverted_safe()                     @ obj.matrix_world
                sh = self.sh_follow
                sh.bind()
                sh.image('posA', self.posA)
                sh.image('prevI', self.prev)
                for r in range(3):
                    sh.uniform_float("p%d" % (r + 1),
                                     (M[r][0], M[r][1], M[r][2], M[r][3]))
                sh.uniform_float("p4", (2.0 + k, 0.0, 0.0, 0.0))
                gpu.compute.dispatch(sh, gx, gy, 1)
                self._coll_prev[k] = cur.copy()

        for _ in range(sub):
            sh = self.sh_int
            sh.bind()
            sh.image('posA', self.posA)
            sh.image('prevI', self.prev)
            push(sh)
            gpu.compute.dispatch(sh, gx, gy, 1)

            src, dst = self.posA, self.posB
            for _i in range(max(g.iterations, 1)):
                sh = self.sh_solve
                sh.bind()
                sh.image('posIn', src)
                sh.image('posOut', dst)
                sh.image('prevI', self.prev)
                sh.image('edges', self.edges)
                sh.image('incOff', self.inc_off)
                sh.image('incLst', self.inc_lst)
                sh.image('sdf', sdf_tex)
                push(sh)
                gpu.compute.dispatch(sh, gx, gy, 1)
                src, dst = dst, src
            self.posA, self.posB = src, dst

        sh = self.sh_tear
        sh.bind()
        sh.image('posA', self.posA)
        sh.image('edges', self.edges)
        push(sh)
        gpu.compute.dispatch(sh, egx, egy, 1)

        sh = self.sh_tens
        sh.bind()
        sh.image('posA', self.posA)
        sh.image('edges', self.edges)
        sh.image('incOff', self.inc_off)
        sh.image('incLst', self.inc_lst)
        sh.image('tens', self.tens)
        push(sh)
        gpu.compute.dispatch(sh, gx, gy, 1)

    def _claim_latched(self, obj, state):
        """A contact latched by Stickiness comes back from the solve kernel
        as plain state 2 — the shader has no idea which member of a collider
        collection it touched. Hand those points to the nearest one so they
        ride it. Only runs on the frames where new latches appear."""
        if len(self.stick_objs) < 2:
            return                       # state 2 already means colliders[0]
        loose = np.isclose(state[:, 3], 2.0) & ~self._assigned
        if not loose.any():
            return
        grp = _assign_collider(obj, self.stick_objs, state[:, :3], loose,
                               self._offset)
        hit = grp >= 0
        if not hit.any():
            return
        state[hit, 3] = 2.0 + grp[hit]
        self._assigned |= hit
        self.posA = _tex(self._gpu, self.n, 4,
                         np.ascontiguousarray(state, np.float32))

    def write_back(self, obj):
        """Read the sim state off the GPU into mesh attributes.
        Returns the arrays so callers can cache them for render replay."""
        state = _read(self.posA, self.n, 4)
        if self.follow:
            self._claim_latched(obj, state)
        pos = np.ascontiguousarray(state[:, :3]).ravel()
        brk = _read(self.edges, self.m, 4)[:, 3] > 0.5
        tens = _read(self.tens, self.n, 1).ravel().copy()
        apply_arrays(obj, pos, brk, tens)
        return pos, brk, tens
