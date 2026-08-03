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
# pos.w is the pin state: 0 free, 1 pinned in place, 2+k pinned AND carried
# along by collider k (see _FOLLOW). Every ">0.5" test means "pinned".
# Web Shot muzzle anchors are state 1 and placed from the emitter every
# frame by _ANCHOR instead.

import json
import random
import re
import traceback

import numpy as np

import bpy
from mathutils import Vector

from .constants import (A_PIN, A_SHOT, A_NOTEAR, A_GPU_POS, A_BROKEN,
                        A_TENSION, A_EMIT, P_EMITTER, P_EMIT_AT)

_W = 1024
_BROKEN_FLAG = False
_BROKEN_MSG = ""


def native_available():
    try:
        import gpu
        return hasattr(gpu, "compute") and hasattr(gpu.types,
                                                   "GPUShaderCreateInfo")
    except Exception:
        return False


def native_broken():
    return _BROKEN_FLAG


def broken_reason():
    """Why the backend was disabled, short enough for a panel row. One
    caught exception disables the solver for the rest of the session, and
    printing it to the system console was no use to anyone who had not
    already opened that console — so it is kept here too."""
    return _BROKEN_MSG


def _mark_broken(ex):
    global _BROKEN_FLAG, _BROKEN_MSG
    _BROKEN_FLAG = True
    _BROKEN_MSG = "%s: %s" % (type(ex).__name__, ex)
    print("Arachne native GPU backend disabled after error:", ex)
    traceback.print_exc()


def _clear_broken():
    global _BROKEN_FLAG, _BROKEN_MSG
    _BROKEN_FLAG = False
    _BROKEN_MSG = ""


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
# 2+k, so a web that latched onto a moving object travels with it instead
# of hanging in the air where it stuck. p1..p3 are the rows of that
# transform, expressed in the web's local space. (Web Shot muzzles use
# _ANCHOR instead — a contact has no fire frame to date it from, so this
# one has to accumulate.)
_FOLLOW = _COMMON + """
void main() {
    int i = int(gl_GlobalInvocationID.y) * WIDTH
          + int(gl_GlobalInvocationID.x);
    if (i >= N_POINTS) { return; }
    vec4 P4 = imageLoad(posA, texel(i));
    /* p4.x selects the group: state 2+k is stuck to host k, so each
       collection member (and the emitter) is dispatched with its own
       transform */
    if (abs(P4.w - p4.x) > 0.25) { return; }
    vec4 PV = imageLoad(prevI, texel(i));
    /* prevI.w = birth time: a muzzle anchor was built where the emitter
       stood at its fire frame, so carrying it before then would apply that
       motion twice. It starts riding the moment it fires (p4.y = now). */
    if (PV.w > p4.y) { return; }
    vec3 P = P4.xyz;
    imageStore(posA, texel(i), vec4(dot(p1.xyz, P) + p1.w,
                                    dot(p2.xyz, P) + p2.w,
                                    dot(p3.xyz, P) + p3.w, P4.w));
    /* carry the verlet history too, or the point reads as having been
       teleported and fires off at host speed once released */
    imageStore(prevI, texel(i), vec4(dot(p1.xyz, PV.xyz) + p1.w,
                                     dot(p2.xyz, PV.xyz) + p2.w,
                                     dot(p3.xyz, PV.xyz) + p3.w, PV.w));
}
"""

# Web Shot muzzle anchors. Unlike a collider — which needs full rigid
# motion and has no idea when a point stuck to it — an emitter anchor
# knows the frame it fired, so its place can be stated outright instead of
# accumulated frame by frame:
#
#     position(t) = emitter(t) + (built position - emitter(fire frame))
#
# The bracketed part is baked into `base` at bind time. Nothing to keep in
# step with, so scrubbing, jumping, dropped frames and rebuilds all land in
# the same place. p1.xyz is the emitter's current location in web space,
# p4.y is now (an anchor holds its built pose until its shot fires).
_ANCHOR = _COMMON + """
void main() {
    int i = int(gl_GlobalInvocationID.y) * WIDTH
          + int(gl_GlobalInvocationID.x);
    if (i >= N_POINTS) { return; }
    vec4 B = imageLoad(base, texel(i));
    if (B.w < 0.5) { return; }           /* not an emitter anchor */
    vec4 PV = imageLoad(prevI, texel(i));
    if (PV.w > p4.y) { return; }         /* not fired yet */
    vec3 P = p1.xyz + B.xyz;
    vec4 P4 = imageLoad(posA, texel(i));
    imageStore(posA, texel(i), vec4(P, P4.w));
    /* prev follows, or the anchor reads as teleported and whips the
       strand hanging off it */
    imageStore(prevI, texel(i), vec4(P, PV.w));
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


def _push_names(source):
    """Which of p1..p8 a kernel actually reads.

    On the Vulkan backend push constants live in one block, so every name
    resolves whether the kernel mentions it or not. The OpenGL backend
    compiles them to plain uniforms instead, and the GLSL compiler drops
    any the kernel never reads, and setting one then raises "uniform pN
    not found". Only _SOLVE uses all eight, so the shared push has to know
    what each kernel wants. Comments go first: a name surviving only in a
    comment is not a reference, and the compiler would strip it."""
    code = re.sub(r"/\*.*?\*/|//[^\n]*", " ", source, flags=re.S)
    return tuple(n for n in _PUSH_NAMES if re.search(r"\b%s\b" % n, code))


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
    for pname in _push_names(source):
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
        if coll.type != 'MESH':          # an emitter empty has no surface
            continue
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


def rest_slack(tension):
    """Rest length as a multiple of built length (Kole tension slack).

    Below 1.0 rest lengths run long, so threads carry slack and droop into
    catenaries. Past 1.0 it runs the other way — rest SHORTER than built, so
    the unilateral constraints pull permanently and the residual gravity
    droop that survives at 1.0 comes out.

    Much gentler slope on that side: the slack rate would contract the web
    by a quarter and yank the anchors, where 2.0 is 15% pre-tension and
    already rigid. 2.0 is where the slider stops but not the dial — the
    range runs to 5.0 (60%) for a cord that has to hold a straight line
    across a long span, since the residual droop grows with the span and
    what looks rigid over a metre does not over ten.

    0.15 is deliberate, and was briefly dropped to 0.06 to stop web shots
    tearing. That was the wrong lever — the fragility was short clot
    segments, now flagged unbreakable at build time, and softening this
    only took the tautness out of the dial people reach for. Pre-tension
    does still cost tear margin, since tearing measures strain against rest
    and this shrinks rest; the panel reports what is left.

    Shared with that panel, so the two cannot drift apart."""
    if tension <= 1.0:
        return 1.0 + (1.0 - tension) * 1.5
    return 1.0 - (tension - 1.0) * 0.15


def _emitter_track(me):
    """The generator's record of where the emitter stood at each volley,
    as [(fire frame, world location)]. Empty for a web built before this
    was baked."""
    raw = me.get(P_EMIT_AT)
    if not raw:
        return []
    try:
        return [(float(row[0]), np.asarray(row[1:4], dtype=np.float64))
                for row in json.loads(raw) if len(row) >= 4]
    except Exception:
        return []


def _track_lookup(track, frame, tol=1e-3):
    """The baked location for `frame`. Matched by nearest rather than by
    equality: the fire frames come back out of a float32 mesh attribute,
    which is not the value that went in."""
    best, best_d = None, tol
    for f, at in track:
        d = abs(f - frame)
        if d <= best_d:
            best, best_d = at, d
    return best


def _anchor_base(web_obj, me, emitter, pos, mask, fire):
    """Bake each muzzle anchor's offset from the emitter.

    A muzzle is built where the emitter stood at its fire frame, so
    `built - emitter(fire)` is the offset it keeps for the rest of the
    shot. Returned as an n x 4 array (offset.xyz, 1 = is an anchor) for the
    anchor kernel, which adds the emitter's current location back on.

    The fire-frame end comes from the table the generator baked, which was
    read off a properly evaluated depsgraph — parents, bones, constraints
    and drivers included. `_obj_loc_at` is only the fallback for webs built
    before that table existed; it sees plain location F-curves and nothing
    else. Either way it is location only, so a rotating emitter carries its
    silk but does not swing it."""
    base = np.zeros((len(pos), 4), np.float32)
    if fire is None or not mask.any():
        return base
    from .generator import _obj_loc_at
    track = _emitter_track(me)
    winv = web_obj.matrix_world.inverted_safe()
    idx = np.flatnonzero(mask)
    for f in np.unique(fire[idx]):
        at = _track_lookup(track, float(f))
        if at is None:
            at = _obj_loc_at(emitter, float(f))
        if at is None:
            continue
        local = np.asarray(winv @ Vector(at), dtype=np.float64)
        rows = idx[fire[idx] == f]
        base[rows, :3] = (pos[rows] - local).astype(np.float32)
        base[rows, 3] = 1.0
    return base


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

        rest = (np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]],
                               axis=1) * rest_slack(g.tension)
                ).astype(np.float32)

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
        fire = None
        a = me.attributes.get(A_SHOT)
        if a is not None and a.domain == 'POINT':
            tmp = np.zeros(n, np.float32)
            a.data.foreach_get("value", tmp)
            fire = tmp.copy()               # fire frame per point
            birth = (tmp / fps).astype(np.float32)
        self.fire = fire                    # kept for pull_force's reveal gate

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
        ride = bool(getattr(g, "stick_follow", False)) and bool(colliders)
        self.stick_objs = list(colliders) if ride else []
        self.n_coll = len(self.stick_objs)
        if ride:
            grp = _assign_collider(obj, self.stick_objs, pos, pin > 0.5,
                                   g.collision_offset)
            hit = grp >= 0
            pin = np.where(hit, 2.0 + grp, pin).astype(np.float32)
            self._assigned = hit.copy()
            # runtime sticky contacts come back as plain state 2, i.e.
            # colliders[0] — only meaningful while collider follow is on
            self.latch = bool(hit.any()) or g.stickiness > 0.0
        else:
            self._assigned = np.zeros(n, np.bool_)
            self.latch = False
        self._offset = g.collision_offset

        # A Web Shot's muzzle anchors are placed straight from the emitter
        # every frame (see _ANCHOR), so they take no part in the collider
        # follow groups — pin state 1 keeps the physics off them.
        self.emitter = None
        name = me.get(P_EMITTER)
        a = me.attributes.get(A_EMIT)
        if name and a is not None and a.domain == 'POINT':
            self.emitter = bpy.data.objects.get(name)
            if self.emitter is None:
                print("Arachne: emitter %r is gone (renamed or deleted) — "
                      "the shot's muzzle anchors stay where they fired."
                      % name)
        base4 = np.zeros((n, 4), np.float32)
        if self.emitter is not None:
            mask = np.zeros(n, np.bool_)
            a.data.foreach_get("value", mask)
            mask &= pin > 0.5
            if mask.any():
                pin = np.where(mask, 1.0, pin).astype(np.float32)
                self._assigned |= mask
                base4 = _anchor_base(obj, me, self.emitter, pos, mask, fire)
            if base4[:, 3].max() < 0.5:
                self.emitter = None
        self.anchors = _tex(gpu, n, 4, base4)
        self._coll_prev = [o.matrix_world.copy() for o in self.stick_objs]
        self._emit_prev = None
        self._carry_frame = None
        self.follow = bool(self._assigned.any()) or self.latch

        # two-way coupling (see pull_force). Single collider only: pin
        # state 2 means colliders[0], and a collection is a merged static
        # SDF bake with no per-member surface to push back against anyway.
        self.drag_obj = colliders[0] if len(colliders) == 1 else None
        self._last_state = None
        self._last_edges = None

        pos4 = np.concatenate([pos, pin[:, None]], 1).astype(np.float32)
        # prev carries the birth time in .w — the shaders preserve it
        prev4 = np.concatenate([pos, birth[:, None]], 1).astype(np.float32)
        self.posA = _tex(gpu, n, 4, pos4)
        self.posB = _tex(gpu, n, 4, pos4)
        self.prev = _tex(gpu, n, 4, prev4)

        self._dummy3d = _tex3d(gpu, 1, np.full((1, 1, 1), 1e3, np.float32))

        pt = 'FLOAT_2D'
        # _push() feeds every kernel from one set of values, so it has to
        # know which names each one compiled with (see _push_names)
        self._pn = {}

        def mk(source, images):
            sh = _shader(gpu, source, images, n, m, sdf_res)
            self._pn[sh] = _push_names(source)
            return sh

        self.sh_int = mk(_INTEGRATE,
                         [('RGBA32F', pt, 'posA'),
                          ('RGBA32F', pt, 'prevI')])
        self.sh_solve = mk(_SOLVE,
                           [('RGBA32F', pt, 'posIn'),
                            ('RGBA32F', pt, 'posOut'),
                            ('RGBA32F', pt, 'prevI'),
                            ('RGBA32F', pt, 'edges'),
                            ('RG32F', pt, 'incOff'),
                            ('R32F', pt, 'incLst'),
                            ('R32F', 'FLOAT_3D', 'sdf')])
        self.sh_follow = mk(_FOLLOW,
                            [('RGBA32F', pt, 'posA'),
                             ('RGBA32F', pt, 'prevI')])
        self.sh_anchor = mk(_ANCHOR,
                            [('RGBA32F', pt, 'posA'),
                             ('RGBA32F', pt, 'prevI'),
                             ('RGBA32F', pt, 'base')])
        self.sh_tear = mk(_TEAR,
                          [('RGBA32F', pt, 'posA'),
                           ('RGBA32F', pt, 'edges')])
        self.sh_tens = mk(_TENSION,
                          [('RGBA32F', pt, 'posA'),
                           ('RGBA32F', pt, 'edges'),
                           ('RG32F', pt, 'incOff'),
                           ('R32F', pt, 'incLst'),
                           ('R32F', pt, 'tens')])

    # -- dispatch helpers ---------------------------------------------------
    def _groups(self, count):
        h = max((count + _W - 1) // _W, 1)
        return (_W + 7) // 8, (h + 7) // 8

    def _push(self, sh, g, dt2, t_now, g_loc, w_loc, sphere,
              sdf_on, sdf_delta, sdf_bmin, sdf_inv):
        vals = {
            "p1": (dt2, g.damping, g.turbulence, t_now),
            "p2": (g_loc[0], g_loc[1], g_loc[2], sdf_delta[2]),
            "p3": (w_loc[0], w_loc[1], w_loc[2], 1.0 + g.stiffness),
            "p4": sphere,
            "p5": (g.collision_offset, g.friction, g.tear_threshold,
                   1.0 if g.enable_tearing else 0.0),
            "p6": (1.0 if g.resist_compression else 0.0, sdf_on,
                   g.stickiness, 2.0 if self.latch else 1.0),
            "p7": (sdf_bmin[0], sdf_bmin[1], sdf_bmin[2], sdf_delta[0]),
            "p8": (sdf_inv[0], sdf_inv[1], sdf_inv[2], sdf_delta[1]),
        }
        # only what this kernel compiled with; the rest are not there to
        # set on the OpenGL backend
        for name in self._pn[sh]:
            sh.uniform_float(name, vals[name])

    def hosts_moved(self):
        """Has anything the anchors are stuck to moved since the last
        carry? Cheap enough to ask on every depsgraph update."""
        if self.emitter is not None:
            if self._emit_prev is None:
                return True
            cur = self.emitter.matrix_world.translation
            if (cur - self._emit_prev).length > 1e-7:
                return True
        return any(host.matrix_world != self._coll_prev[k]
                   for k, host in enumerate(self.stick_objs))

    def carry(self, obj, stepping=False):
        """Put every kinematic anchor where its host says it should be.

        Called from step(), and also on frames the sim only holds — jumping
        the timeline, dropped frames during playback and scrubbing all skip
        stepping, and an anchor left frozen there parts company with the
        object it is welded to. Anchors are kinematic, not simulated, so
        they can be placed on any frame; the rest of the web still needs the
        frames played through."""
        gpu = self._gpu
        scene = bpy.context.scene
        frame = scene.frame_current
        t_now = frame / max(scene.render.fps, 1)
        gx, gy = self._groups(self.n)

        # Web Shot muzzles: stated outright from the emitter's location, so
        # this is correct on any frame, in any order, however the timeline
        # got here
        if self.emitter is not None:
            # world for the moved-since-last-carry test, web-local for the
            # kernel — the web object need not sit at the origin
            self._emit_prev = self.emitter.matrix_world.translation.copy()
            E = obj.matrix_world.inverted_safe() @ self._emit_prev
            sh = self.sh_anchor
            sh.bind()
            sh.image('posA', self.posA)
            sh.image('prevI', self.prev)
            sh.image('base', self.anchors)
            sh.uniform_float("p1", (E.x, E.y, E.z, 0.0))
            sh.uniform_float("p4", (0.0, t_now, 0.0, 0.0))
            gpu.compute.dispatch(sh, gx, gy, 1)

        if not self.follow:
            return
        # Scrubbed backwards: anchors that fired after this frame are unborn
        # again and would sit out the transform, losing their place in the
        # chain of increments for good. Leave the whole delta pending — the
        # next real step applies it in one go and everything re-syncs.
        if not stepping and self._carry_frame is not None and (
                frame < self._carry_frame):
            return
        self._carry_frame = frame
        wi = obj.matrix_world.inverted_safe()
        for k, host in enumerate(self.stick_objs):
            cur = host.matrix_world
            if cur == self._coll_prev[k]:
                continue
            M = wi @ cur @ self._coll_prev[k].inverted_safe()                 @ obj.matrix_world
            sh = self.sh_follow
            sh.bind()
            sh.image('posA', self.posA)
            sh.image('prevI', self.prev)
            for r in range(3):
                sh.uniform_float("p%d" % (r + 1),
                                 (M[r][0], M[r][1], M[r][2], M[r][3]))
            # p4.y = now, so anchors only start riding once they fire
            sh.uniform_float("p4", (2.0 + k, t_now, 0.0, 0.0))
            gpu.compute.dispatch(sh, gx, gy, 1)
            self._coll_prev[k] = cur.copy()

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

        # carry stuck points along with their host first, so the physics
        # substeps see them already in their new place
        self.carry(obj, stepping=True)

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
        if self.n_coll < 2:
            return                       # state 2 already means colliders[0]
        loose = np.isclose(state[:, 3], 2.0) & ~self._assigned
        if not loose.any():
            return
        # only the colliders are candidates — an emitter has no surface to
        # latch onto, and it sits past n_coll in stick_objs
        grp = _assign_collider(obj, self.stick_objs[:self.n_coll],
                               state[:, :3], loose, self._offset)
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
        if self.latch:
            self._claim_latched(obj, state)
        pos = np.ascontiguousarray(state[:, :3]).ravel()
        edges = _read(self.edges, self.m, 4)
        brk = edges[:, 3] > 0.5
        tens = _read(self.tens, self.n, 1).ravel().copy()
        apply_arrays(obj, pos, brk, tens)
        # kept for pull_force, which runs after this and would otherwise
        # pay for the same two readbacks again
        self._last_state = state
        self._last_edges = edges
        return pos, brk, tens

    # -- two-way coupling ---------------------------------------------------

    def pull_force(self, g):
        """Net force the web is exerting on the collider, in world space.
        Returns None when there is nothing to report.

        The web only ever touches the collider through points stuck to it —
        impact anchors and Stickiness latches, pin state 2+k. Those are
        kinematic, so the stretch their incident threads carry is force the
        solver silently throws away; summing it is the reaction the collider
        never got. Unilateral, like the constraints themselves: a slack
        thread pulls with nothing, and a thread stuck at both ends is an
        internal force that cancels.

        Reporting only — Bullet moves the object (see gpu_solver's pull
        field), so mass, friction, gravity and tumble are its business."""
        state, edges = self._last_state, self._last_edges
        if (self.drag_obj is None or state is None or edges is None
                or not getattr(g, "pull_collider", False)):
            return None

        # points welded to this collider: pin state 2 is colliders[0]
        stuck = np.isclose(state[:, 3], 2.0)
        if not stuck.any():
            return None

        live = edges[:, 3] <= 0.5            # torn threads pull with nothing
        ia = edges[:, 0].astype(np.int32)
        ib = edges[:, 1].astype(np.int32)
        pos = state[:, :3]

        # A Web Shot's impact anchors are pinned to the collider from the
        # first frame — that is how the generator builds them, long before
        # the flying tip gets there. Without this the whole web hauls on the
        # object while it is still visibly in the air. Same rule the reveal
        # uses: a point exists once its shot has reached it, and a thread
        # only transmits once both of its ends do.
        if self.fire is not None:
            born = self.fire <= bpy.context.scene.frame_current
            if not born.any():
                return None
            stuck = stuck & born
            if not stuck.any():
                return None
            live = live & born[ia] & born[ib]

        force = np.zeros(3, np.float64)
        for src, dst in ((ia, ib), (ib, ia)):
            sel = live & stuck[src] & ~stuck[dst]
            if not sel.any():
                continue
            d = pos[dst[sel]] - pos[src[sel]]
            length = np.linalg.norm(d, axis=1)
            ok = length > 1e-9
            if not ok.any():
                continue
            d, length = d[ok], length[ok]
            stretch = np.maximum(length - edges[sel, 2][ok], 0.0)
            force += (d / length[:, None] * stretch[:, None]).sum(0)
        return force * float(g.pull_strength)

