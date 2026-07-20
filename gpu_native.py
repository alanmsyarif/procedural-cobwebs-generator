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
#   p6 = (resist_compression, sdf_on, 0, 0)
#   p7 = (sdf_box_min.xyz, sdf_delta.x)
#   p8 = (sdf_inv_cell.xyz, sdf_delta.y)

import random

import numpy as np

import bpy
from mathutils import Vector

from .constants import A_PIN, A_GPU_POS, A_BROKEN, A_TENSION

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
    print("SWF native GPU backend disabled after error:", ex)


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
    if (P4.w > 0.5) { imageStore(prevI, texel(i), vec4(P, 0.0)); return; }
    vec3 pv = imageLoad(prevI, texel(i)).xyz;
    vec3 vel = (P - pv) * p1.y;
    vec3 nse = vec3(n1(P, p1.w, 0.0), n1(P, p1.w, 17.0), n1(P, p1.w, 39.0));
    vec3 f = p2.xyz + p3.xyz + nse * p1.z;
    imageStore(prevI, texel(i), vec4(P, 0.0));
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
                vec3 pv = imageLoad(prevI, texel(i)).xyz;
                imageStore(prevI, texel(i),
                           vec4(pv + (target - pv) * p5.y, 0.0));
                P = target;
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
            vec3 pv = imageLoad(prevI, texel(i)).xyz;
            imageStore(prevI, texel(i),
                       vec4(pv + (target - pv) * p5.y, 0.0));
            P = target;
        }
    }
    imageStore(posOut, texel(i), vec4(P, P4.w));
}
"""

_TEAR = _COMMON + """
void main() {
    int e = int(gl_GlobalInvocationID.y) * WIDTH
          + int(gl_GlobalInvocationID.x);
    if (e >= M_EDGES) { return; }
    vec4 E = imageLoad(edges, texel(e));
    if (E.w > 0.5) { return; }
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

def _bake_sdf(gpu, web_obj, coll, res):
    """Signed distance field of the collider, in web-local space,
    sampled on a res^3 grid via BVH nearest queries."""
    from mathutils.bvhtree import BVHTree
    deps = bpy.context.evaluated_depsgraph_get()
    ob = coll.evaluated_get(deps)
    me = ob.to_mesh()
    me.calc_loop_triangles()
    M = web_obj.matrix_world.inverted() @ coll.matrix_world
    verts = [tuple(M @ v.co) for v in me.vertices]
    tris = [tuple(lt.vertices) for lt in me.loop_triangles]
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

    print("SWF: baking collider SDF (%d^3)..." % res)
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
    print("SWF: SDF bake done.")

    inv_cell = (res - 1) / np.maximum(bmax - bmin, 1e-9)
    bake_loc = np.array(
        web_obj.matrix_world.inverted() @ coll.matrix_world.translation)
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

        pos4 = np.concatenate([pos, pin[:, None]], 1).astype(np.float32)
        edge4 = np.concatenate(
            [edges.astype(np.float32), rest[:, None], broken[:, None]],
            1).astype(np.float32)

        self.posA = _tex(gpu, n, 4, pos4)
        self.posB = _tex(gpu, n, 4, pos4)
        self.prev = _tex(gpu, n, 4, pos4)
        self.edges = _tex(gpu, m, 4, edge4)
        self.inc_off = _tex(gpu, n, 2, inc_off)
        self.inc_lst = _tex(gpu, len(lst), 1, lst[:, None])
        self.tens = _tex(gpu, n, 1)

        # SDF collision (optional) + dummy 3D texture for sphere mode
        self.sdf = None
        sdf_res = 1
        if (g.collision_shape == 'MESH_SDF' and g.enable_collision
                and g.collider is not None):
            self.sdf = _bake_sdf(gpu, obj, g.collider, g.sdf_resolution)
            if self.sdf is not None:
                sdf_res = self.sdf["res"]
            else:
                print("SWF: SDF bake failed (no faces?) — "
                      "falling back to sphere collision.")
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
                                sdf_on, 0.0, 0.0))
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
        coll = g.collider
        if g.enable_collision and coll is not None:
            if self.sdf is not None:
                cur = np.array(obj.matrix_world.inverted()
                               @ coll.matrix_world.translation)
                d = cur - self.sdf["bake_loc"]
                sdf_on = 1.0
                sdf_delta = (float(d[0]), float(d[1]), float(d[2]))
                sdf_bmin = tuple(float(x) for x in self.sdf["bmin"])
                sdf_inv = tuple(float(x) for x in self.sdf["inv_cell"])
            else:
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

    def write_back(self, obj):
        """Read the sim state off the GPU into mesh attributes.
        Returns the arrays so callers can cache them for render replay."""
        pos = np.ascontiguousarray(_read(self.posA, self.n, 4)[:, :3]).ravel()
        brk = _read(self.edges, self.m, 4)[:, 3] > 0.5
        tens = _read(self.tens, self.n, 1).ravel().copy()
        apply_arrays(obj, pos, brk, tens)
        return pos, brk, tens
