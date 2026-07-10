# Native Blender GPU backend — runs the web solver as GLSL compute shaders
# through Blender's own `gpu` module. No external dependencies: uses the
# same GPU backend Blender itself runs on (Vulkan / Metal / OpenGL).
#
# Differences from the Taichi backend, forced by the gpu-module API:
#   * Data lives in RGBA32F textures (image load/store), not buffers.
#   * GLSL image atomics are integer-only, so the constraint solve is a
#     GATHER: each vertex owns a precomputed list of its incident edges
#     (static, since broken edges are masked rather than deleted) and sums
#     its own corrections. No atomics, fully deterministic.
#   * Jacobi iteration needs ping-pong position textures (read A, write B).
#
# Texture layout (width-major, index i -> texel (i % W, i / W)):
#   posA/posB : xyz = position, w = pinned flag
#   prev      : xyz = previous position (verlet state)
#   edges     : x = vert a, y = vert b, z = rest length, w = broken flag
#   inc_off   : x = start offset into inc_lst, y = incident edge count
#   inc_lst   : x = edge index (flattened incidence lists)
#   tens      : x = per-point normalized tension
#
# Physics is identical to the Taichi backend: verlet + gravity/wind/
# turbulence, unilateral distance constraints with valence-averaged
# corrections and SOR, sphere collision with friction, threshold tearing,
# tension slack from the Kole/Pixar adaptation.

import numpy as np

import bpy

from .constants import A_PIN, A_GPU_POS, A_BROKEN, A_TENSION

_W = 1024          # texture width; height grows with element count
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

# params1: dt2, damping, turbulence, time
# params2: gravity.xyz, n_points
# params3: wind.xyz, sor
# params4: sphere.xyz, sphere radius
# params5: collision offset, friction, tear threshold, tearing enable
# params6: n_edges, resist_compression, unused, unused
_PUSH = """
    vec4 params1; vec4 params2; vec4 params3;
    vec4 params4; vec4 params5; vec4 params6;
"""

_INTEGRATE = _COMMON + """
void main() {
    int i = int(gl_GlobalInvocationID.y) * WIDTH
          + int(gl_GlobalInvocationID.x);
    if (i >= int(params2.w)) { return; }
    vec4 P4 = imageLoad(posA, texel(i));
    vec3 P = P4.xyz;
    if (P4.w > 0.5) { imageStore(prevI, texel(i), vec4(P, 0.0)); return; }
    vec3 pv = imageLoad(prevI, texel(i)).xyz;
    vec3 vel = (P - pv) * params1.y;
    vec3 nse = vec3(n1(P, params1.w, 0.0), n1(P, params1.w, 17.0),
                    n1(P, params1.w, 39.0));
    vec3 f = params2.xyz + params3.xyz + nse * params1.z;
    imageStore(prevI, texel(i), vec4(P, 0.0));
    imageStore(posA, texel(i), vec4(P + vel + f * params1.x, P4.w));
}
"""

_SOLVE = _COMMON + """
void main() {
    int i = int(gl_GlobalInvocationID.y) * WIDTH
          + int(gl_GlobalInvocationID.x);
    if (i >= int(params2.w)) { return; }
    vec4 P4 = imageLoad(posIn, texel(i));
    vec3 P = P4.xyz;
    if (P4.w > 0.5) { imageStore(posOut, texel(i), P4); return; }

    vec2 off = imageLoad(incOff, texel(i)).xy;
    int start = int(off.x); int cnt = int(off.y);
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
            /* unilateral: silk pulls, never pushes */
            if (stretch > 0.0 || params6.y > 0.5) {
                acc += d * (stretch / len * 0.5);
            }
        }
    }
    if (alive > 0) { P += acc / float(alive) * params3.w; }

    /* sphere collision + friction */
    if (params4.w > 0.0) {
        vec3 c = params4.xyz;
        vec3 d = P - c;
        float len = length(d);
        float rr = params4.w + params5.x;
        if (len < rr) {
            vec3 nrm = d / max(len, 1e-9);
            vec3 target = c + nrm * rr;
            vec3 pv = imageLoad(prevI, texel(i)).xyz;
            imageStore(prevI, texel(i),
                       vec4(pv + (target - pv) * params5.y, 0.0));
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
    if (e >= int(params6.x)) { return; }
    vec4 E = imageLoad(edges, texel(e));
    if (E.w > 0.5) { return; }
    vec3 A = imageLoad(posA, texel(int(E.x))).xyz;
    vec3 B = imageLoad(posA, texel(int(E.y))).xyz;
    float len = length(B - A);
    if (params5.w > 0.5 && len > E.z * params5.z) {
        imageStore(edges, texel(e), vec4(E.xyz, 1.0));
    }
}
"""

_TENSION = _COMMON + """
void main() {
    int i = int(gl_GlobalInvocationID.y) * WIDTH
          + int(gl_GlobalInvocationID.x);
    if (i >= int(params2.w)) { return; }
    vec2 off = imageLoad(incOff, texel(i)).xy;
    int start = int(off.x); int cnt = int(off.y);
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
        float t = (ratio - 1.0) / max(params5.z - 1.0, 0.01);
        tmax = max(tmax, clamp(t, 0.0, 1.0));
    }
    imageStore(tens, texel(i), vec4(tmax, 0.0, 0.0, 0.0));
}
"""


# ---------------------------------------------------------------------------
#  Backend state
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


def _read(tex, count, channels):
    buf = tex.read()
    try:
        arr = np.asarray(buf, dtype=np.float32)
    except Exception:
        arr = np.array(buf.to_list(), dtype=np.float32)
    return arr.reshape(-1, channels)[:count]


def _shader(gpu, source, images, n_push_vec4=6):
    info = gpu.types.GPUShaderCreateInfo()
    info.define("WIDTH", str(_W))
    for slot, (fmt, name) in enumerate(images):
        info.image(slot, fmt, 'FLOAT_2D', name,
                   qualifiers={"READ", "WRITE"})
    for pname in ("params1", "params2", "params3",
                  "params4", "params5", "params6")[:n_push_vec4]:
        info.push_constant('VEC4', pname)
    info.local_group_size(8, 8)
    info.compute_source(source)
    return gpu.shader.create_from_info(info)


class NativeState:
    """Same interface as the Taichi _State: step() / write_back()."""

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

        slack = 1.0 + (1.0 - g.tension) * 1.5
        rest = (np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]],
                               axis=1) * slack).astype(np.float32)

        broken = np.zeros(m, np.float32)
        if g.deteriorate > 0.0 and m:
            import random as _r
            rnd = _r.Random(g.seed)
            broken[[e for e in range(m)
                    if rnd.random() < g.deteriorate]] = 1.0

        # incidence lists for the gather solve (static topology)
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

        self.sh_int = _shader(gpu, _INTEGRATE,
                              [('RGBA32F', 'posA'), ('RGBA32F', 'prevI')])
        self.sh_solve = _shader(gpu, _SOLVE,
                                [('RGBA32F', 'posIn'), ('RGBA32F', 'posOut'),
                                 ('RGBA32F', 'prevI'), ('RGBA32F', 'edges'),
                                 ('RG32F', 'incOff'), ('R32F', 'incLst')])
        self.sh_tear = _shader(gpu, _TEAR,
                               [('RGBA32F', 'posA'), ('RGBA32F', 'edges')])
        self.sh_tens = _shader(gpu, _TENSION,
                               [('RGBA32F', 'posA'), ('RGBA32F', 'edges'),
                                ('RG32F', 'incOff'), ('R32F', 'incLst'),
                                ('R32F', 'tens')])

    # -- dispatch helpers ---------------------------------------------------
    def _groups(self, count):
        h = max((count + _W - 1) // _W, 1)
        return (_W + 7) // 8, (h + 7) // 8

    def _push(self, sh, g, dt2, t_now, sphere):
        sh.uniform_float("params1", (dt2, g.damping, g.turbulence, t_now))
        sh.uniform_float("params2", (*g.gravity, float(self.n)))
        sh.uniform_float("params3", (*g.wind, 1.0 + g.stiffness))
        sh.uniform_float("params4", sphere)
        sh.uniform_float("params5", (g.collision_offset, g.friction,
                                     g.tear_threshold,
                                     1.0 if g.enable_tearing else 0.0))
        sh.uniform_float("params6", (float(self.m),
                                     1.0 if g.resist_compression else 0.0,
                                     0.0, 0.0))

    def step(self, obj, g, dt):
        gpu = self._gpu
        sub = max(g.substeps, 1)
        dt2 = (dt / sub) ** 2
        sphere = (0.0, 0.0, 0.0, 0.0)
        coll = g.collider
        if g.enable_collision and coll is not None:
            loc = obj.matrix_world.inverted() @ coll.matrix_world.translation
            sphere = (loc.x, loc.y, loc.z, max(coll.dimensions) * 0.5)
        t_now = bpy.context.scene.frame_current / max(
            bpy.context.scene.render.fps, 1)
        gx, gy = self._groups(self.n)
        egx, egy = self._groups(self.m)

        for _ in range(sub):
            sh = self.sh_int
            sh.bind()
            sh.image('posA', self.posA)
            sh.image('prevI', self.prev)
            self._push(sh, g, dt2, t_now, sphere)
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
                self._push(sh, g, dt2, t_now, sphere)
                gpu.compute.dispatch(sh, gx, gy, 1)
                src, dst = dst, src
            self.posA, self.posB = src, dst

        sh = self.sh_tear
        sh.bind()
        sh.image('posA', self.posA)
        sh.image('edges', self.edges)
        self._push(sh, g, dt2, t_now, sphere)
        gpu.compute.dispatch(sh, egx, egy, 1)

        sh = self.sh_tens
        sh.bind()
        sh.image('posA', self.posA)
        sh.image('edges', self.edges)
        sh.image('incOff', self.inc_off)
        sh.image('incLst', self.inc_lst)
        sh.image('tens', self.tens)
        self._push(sh, g, dt2, t_now, sphere)
        gpu.compute.dispatch(sh, gx, gy, 1)

    def write_back(self, obj):
        from .gpu_solver import _ensure_attr
        me = obj.data
        _ensure_attr(me, A_GPU_POS, 'FLOAT_VECTOR', 'POINT')
        _ensure_attr(me, A_BROKEN, 'BOOLEAN', 'EDGE')
        _ensure_attr(me, A_TENSION, 'FLOAT', 'POINT')
        me.attributes[A_GPU_POS].data.foreach_set(
            "vector", _read(self.posA, self.n, 4)[:, :3].ravel())
        me.attributes[A_BROKEN].data.foreach_set(
            "value", _read(self.edges, self.m, 4)[:, 3] > 0.5)
        me.attributes[A_TENSION].data.foreach_set(
            "value", _read(self.tens, self.n, 1).ravel())
        me.update_tag()
