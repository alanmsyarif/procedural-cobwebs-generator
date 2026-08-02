"""Self-checks for the Web Shot geometry: the Clot Smooth sampling in
generator.shot_samples, and the collider test in generator._buried.

Run with plain Python, no Blender: `python test_shot.py`. generator.py
imports bpy at module scope and shot_samples is nested inside the shot
builder, so both functions are lifted out of the source and shot_samples is
handed a stand-in for the closed-over `p`.
"""
import ast
import pathlib
import types

import numpy as np

_src = pathlib.Path(__file__).with_name("generator.py").read_text("utf-8")
_want = {"shot_samples", "_buried"}
_fns = [n for n in ast.walk(ast.parse(_src))
        if isinstance(n, ast.FunctionDef) and n.name in _want]
assert {f.name for f in _fns} == _want, "generator.py renamed a function"
_ns = {"np": np}
exec(compile(ast.Module(_fns, []), "generator.py", "exec"), _ns)
shot_samples, _buried = _ns["shot_samples"], _ns["_buried"]


def settings(smooth=1, detail=5, twist=4.0):
    # `p` is a closure variable in the real code, a global here — rebinding
    # it in the namespace is what swaps the settings between cases
    _ns["p"] = types.SimpleNamespace(detail=detail, shot_clot_twist=twist,
                                     shot_clot_smooth=smooth)


def buried():
    """Cube spanning x 7..9, z 0.5..2.5. CLEAR is outside()'s default."""
    CLEAR = 8e-3

    def call(co, surf, nrm):
        co, surf, nrm = (np.array(v, float) for v in (co, surf, nrm))
        return _buried(co, surf, nrm, float(np.linalg.norm(co - surf)), CLEAR)

    # The regression: a strand still 4.5 m short of the target, half a
    # millimetre under the plane of its bottom face. Nearest point is on
    # that face's edge, so the normal's sign is meaningless — calling this
    # buried teleported the vertex onto the cube and left two 4.5 m edges.
    assert not call((2.488, -0.009, 0.4995), (7.0, -0.009, 0.5), (0, 0, -1))
    # ...and it must stay wrong however far under the plane it drifts,
    # as long as it is nowhere near the solid
    assert not call((2.488, 0.0, 0.3), (7.0, 0.0, 0.5), (0, 0, -1))

    # genuinely inside: the surface faces away, near or deep
    assert call((8.0, 0.0, 1.5), (7.0, 0.0, 1.5), (-1, 0, 0))
    assert call((7.001, 0.0, 1.5), (7.0, 0.0, 1.5), (-1, 0, 0))
    # inside but off-axis — still clearly behind the face
    assert call((7.9, 0.0, 1.8), (7.0, 0.0, 1.5), (-1, 0, 0))

    # outside and clear of it: left alone
    assert not call((6.9, 0.0, 1.5), (7.0, 0.0, 1.5), (-1, 0, 0))
    # outside but within the clearance band: nudged, and the nudge is small
    # by construction because `surf` is right there
    assert call((6.995, 0.0, 1.5), (7.0, 0.0, 1.5), (-1, 0, 0))

    # degenerate: co exactly on the surface
    assert call((7.0, 0.0, 1.5), (7.0, 0.0, 1.5), (-1, 0, 0))


def demo():
    CLOT = 0.5
    settings(smooth=1)
    base, base_s = shot_samples(CLOT)

    # a sampling is only usable if it spans the whole flight and never
    # doubles back — the solver builds one segment per consecutive pair
    assert base[0] == 0.0 and base[-1] == 1.0
    assert all(b > a for a, b in zip(base, base[1:])), "ts must increase"

    for k in (2, 5, 8):
        settings(smooth=k)
        ts, s = shot_samples(CLOT)
        # the braid is cut k times finer, and only the braid
        assert s == base_s * k, "Clot Smooth must scale the braid count"
        assert len(ts) - 1 - s == len(base) - 1 - base_s, "fan changed"
        # clot_s still lands exactly on `clot`, which is what lets the
        # binder flags and the lashing stations stay meaningful
        assert ts[s] == CLOT
        assert all(b > a for a, b in zip(ts, ts[1:]))
        # every original station is still a station, so a tie strided by k
        # sits where Clot Smooth 1 put it
        assert all(abs(ts[i * k] - base[i]) < 1e-12
                   for i in range(base_s + 1)), "stations moved"
        # the shortcut constraints walk range(0, clot_s, k) and index s + k,
        # so the braid must divide evenly or the last one runs off the end
        assert s % k == 0 and s // k == base_s

    # smooth = 1 is byte-identical to the old sampling: default changes nothing
    settings(smooth=1)
    assert shot_samples(CLOT) == (base, base_s)

    # no clot, nothing to subdivide — the dial must not touch the fan
    settings(smooth=8)
    ts, s = shot_samples(0.0)
    settings(smooth=1)
    assert (ts, s) == shot_samples(0.0) and s == 0

    # Clot Twist drives the floor, and the dial multiplies whatever it gives
    settings(smooth=1, twist=12.0)
    _, hi = shot_samples(CLOT)
    settings(smooth=1, twist=0.0)
    _, lo = shot_samples(CLOT)
    assert hi > lo
    settings(smooth=4, twist=0.0)
    assert shot_samples(CLOT)[1] == lo * 4

    print("ok")


if __name__ == "__main__":
    buried()
    demo()
