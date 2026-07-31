# ============================================================================
#  Arachne — procedural spider webs with a native GPU solver
#  ---------------------------------------------------------------------------
#  * Generator  — orb webs and chaotic spider-spun cobwebs (Pixar / Kole
#                 construction), pins pre-written into swf_pin
#  * GPU Solver — Blender-native GLSL compute (no dependencies): verlet
#                 with tension slack, unilateral silk constraints, wind,
#                 collision, friction, tearing, deteriorate, pre-warm
#  * Strandify  — silk or synth-web tubes, opt-in dripping dew physics,
#                 tension heatmap material
#
#  QUICK START: (optionally select a collider mesh) -> N-panel > Arachne
#  > "Generate Web" -> "Add GPU Solver" -> "Add Strandify" -> play from
#  frame 1.
#
#  Blender 4.2+ / 5.x.
# ============================================================================

# Keep `version` and constants.BUILD in step — the first is what Blender's
# Add-ons list shows, the second is drawn in the N-panel, and comparing the
# two is how you catch Blender still running a cached older copy.
bl_info = {
    "name": "Arachne",
    "author": "Amsy",
    "version": (4, 2, 1),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N) > Arachne",
    "description": "Procedural webs with a native-GPU tearing solver, "
                   "natural-silk or synthetic web-fluid strandify and "
                   "opt-in dripping dew. Web Shot strands fire in bursts "
                   "and stay stuck to their emitter",
    "category": "Add Mesh",
}

# -- submodule reload guard --------------------------------------------------
if "generator" in locals():
    import importlib
    for _m in (constants, nodeutils, materials, generator, gpu_native,
               gpu_solver, strandify, ui):
        importlib.reload(_m)
else:
    from . import (constants, nodeutils, materials, generator, gpu_native,
                   gpu_solver, strandify, ui)

_modules = (generator, gpu_solver, strandify, ui)


def register():
    for m in _modules:
        m.register()


def unregister():
    for m in reversed(_modules):
        m.unregister()


if __name__ == "__main__":
    register()
