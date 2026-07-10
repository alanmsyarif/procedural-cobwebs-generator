# ============================================================================
#  Spider Web Forge v2.2 (Lite) — generator + native GPU solver
#  ---------------------------------------------------------------------------
#  * Generator  — orb webs and chaotic spider-spun cobwebs (Pixar / Kole
#                 construction), pins pre-written into swf_pin
#  * GPU Solver — Blender-native GLSL compute (no dependencies): verlet
#                 with tension slack, unilateral silk constraints, wind,
#                 collision, friction, tearing, deteriorate, pre-warm
#  * Strandify  — silk tubes, dew droplets, tension heatmap material
#
#  QUICK START: (optionally select a collider mesh) -> N-panel > Web Forge
#  > "Create Web + Sim + Strands" -> play from frame 1.
#
#  Blender 4.2+ / 5.x.
# ============================================================================

bl_info = {
    "name": "Spider Web Forge",
    "author": "Amsy",
    "version": (2, 2, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N) > Web Forge",
    "description": "Procedural webs with a native-GPU tearing solver, "
                   "silk strandify and dew",
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
