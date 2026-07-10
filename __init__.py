# ============================================================================
#  Spider Web Forge v2.0 — all-in-one procedural web toolkit
#  ---------------------------------------------------------------------------
#  * Generator  — procedural orb webs (radials, spiral, anchors) with pins
#                 pre-written into the swf_pin attribute
#  * Solver     — custom verlet/PBD Simulation-Zone solver with gravity,
#                 wind + turbulence, collision, friction, and TEARING
#  * Strandify  — silk tubes with noisy radius, Fresnel silk material,
#                 optional dew droplets
#
#  QUICK START
#  -----------
#  1. (Optional) select a mesh to use as the collider.
#  2. N-panel > Web Forge > "Create Web + Sim + Strands".
#  3. Rewind to frame 1 and press play. Push the collider through the web.
#
#  Notes: sim runs in the web object's local space (keep it un-rotated or
#  rotate the Gravity input); colliders need faces; raise Substeps for fast
#  colliders; bake via Physics properties > Simulation Nodes.
#
#  Blender 4.2+ / 5.x. Remove the old single-file Spider Web Forge v1
#  before installing this package to avoid duplicate panels.
# ============================================================================

bl_info = {
    "name": "Spider Web Forge",
    "author": "Amsy",
    "version": (2, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N) > Web Forge",
    "description": "Procedural orb webs with a tearing verlet solver, wind, "
                   "collision, silk strandify and dew",
    "category": "Add Mesh",
}

# -- submodule reload guard --------------------------------------------------
# Blender caches Python modules; without this, updating the addon files can
# leave old bytecode running (tracebacks then show mismatched source lines).
if "generator" in locals():
    import importlib
    for _m in (constants, nodeutils, materials, generator, solver,
               strandify, gpu_native, gpu_solver, ui):
        importlib.reload(_m)
else:
    from . import (constants, nodeutils, materials, generator, solver,
                   strandify, gpu_native, gpu_solver, ui)

_modules = (generator, solver, strandify, gpu_solver, ui)


def register():
    for m in _modules:
        m.register()


def unregister():
    for m in reversed(_modules):
        m.unregister()


if __name__ == "__main__":
    register()
