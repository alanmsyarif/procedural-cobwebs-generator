# Shared names used across Arachne modules.
#
# BUILD is drawn at the bottom of the N-panel and must match bl_info's
# version in __init__.py. Blender caches imported modules, so copying new
# files into the addons folder changes nothing until scripts are reloaded —
# the panel showing an older build than the Add-ons list is how that shows.
BUILD = "4.4.0 bake for render"

#
# NOTE: the `swf_` prefix on attributes and custom properties below is
# deliberate legacy. Those keys are written into .blend files (mesh
# attributes, object/scene properties), so renaming them would silently
# break every web saved before the Arachne rebrand — pins, solver state
# and dew droplets would all reset. Display names are rebranded; these
# storage keys stay put. Only rename them alongside a migration pass.

GROUP_SOLVER    = "Arachne Tearing Solver"
GROUP_STRANDIFY = "Arachne Strandify"
GROUP_GPU_APPLY = "Arachne GPU Apply"

# The baked variant of the apply group (see gpu_solver). Deliberately a
# suffix of GROUP_GPU_APPLY: every lookup of the apply modifier matches on
# that prefix, so a baked web is still "the web with an apply modifier" to
# the rest of the add-on.
GROUP_BAKE_APPLY = GROUP_GPU_APPLY + " Bake"

MAT_SILK    = "Arachne Silk"
MAT_DEW     = "Arachne Dew"
MAT_TENSION = "Arachne Tension"
MAT_SYNTH   = "Arachne Synth-Web"

# Attributes (prefixed to avoid collisions with user data)
A_PREV   = "swf_prev"       # POINT vector — previous position (verlet state)
A_REST   = "swf_rest"       # EDGE  float  — rest length captured at frame 1
A_PIN    = "swf_pin"        # POINT bool   — pinned/anchor vertices
A_CORR   = "swf_corr"       # EDGE  vector — per-edge correction (scratch)
A_ACCUM  = "swf_accum"      # POINT vector — accumulated correction (scratch)
A_TENS_E = "swf_tens_edge"  # EDGE  float  — normalized stretch (0=rest, 1=tear)
A_TENSION = "swf_tension"   # POINT float  — edge tension averaged to points
A_GPU_POS = "swf_gpu_pos"   # POINT vector — GPU solver positions writeback
A_BROKEN  = "swf_broken"    # EDGE  bool   — GPU solver torn-edge mask
A_BREAK_F = "swf_break_f"   # EDGE  float  — first frame this edge tore, or
                            #                BAKE_NEVER. Tearing is one-way
                            #                (the tear kernel skips an edge
                            #                already torn), so one frame
                            #                number per edge replaces a
                            #                per-frame broken-edge cache.
A_NOTEAR  = "swf_notear"    # EDGE  bool   — edge the solver must never tear
A_SHOT    = "swf_shot_t"    # POINT float  — frame the flying tip reaches
                            #                this point (web shot reveal)
A_EMIT    = "swf_emit"      # POINT bool   — muzzle anchor: rides the web
                            #                shot emitter object as it moves

# Mesh-datablock custom property: name of the object a Web Shot was fired
# from. The GPU solver looks it up to carry A_EMIT points along with it.
P_EMITTER = "swf_emitter"

# Mesh-datablock custom property: [[fire frame, x, y, z], ...] — where the
# emitter stood at each volley, baked at build time. The solver subtracts
# these to get each muzzle's offset, instead of re-deriving an animated
# emitter's position itself (see gpu_native._anchor_base).
P_EMIT_AT = "swf_emit_at"


# Baked web sim. The cache is a plain mesh of frames x vertices loose
# vertices, sampled per frame by the baked apply group. Loose vertices are
# not renderable geometry, so the cache object never shows up in a render.
P_BAKE_CACHE = "swf_bake_cache"   # object prop: name of that cache object
P_BAKE_START = "swf_bake_start"   # object prop: first baked frame
P_BAKE_END   = "swf_bake_end"     # object prop: last baked frame
BAKE_CACHE   = "Arachne Bake %s"  # cache object name, formatted with the web
BAKE_NEVER   = 1.0e9              # A_BREAK_F value for an edge that held
# Object custom property: name of the wind-field empty that feeds the web's
# pull into Blender's rigid body sim (see gpu_solver's pull field).
P_PULL = "swf_pull_field"

# Name of that empty, formatted with the web object's name
PULL_FIELD = "Arachne Pull %s"

# Dew droplet physics state (POINT domain, on the droplet point cloud)
A_DEW_HOME = "swf_dew_home"  # vector — birth position (respawn target)
A_DEW_SIZE = "swf_dew_size"  # float  — normalized size, drips around 1.0
A_DEW_FALL = "swf_dew_fall"  # bool   — droplet is in free fall
A_DEW_PREV = "swf_dew_prev"  # vector — previous position (verlet state)
A_DEW_RAND = "swf_dew_rand"  # float  — per-droplet random identity
A_DEW_NPOS = "swf_dew_npos"  # vector — next position (scratch)
A_DEW_RESP = "swf_dew_resp"  # bool   — respawning this frame (scratch)
A_DEW_DET  = "swf_dew_det"   # bool   — detaching this frame (scratch)
