# Shared names used across Arachne modules.
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

MAT_SILK    = "Arachne Silk"
MAT_DEW     = "Arachne Dew"
MAT_TENSION = "Arachne Tension"

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
A_NOTEAR  = "swf_notear"    # EDGE  bool   — edge the solver must never tear
A_SHOT    = "swf_shot_t"    # POINT float  — frame the flying tip reaches
                            #                this point (web shot reveal)

# Dew droplet physics state (POINT domain, on the droplet point cloud)
A_DEW_HOME = "swf_dew_home"  # vector — birth position (respawn target)
A_DEW_SIZE = "swf_dew_size"  # float  — normalized size, drips around 1.0
A_DEW_FALL = "swf_dew_fall"  # bool   — droplet is in free fall
A_DEW_PREV = "swf_dew_prev"  # vector — previous position (verlet state)
A_DEW_RAND = "swf_dew_rand"  # float  — per-droplet random identity
A_DEW_NPOS = "swf_dew_npos"  # vector — next position (scratch)
A_DEW_RESP = "swf_dew_resp"  # bool   — respawning this frame (scratch)
A_DEW_DET  = "swf_dew_det"   # bool   — detaching this frame (scratch)
