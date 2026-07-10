# Shared names used across Spider Web Forge modules.

GROUP_SOLVER    = "SWF Tearing Solver"
GROUP_STRANDIFY = "SWF Strandify"
GROUP_GPU_APPLY = "SWF GPU Apply"

MAT_SILK    = "SWF Silk"
MAT_DEW     = "SWF Dew"
MAT_TENSION = "SWF Tension"

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
