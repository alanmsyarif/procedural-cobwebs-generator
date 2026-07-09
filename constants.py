# Shared names used across Spider Web Forge modules.

GROUP_SOLVER    = "SWF Tearing Solver"
GROUP_STRANDIFY = "SWF Strandify"

MAT_SILK = "SWF Silk"
MAT_DEW  = "SWF Dew"

# Attributes (prefixed to avoid collisions with user data)
A_PREV  = "swf_prev"    # POINT vector — previous position (verlet state)
A_REST  = "swf_rest"    # EDGE  float  — rest length captured at frame 1
A_PIN   = "swf_pin"     # POINT bool   — pinned/anchor vertices
A_CORR  = "swf_corr"    # EDGE  vector — per-edge correction (scratch)
A_ACCUM = "swf_accum"   # POINT vector — accumulated correction (scratch)
