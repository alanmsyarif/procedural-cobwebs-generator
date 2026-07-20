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

# Dew droplet physics state (POINT domain, on the droplet point cloud)
A_DEW_HOME = "swf_dew_home"  # vector — birth position (respawn target)
A_DEW_SIZE = "swf_dew_size"  # float  — normalized size, drips around 1.0
A_DEW_FALL = "swf_dew_fall"  # bool   — droplet is in free fall
A_DEW_PREV = "swf_dew_prev"  # vector — previous position (verlet state)
A_DEW_RAND = "swf_dew_rand"  # float  — per-droplet random identity
A_DEW_NPOS = "swf_dew_npos"  # vector — next position (scratch)
A_DEW_RESP = "swf_dew_resp"  # bool   — respawning this frame (scratch)
A_DEW_DET  = "swf_dew_det"   # bool   — detaching this frame (scratch)
