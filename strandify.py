# Strandify: turns the (simulated, possibly torn) edge mesh into renderable
# silk geometry — curves with noisy radius swept into tubes — plus dew
# droplets with their own physics, simulated in a nested Simulation Zone:
#
#   * droplets cling to the silk tubes (nearest-surface snap each frame,
#     so they ride the deforming web)
#   * tangential gravity slides them around the tube to its underside and
#     along the strand toward low points — they visibly hang and pool
#   * condensation makes them grow; once weight beats their per-droplet
#     surface-tension threshold they detach and free-fall (verlet)
#   * droplets whose strand tears away are flung off
#   * a drop that has fallen "Drip Distance" below its birth spot respawns
#     there as a tiny bead, so the web keeps dripping
#
# Add this modifier AFTER the tearing solver so torn strands render too.

import bpy
from bpy.types import Operator

from .constants import (
    GROUP_STRANDIFY, A_DEW_HOME, A_DEW_SIZE, A_DEW_FALL, A_DEW_PREV,
    A_DEW_RAND, A_DEW_NPOS, A_DEW_RESP, A_DEW_DET,
)
from .nodeutils import H, sock_in, minmax_sockets, input_identifier
from .materials import (ensure_silk_material, ensure_dew_material,
                        ensure_tension_material)


def _build_group():
    nt = bpy.data.node_groups.new(GROUP_STRANDIFY, "GeometryNodeTree")
    iface = nt.interface
    h = H(nt)

    iface.new_socket(name="Geometry", in_out='INPUT',
                     socket_type='NodeSocketGeometry')
    sock_in(iface, "Strand Radius",      'NodeSocketFloat', 0.001,
            0.00005, 0.01)
    sock_in(iface, "Radius Variation",   'NodeSocketFloat', 0.4, 0.0, 1.0)
    sock_in(iface, "Profile Resolution", 'NodeSocketInt',   6, 3, 16)
    sock_in(iface, "Smooth Segments",    'NodeSocketInt',   4, 1, 12)
    sock_in(iface, "Silk Material",      'NodeSocketMaterial')
    sock_in(iface, "Show Tension",       'NodeSocketBool',  False)
    sock_in(iface, "Tension Material",   'NodeSocketMaterial')
    sock_in(iface, "Enable Dew",         'NodeSocketBool',  True)
    sock_in(iface, "Dew Per Span",       'NodeSocketFloat', 0.3, 0.0, 3.0)
    sock_in(iface, "Dew Amount",         'NodeSocketFloat', 0.25, 0.0, 1.0)
    sock_in(iface, "Dew Size",           'NodeSocketFloat', 0.004,
            0.0005, 0.05)
    sock_in(iface, "Dew Physics",        'NodeSocketBool',  True)
    sock_in(iface, "Dew Growth",         'NodeSocketFloat', 0.15, 0.0, 2.0)
    sock_in(iface, "Dew Slide",          'NodeSocketFloat', 0.3, 0.0, 1.0)
    sock_in(iface, "Drip Distance",      'NodeSocketFloat', 3.0, 0.1, 100.0)
    sock_in(iface, "Dew Material",       'NodeSocketMaterial')
    sock_in(iface, "Seed",               'NodeSocketInt',   0, 0, 100000)
    iface.new_socket(name="Geometry", in_out='OUTPUT',
                     socket_type='NodeSocketGeometry')

    gi = h.n("NodeGroupInput", -1200, 0)
    go = h.n("NodeGroupOutput", 6350, 0)
    g = gi.outputs

    # ---- strands -----------------------------------------------------------
    m2c = h.n("GeometryNodeMeshToCurve", -950, 100)
    h.lk(g["Geometry"], m2c.inputs["Mesh"])

    # Catmull-Rom interpolation smooths through the simulated points;
    # Smooth Segments = evaluated points per span (1 = angular/off)
    ctype = h.n("GeometryNodeCurveSplineType", -800, 100,
                label="smooth", spline_type='CATMULL_ROM')
    h.lk(m2c.outputs["Curve"], ctype.inputs["Curve"])
    cres = h.n("GeometryNodeSetSplineResolution", -680, 100)
    h.lk(ctype.outputs["Curve"], cres.inputs["Geometry"])
    h.lk(g["Smooth Segments"], cres.inputs["Resolution"])

    pos = h.n("GeometryNodeInputPosition", -900, -300)
    noise = h.n("ShaderNodeTexNoise", -700, -300, label="radius noise")
    h.lk(pos.outputs["Position"], noise.inputs["Vector"])
    noise.inputs["Scale"].default_value = 25.0

    cen = h.ma('SUBTRACT', -500, -300, noise.outputs["Fac"], 0.5)
    cen2 = h.ma('MULTIPLY', -350, -300, cen.outputs["Value"], 2.0)
    varied = h.ma('MULTIPLY', -200, -300,
                  cen2.outputs["Value"], g["Radius Variation"])
    plus1 = h.ma('ADD', -50, -300, varied.outputs["Value"], 1.0)
    radius = h.ma('MULTIPLY', 100, -300,
                  plus1.outputs["Value"], g["Strand Radius"],
                  label="strand radius")

    setrad = h.n("GeometryNodeSetCurveRadius", -550, 100)
    h.lk(cres.outputs["Geometry"], setrad.inputs["Curve"])
    h.lk(radius.outputs["Value"], setrad.inputs["Radius"])

    circle = h.n("GeometryNodeCurvePrimitiveCircle", -300, -100,
                 label="profile")
    h.lk(g["Profile Resolution"], circle.inputs["Resolution"])
    circle.inputs["Radius"].default_value = 0.001

    c2m = h.n("GeometryNodeCurveToMesh", 0, 100)
    h.lk(setrad.outputs["Curve"], c2m.inputs["Curve"])
    h.lk(circle.outputs["Curve"], c2m.inputs["Profile Curve"])

    mat_sw = h.n("GeometryNodeSwitch", 100, -100,
                 input_type='MATERIAL', label="tension view")
    h.lk(g["Show Tension"], mat_sw.inputs["Switch"])
    h.lk(g["Silk Material"], mat_sw.inputs["False"])
    h.lk(g["Tension Material"], mat_sw.inputs["True"])

    silk = h.n("GeometryNodeSetMaterial", 250, 100)
    h.lk(c2m.outputs["Mesh"], silk.inputs["Geometry"])
    h.lk(mat_sw.outputs["Output"], silk.inputs["Material"])

    # ---- dew: initial scatter (feeds the sim zone's frame-1 state) ---------
    # distribute by COUNT derived from control-point count (constant during
    # simulation) so droplet count never depends on deformed length
    slen = h.n("GeometryNodeSplineLength", -650, -700)
    spans = h.ma('SUBTRACT', -500, -700,
                 slen.outputs["Point Count"], 1.0)
    dcount = h.ma('MULTIPLY', -500, -900,
                  spans.outputs["Value"], g["Dew Per Span"])
    dround = h.ma('ROUND', -350, -900, dcount.outputs["Value"])
    dmax = h.ma('MAXIMUM', -350, -700, dround.outputs["Value"], 1.0)

    c2p = h.n("GeometryNodeCurveToPoints", -200, -700, mode='COUNT')
    h.lk(setrad.outputs["Curve"], c2p.inputs["Curve"])
    h.lk(dmax.outputs["Value"], c2p.inputs["Count"])

    rv = h.n("FunctionNodeRandomValue", -300, -1000, data_type='FLOAT')
    mn, mx = minmax_sockets(rv)
    mn.default_value = 0.0
    mx.default_value = 1.0
    h.lk(g["Seed"], rv.inputs["Seed"])

    drop = h.cmp('FLOAT', 'GREATER_THAN', -50, -1000,
                 rv.outputs["Value"], g["Dew Amount"], label="thin out")
    delete = h.n("GeometryNodeDeleteGeometry", 0, -700, domain='POINT')
    h.lk(c2p.outputs["Points"], delete.inputs["Geometry"])
    h.lk(drop.outputs["Result"], delete.inputs["Selection"])

    # per-droplet identity random in 0..1 — modulates growth rate, detach
    # threshold, and render size, and is re-hashed on every drip cycle
    seed2 = h.ma('ADD', -450, -1250, g["Seed"], 2.0)
    r_init = h.n("FunctionNodeRandomValue", -300, -1250, data_type='FLOAT')
    imn, imx = minmax_sockets(r_init)
    imn.default_value = 0.0
    imx.default_value = 1.0
    h.lk(seed2.outputs["Value"], r_init.inputs["Seed"])

    dpos0 = h.n("GeometryNodeInputPosition", 150, -1000)
    # birth size: decorrelated hash of the identity → 0.05..0.40
    sh0 = h.ma('MULTIPLY', 150, -1250, r_init.outputs["Value"], 3.77)
    sh1 = h.ma('ADD', 300, -1250, sh0.outputs["Value"], 0.211)
    sh2 = h.ma('FRACT', 450, -1250, sh1.outputs["Value"])
    sh3 = h.ma('MULTIPLY', 600, -1250, sh2.outputs["Value"], 0.35)
    size0 = h.ma('ADD', 750, -1250, sh3.outputs["Value"], 0.05)

    st_rand0 = h.store('FLOAT', 'POINT', A_DEW_RAND, 200, -700,
                       geo=delete.outputs["Geometry"],
                       value=r_init.outputs["Value"])
    st_home0 = h.store('FLOAT_VECTOR', 'POINT', A_DEW_HOME, 400, -700,
                       geo=st_rand0.outputs["Geometry"],
                       value=dpos0.outputs["Position"])
    st_size0 = h.store('FLOAT', 'POINT', A_DEW_SIZE, 600, -700,
                       geo=st_home0.outputs["Geometry"],
                       value=size0.outputs["Value"])
    st_fall0 = h.store('BOOLEAN', 'POINT', A_DEW_FALL, 800, -700,
                       geo=st_size0.outputs["Geometry"])  # starts attached
    st_dprev0 = h.store('FLOAT_VECTOR', 'POINT', A_DEW_PREV, 1000, -700,
                        geo=st_fall0.outputs["Geometry"],
                        value=dpos0.outputs["Position"])

    # ---- dew: droplet physics sim zone -------------------------------------
    dz_in, dz_out = h.sim_zone(1300, 4700, y=-700)
    h.lk(st_dprev0.outputs["Geometry"], dz_in.inputs["Geometry"])
    ddt = dz_in.outputs["Delta Time"]

    # state reads (evaluated per store node, ordering below keeps them "old")
    dpos = h.n("GeometryNodeInputPosition", 1500, -1100)
    dprev_a = h.named('FLOAT_VECTOR', A_DEW_PREV, 1500, -1250)
    dhome_a = h.named('FLOAT_VECTOR', A_DEW_HOME, 1500, -1400)
    dsize_a = h.named('FLOAT', A_DEW_SIZE, 1500, -1550)
    dfall_a = h.named('BOOLEAN', A_DEW_FALL, 1500, -1700)
    drand_a = h.named('FLOAT', A_DEW_RAND, 1500, -1850)

    # cling: snap to the nearest point on the silk tube surface (same
    # nearest-surface pattern the solver uses for collision), so attached
    # droplets ride the deforming web and survive topology changes
    tube_pos = h.n("GeometryNodeInputPosition", 1500, -2100)
    dsns_p = h.n("GeometryNodeSampleNearestSurface", 1750, -2000,
                 label="strand hit pos", data_type='FLOAT_VECTOR')
    h.lk(c2m.outputs["Mesh"], dsns_p.inputs["Mesh"])
    h.lk(tube_pos.outputs["Position"], dsns_p.inputs["Value"])
    h.lk(dpos.outputs["Position"], dsns_p.inputs["Sample Position"])

    tube_nrm = h.n("GeometryNodeInputNormal", 1500, -2400)
    dsns_n = h.n("GeometryNodeSampleNearestSurface", 1750, -2300,
                 label="strand hit normal", data_type='FLOAT_VECTOR')
    h.lk(c2m.outputs["Mesh"], dsns_n.inputs["Mesh"])
    h.lk(tube_nrm.outputs["Normal"], dsns_n.inputs["Value"])
    h.lk(dpos.outputs["Position"], dsns_n.inputs["Sample Position"])

    # strand torn away beneath the droplet? (nearest surface jumped far)
    hitd = h.vm('DISTANCE', 2050, -2000,
                dpos.outputs["Position"], dsns_p.outputs["Value"])
    lost = h.cmp('FLOAT', 'GREATER_THAN', 2250, -2000,
                 hitd.outputs["Value"], 0.15, label="strand lost")

    # tangential gravity  g_t = g - n(g.n)  slides the droplet around the
    # tube to its underside and along the strand toward low points
    gdn = h.vm('DOT_PRODUCT', 2050, -2300, (0.0, 0.0, -1.0),
               dsns_n.outputs["Value"])
    gnrm = h.vscale(2250, -2300, dsns_n.outputs["Value"],
                    gdn.outputs["Value"])
    gtan = h.vm('SUBTRACT', 2450, -2300, (0.0, 0.0, -1.0),
                gnrm.outputs["Vector"])
    # slide speed ~ 0.5 m/s * size * Dew Slide (heavy drops slide sooner)
    slide0 = h.ma('MULTIPLY', 2450, -2500, dsize_a.outputs["Attribute"],
                  g["Dew Slide"])
    slide1 = h.ma('MULTIPLY', 2650, -2500, slide0.outputs["Value"], 0.5)
    phys_gate = h.ma('MULTIPLY', 2650, -2700, slide1.outputs["Value"],
                     g["Dew Physics"], label="physics gate")
    slide2 = h.ma('MULTIPLY', 2850, -2500, phys_gate.outputs["Value"], ddt)
    slide = h.vscale(3050, -2400, gtan.outputs["Vector"],
                     slide2.outputs["Value"], label="slide step")
    p_att = h.vm('ADD', 3250, -2300, dsns_p.outputs["Value"],
                 slide.outputs["Vector"], label="attached pos")

    # free fall: verlet with slight drag
    velv = h.vm('SUBTRACT', 2050, -1250, dpos.outputs["Position"],
                dprev_a.outputs["Attribute"])
    veld = h.vscale(2250, -1250, velv.outputs["Vector"], 0.995)
    dt2d = h.ma('MULTIPLY', 2050, -1450, ddt, ddt)
    gz = h.ma('MULTIPLY', 2250, -1450, dt2d.outputs["Value"], -9.81)
    gvec = h.n("ShaderNodeCombineXYZ", 2450, -1450)
    h.lk(gz.outputs["Value"], gvec.inputs["Z"])
    pf0 = h.vm('ADD', 2450, -1250, dpos.outputs["Position"],
               veld.outputs["Vector"])
    p_fall = h.vm('ADD', 2650, -1250, pf0.outputs["Vector"],
                  gvec.outputs["Vector"], label="fall pos")

    # detach: weight beats surface tension (per-droplet threshold), or the
    # strand tore away — only while physics is on and still attached
    th0 = h.ma('MULTIPLY', 2050, -1700, drand_a.outputs["Attribute"], 0.6)
    thresh = h.ma('ADD', 2250, -1700, th0.outputs["Value"], 0.7,
                  label="detach threshold")
    heavy = h.cmp('FLOAT', 'GREATER_EQUAL', 2450, -1700,
                  dsize_a.outputs["Attribute"], thresh.outputs["Value"],
                  label="too heavy")
    hl = h.bmath('OR', 2650, -1750, heavy.outputs["Result"],
                 lost.outputs["Result"])
    nfall = h.bmath('NOT', 2450, -1900, dfall_a.outputs["Attribute"])
    det0 = h.bmath('AND', 2850, -1800, hl.outputs["Boolean"],
                   nfall.outputs["Boolean"])
    det = h.bmath('AND', 3050, -1800, det0.outputs["Boolean"],
                  g["Dew Physics"])

    # respawn: fallen Drip Distance below home, and home strand still there
    sep_h = h.n("ShaderNodeSeparateXYZ", 2050, -3000)
    h.lk(dhome_a.outputs["Attribute"], sep_h.inputs["Vector"])
    sep_p = h.n("ShaderNodeSeparateXYZ", 2050, -3150)
    h.lk(dpos.outputs["Position"], sep_p.inputs["Vector"])
    below = h.ma('SUBTRACT', 2250, -3050, sep_h.outputs["Z"],
                 sep_p.outputs["Z"], label="fall depth")
    fell = h.cmp('FLOAT', 'GREATER_THAN', 2450, -3050,
                 below.outputs["Value"], g["Drip Distance"])
    hsns = h.n("GeometryNodeSampleNearestSurface", 2050, -3350,
               label="home hit pos", data_type='FLOAT_VECTOR')
    h.lk(c2m.outputs["Mesh"], hsns.inputs["Mesh"])
    h.lk(tube_pos.outputs["Position"], hsns.inputs["Value"])
    h.lk(dhome_a.outputs["Attribute"], hsns.inputs["Sample Position"])
    hdist = h.vm('DISTANCE', 2350, -3350, dhome_a.outputs["Attribute"],
                 hsns.outputs["Value"])
    home_ok = h.cmp('FLOAT', 'LESS_THAN', 2550, -3350,
                    hdist.outputs["Value"], 0.15, label="home intact")
    resp0 = h.bmath('AND', 2750, -3150, dfall_a.outputs["Attribute"],
                    fell.outputs["Result"])
    resp = h.bmath('AND', 2950, -3150, resp0.outputs["Boolean"],
                   home_ok.outputs["Result"])

    # next position: respawn at home / keep falling / cling to strand
    sw_fall = h.n("GeometryNodeSwitch", 3450, -1500, input_type='VECTOR',
                  label="falling?")
    h.lk(dfall_a.outputs["Attribute"], sw_fall.inputs["Switch"])
    h.lk(p_att.outputs["Vector"], sw_fall.inputs["False"])
    h.lk(p_fall.outputs["Vector"], sw_fall.inputs["True"])
    sw_pos = h.n("GeometryNodeSwitch", 3650, -1500, input_type='VECTOR',
                 label="respawn?")
    h.lk(resp.outputs["Boolean"], sw_pos.inputs["Switch"])
    h.lk(sw_fall.outputs["Output"], sw_pos.inputs["False"])
    h.lk(dhome_a.outputs["Attribute"], sw_pos.inputs["True"])

    # ---- state update chain: every Store's field must read pre-update
    # values, so decisions are committed to scratch attrs first and each
    # attr is written only after its last "old" read
    st_npos = h.store('FLOAT_VECTOR', 'POINT', A_DEW_NPOS, 1700, -700,
                      geo=dz_in.outputs["Geometry"],
                      value=sw_pos.outputs["Output"])
    st_resp = h.store('BOOLEAN', 'POINT', A_DEW_RESP, 1900, -700,
                      geo=st_npos.outputs["Geometry"],
                      value=resp.outputs["Boolean"])
    st_det = h.store('BOOLEAN', 'POINT', A_DEW_DET, 2100, -700,
                     geo=st_resp.outputs["Geometry"],
                     value=det.outputs["Boolean"])

    dresp_a = h.named('BOOLEAN', A_DEW_RESP, 2300, -1100)
    ddet_a = h.named('BOOLEAN', A_DEW_DET, 2300, -1250)
    dnpos_a = h.named('FLOAT_VECTOR', A_DEW_NPOS, 2300, -1400)

    # prev: falling keeps verlet history; attaching/respawning resets it
    nresp = h.bmath('NOT', 3150, -3000, dresp_a.outputs["Attribute"])
    keepv = h.bmath('AND', 3350, -2950, dfall_a.outputs["Attribute"],
                    nresp.outputs["Boolean"])
    sw_prev = h.n("GeometryNodeSwitch", 3550, -2950, input_type='VECTOR',
                  label="prev pos")
    h.lk(keepv.outputs["Boolean"], sw_prev.inputs["Switch"])
    h.lk(dnpos_a.outputs["Attribute"], sw_prev.inputs["False"])
    h.lk(dpos.outputs["Position"], sw_prev.inputs["True"])
    st_dprev = h.store('FLOAT_VECTOR', 'POINT', A_DEW_PREV, 2300, -700,
                       geo=st_det.outputs["Geometry"],
                       value=sw_prev.outputs["Output"])

    # size: condensation growth while attached; reborn small on respawn
    gr0 = h.ma('MULTIPLY', 3150, -3550, drand_a.outputs["Attribute"], 5.19)
    gr1 = h.ma('ADD', 3300, -3550, gr0.outputs["Value"], 0.37)
    gr2 = h.ma('FRACT', 3450, -3550, gr1.outputs["Value"])
    gr3 = h.ma('MULTIPLY', 3600, -3550, gr2.outputs["Value"], 0.8)
    grate = h.ma('ADD', 3750, -3550, gr3.outputs["Value"], 0.6,
                 label="growth variance")
    gg0 = h.ma('MULTIPLY', 3150, -3750, g["Dew Growth"], g["Dew Physics"])
    gg1 = h.ma('MULTIPLY', 3300, -3750, gg0.outputs["Value"], ddt)
    gg2 = h.ma('MULTIPLY', 3450, -3750, gg1.outputs["Value"],
               grate.outputs["Value"])
    gact = h.ma('MULTIPLY', 3600, -3750, gg2.outputs["Value"],
                nfall.outputs["Boolean"], label="growth (attached)")
    grown = h.ma('ADD', 3750, -3750, dsize_a.outputs["Attribute"],
                 gact.outputs["Value"])
    rs0 = h.ma('MULTIPLY', 3600, -3950, drand_a.outputs["Attribute"], 3.77)
    rs1 = h.ma('ADD', 3750, -3950, rs0.outputs["Value"], 0.211)
    rs2 = h.ma('FRACT', 3900, -3950, rs1.outputs["Value"])
    rs3 = h.ma('MULTIPLY', 4050, -3950, rs2.outputs["Value"], 0.35)
    rsize = h.ma('ADD', 4200, -3950, rs3.outputs["Value"], 0.05,
                 label="reborn size")
    sw_size = h.n("GeometryNodeSwitch", 4200, -3750, input_type='FLOAT',
                  label="size")
    h.lk(dresp_a.outputs["Attribute"], sw_size.inputs["Switch"])
    h.lk(grown.outputs["Value"], sw_size.inputs["False"])
    h.lk(rsize.outputs["Value"], sw_size.inputs["True"])
    st_dsize = h.store('FLOAT', 'POINT', A_DEW_SIZE, 2500, -700,
                       geo=st_dprev.outputs["Geometry"],
                       value=sw_size.outputs["Output"])

    # falling flag
    fd = h.bmath('OR', 3350, -3250, dfall_a.outputs["Attribute"],
                 ddet_a.outputs["Attribute"])
    fnew = h.bmath('AND', 3550, -3250, fd.outputs["Boolean"],
                   nresp.outputs["Boolean"])
    st_dfall = h.store('BOOLEAN', 'POINT', A_DEW_FALL, 2700, -700,
                       geo=st_dsize.outputs["Geometry"],
                       value=fnew.outputs["Boolean"])

    # identity: re-hash each drip cycle so thresholds/sizes vary
    rh0 = h.ma('ADD', 3900, -3350, drand_a.outputs["Attribute"], 0.618034)
    rh1 = h.ma('FRACT', 4050, -3350, rh0.outputs["Value"])
    sw_rand = h.n("GeometryNodeSwitch", 4200, -3350, input_type='FLOAT',
                  label="identity")
    h.lk(dresp_a.outputs["Attribute"], sw_rand.inputs["Switch"])
    h.lk(drand_a.outputs["Attribute"], sw_rand.inputs["False"])
    h.lk(rh1.outputs["Value"], sw_rand.inputs["True"])
    st_drand = h.store('FLOAT', 'POINT', A_DEW_RAND, 2900, -700,
                       geo=st_dfall.outputs["Geometry"],
                       value=sw_rand.outputs["Output"])

    sp_dew = h.n("GeometryNodeSetPosition", 3100, -700, label="move droplets")
    h.lk(st_drand.outputs["Geometry"], sp_dew.inputs["Geometry"])
    h.lk(dnpos_a.outputs["Attribute"], sp_dew.inputs["Position"])
    h.lk(sp_dew.outputs["Geometry"], dz_out.inputs["Geometry"])

    # ---- dew: render --------------------------------------------------------
    # cull drops that fell past the recycle depth with no home to return to
    size_r = h.named('FLOAT', A_DEW_SIZE, 4900, -1100)
    rand_r = h.named('FLOAT', A_DEW_RAND, 4900, -1250)
    fall_r = h.named('BOOLEAN', A_DEW_FALL, 4900, -1400)
    home_r = h.named('FLOAT_VECTOR', A_DEW_HOME, 4900, -1550)
    pos_r = h.n("GeometryNodeInputPosition", 4900, -1700)
    csep_h = h.n("ShaderNodeSeparateXYZ", 5100, -1550)
    h.lk(home_r.outputs["Attribute"], csep_h.inputs["Vector"])
    csep_p = h.n("ShaderNodeSeparateXYZ", 5100, -1700)
    h.lk(pos_r.outputs["Position"], csep_p.inputs["Vector"])
    cbelow = h.ma('SUBTRACT', 5300, -1600, csep_h.outputs["Z"],
                  csep_p.outputs["Z"])
    cdepth = h.ma('MULTIPLY', 5300, -1800, g["Drip Distance"], 1.2)
    cpast = h.cmp('FLOAT', 'GREATER_THAN', 5500, -1650,
                  cbelow.outputs["Value"], cdepth.outputs["Value"])
    cull = h.bmath('AND', 5700, -1600, cpast.outputs["Result"],
                   fall_r.outputs["Attribute"])
    cdel = h.n("GeometryNodeDeleteGeometry", 5000, -700, domain='POINT',
               label="cull lost drops")
    h.lk(dz_out.outputs["Geometry"], cdel.inputs["Geometry"])
    h.lk(cull.outputs["Boolean"], cdel.inputs["Selection"])

    ico = h.n("GeometryNodeMeshIcoSphere", 5000, -1000, label="droplet")
    ico.inputs["Radius"].default_value = 1.0
    ico.inputs["Subdivisions"].default_value = 2

    # render radius grows with simulated size; falling drops stretch into
    # a teardrop along world Z
    sb0 = h.ma('MULTIPLY', 5500, -1100, size_r.outputs["Attribute"], 1.5)
    sb1 = h.ma('ADD', 5650, -1100, sb0.outputs["Value"], 0.5)
    vb0 = h.ma('MULTIPLY', 5500, -1250, rand_r.outputs["Attribute"], 0.6)
    vb1 = h.ma('ADD', 5650, -1250, vb0.outputs["Value"], 0.7)
    sfac = h.ma('MULTIPLY', 5800, -1150, sb1.outputs["Value"],
                vb1.outputs["Value"])
    sbase = h.ma('MULTIPLY', 5950, -1150, sfac.outputs["Value"],
                 g["Dew Size"], label="droplet radius")
    sw_x = h.n("GeometryNodeSwitch", 5800, -1400, input_type='FLOAT',
               label="squash xy")
    h.lk(fall_r.outputs["Attribute"], sw_x.inputs["Switch"])
    sw_x.inputs["False"].default_value = 1.0
    sw_x.inputs["True"].default_value = 0.85
    sw_z = h.n("GeometryNodeSwitch", 5800, -1550, input_type='FLOAT',
               label="stretch z")
    h.lk(fall_r.outputs["Attribute"], sw_z.inputs["Switch"])
    sw_z.inputs["False"].default_value = 1.0
    sw_z.inputs["True"].default_value = 1.3
    sxy = h.ma('MULTIPLY', 6100, -1300, sbase.outputs["Value"],
               sw_x.outputs["Output"])
    sz = h.ma('MULTIPLY', 6100, -1500, sbase.outputs["Value"],
              sw_z.outputs["Output"])
    dscale = h.n("ShaderNodeCombineXYZ", 6250, -1400, label="drop scale")
    h.lk(sxy.outputs["Value"], dscale.inputs["X"])
    h.lk(sxy.outputs["Value"], dscale.inputs["Y"])
    h.lk(sz.outputs["Value"], dscale.inputs["Z"])

    iop = h.n("GeometryNodeInstanceOnPoints", 5250, -700)
    h.lk(cdel.outputs["Geometry"], iop.inputs["Points"])
    h.lk(ico.outputs["Mesh"], iop.inputs["Instance"])
    h.lk(dscale.outputs["Vector"], iop.inputs["Scale"])

    real = h.n("GeometryNodeRealizeInstances", 5500, -700)
    h.lk(iop.outputs["Instances"], real.inputs["Geometry"])

    dewmat = h.n("GeometryNodeSetMaterial", 5700, -700)
    h.lk(real.outputs["Geometry"], dewmat.inputs["Geometry"])
    h.lk(g["Dew Material"], dewmat.inputs["Material"])

    dew_switch = h.n("GeometryNodeSwitch", 5900, -400,
                     input_type='GEOMETRY', label="dew on/off")
    h.lk(g["Enable Dew"], dew_switch.inputs["Switch"])
    h.lk(dewmat.outputs["Geometry"], dew_switch.inputs["True"])

    join = h.n("GeometryNodeJoinGeometry", 6100, 0)
    h.lk(dew_switch.outputs["Output"], join.inputs["Geometry"])
    h.lk(silk.outputs["Geometry"], join.inputs["Geometry"])
    h.lk(join.outputs["Geometry"], go.inputs["Geometry"])
    return nt


STRANDIFY_VERSION = 5


def ensure_strandify_group():
    nt = bpy.data.node_groups.get(GROUP_STRANDIFY)
    if nt is not None:
        if nt.get("swf_version", 0) >= STRANDIFY_VERSION:
            return nt
        nt.name = GROUP_STRANDIFY + ".old"
    nt = _build_group()
    nt["swf_version"] = STRANDIFY_VERSION
    return nt


def apply_strandify(obj):
    """Add the strandify modifier with silk/dew materials pre-assigned."""
    group = ensure_strandify_group()
    mod = obj.modifiers.new("SWF Strandify", 'NODES')
    mod.node_group = group
    for sock_name, mat in (("Silk Material", ensure_silk_material()),
                           ("Dew Material", ensure_dew_material()),
                           ("Tension Material", ensure_tension_material())):
        ident = input_identifier(group, sock_name)
        if ident:
            try:
                mod[ident] = mat
            except (KeyError, TypeError):
                pass
    return mod


class SWF_OT_bake_dew(Operator):
    """Bake the dew droplet simulation to disk so renders replay it
    exactly (the render depsgraph can't reuse the viewport's live sim).
    Stepping through the frames also fills the web solver's render
    cache. Save the .blend first — bakes are stored beside it"""
    bl_idname = "swf.bake_dew"
    bl_label = "Bake Dew for Render"

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == 'MESH'

    def execute(self, context):
        if not bpy.data.is_saved:
            self.report({'ERROR'},
                        "Save the .blend file first — simulation bakes are "
                        "stored next to it.")
            return {'CANCELLED'}
        scene = context.scene
        frame = scene.frame_current
        try:
            bpy.ops.object.simulation_nodes_cache_bake(selected=False)
        except RuntimeError as ex:
            self.report({'ERROR'}, "Bake failed: %s" % ex)
            return {'CANCELLED'}
        finally:
            scene.frame_set(frame)
        self.report({'INFO'},
                    "Dew simulation baked — renders now replay it exactly.")
        return {'FINISHED'}


class SWF_OT_free_dew_bake(Operator):
    """Delete the baked dew simulation so it simulates live again
    (edit dew settings, then re-bake before rendering)"""
    bl_idname = "swf.free_dew_bake"
    bl_label = "Free Dew Bake"

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == 'MESH'

    def execute(self, context):
        try:
            bpy.ops.object.simulation_nodes_cache_delete(selected=False)
        except RuntimeError as ex:
            self.report({'ERROR'}, "Couldn't free bake: %s" % ex)
            return {'CANCELLED'}
        return {'FINISHED'}


class SWF_OT_add_strandify(Operator):
    """Add the strandify modifier (silk tubes + dew) to the active object"""
    bl_idname = "swf.add_strandify"
    bl_label = "Add Strandify"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == 'MESH'

    def execute(self, context):
        apply_strandify(context.object)
        return {'FINISHED'}


classes = (SWF_OT_add_strandify, SWF_OT_bake_dew, SWF_OT_free_dew_bake)


def _safe_register(cls):
    """Register defensively: if a class with this name survived a failed
    or partial previous enable, evict it first."""
    old = getattr(bpy.types, cls.__name__, None)
    if old is not None:
        try:
            bpy.utils.unregister_class(old)
        except RuntimeError:
            pass
    bpy.utils.register_class(cls)


def register():
    for c in classes:
        _safe_register(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
