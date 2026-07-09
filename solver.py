# Verlet / position-based-dynamics web solver with TEARING, built inside
# Geometry Nodes Simulation Zones.
#
#   * Verlet integration with damping, gravity, and wind (steady + turbulent)
#   * Edge distance constraints, relaxed iteratively
#   * Collision against a mesh object via nearest-surface projection
#   * Friction on contact
#   * Tearing: edges stretched past rest_length * threshold are deleted,
#     freed strands recoil on subsequent frames
#   * Optional cleanup of points orphaned by torn edges
#
# Structure: Simulation Zone > substep Repeat Zone > constraint Repeat Zone
# > valence-gather Repeat Zone (scatters per-edge corrections to points via
# Edges of Vertex, robust across tearing topology changes).

import bpy
from bpy.types import Operator

from .constants import (
    GROUP_SOLVER, A_PREV, A_REST, A_PIN, A_CORR, A_ACCUM,
)
from .nodeutils import H, sock_in, input_identifier


def _build_group():
    nt = bpy.data.node_groups.new(GROUP_SOLVER, "GeometryNodeTree")
    iface = nt.interface
    h = H(nt)

    # ---- interface -------------------------------------------------------
    iface.new_socket(name="Geometry", in_out='INPUT',
                     socket_type='NodeSocketGeometry')
    sock_in(iface, "Pin Vertices",        'NodeSocketBool',   False)
    sock_in(iface, "Collision Object",    'NodeSocketObject')
    sock_in(iface, "Enable Collision",    'NodeSocketBool',   True)
    sock_in(iface, "Collision Offset",    'NodeSocketFloat',  0.01, 0.0, 1.0)
    sock_in(iface, "Friction",            'NodeSocketFloat',  0.5, 0.0, 1.0)
    sock_in(iface, "Gravity",             'NodeSocketVector', (0.0, 0.0, -9.81))
    sock_in(iface, "Wind",                'NodeSocketVector', (0.0, 0.0, 0.0))
    sock_in(iface, "Wind Turbulence",     'NodeSocketFloat',  0.5, 0.0, 50.0)
    sock_in(iface, "Damping",             'NodeSocketFloat',  0.99, 0.0, 1.0)
    sock_in(iface, "Stiffness",           'NodeSocketFloat',  1.0, 0.0, 1.0)
    sock_in(iface, "Iterations",          'NodeSocketInt',    8, 1, 64)
    sock_in(iface, "Substeps",            'NodeSocketInt',    3, 1, 16)
    sock_in(iface, "Max Valence",         'NodeSocketInt',    16, 1, 64)
    sock_in(iface, "Enable Tearing",      'NodeSocketBool',   True)
    sock_in(iface, "Tear Threshold",      'NodeSocketFloat',  1.5, 1.01, 10.0)
    sock_in(iface, "Remove Loose Points", 'NodeSocketBool',   True)
    iface.new_socket(name="Geometry", in_out='OUTPUT',
                     socket_type='NodeSocketGeometry')

    gi = h.n("NodeGroupInput", -2400, 0)
    go = h.n("NodeGroupOutput", 8400, 0)
    g = gi.outputs

    # =====================================================================
    #  PREP (only frame 1 matters — the zone feeds itself afterwards)
    # =====================================================================
    ev0 = h.n("GeometryNodeInputMeshEdgeVertices", -2400, -400)
    d0 = h.vm('DISTANCE', -2200, -400,
              ev0.outputs["Position 1"], ev0.outputs["Position 2"],
              label="Rest Length")
    st_rest = h.store('FLOAT', 'EDGE', A_REST, -2000, 0,
                      geo=g["Geometry"], value=d0.outputs["Value"])

    pos0 = h.n("GeometryNodeInputPosition", -2000, -400)
    st_prev0 = h.store('FLOAT_VECTOR', 'POINT', A_PREV, -1800, 0,
                       geo=st_rest.outputs["Geometry"],
                       value=pos0.outputs["Position"])
    st_pin0 = h.store('BOOLEAN', 'POINT', A_PIN, -1600, 0,
                      geo=st_prev0.outputs["Geometry"],
                      value=g["Pin Vertices"])

    # =====================================================================
    #  SIMULATION ZONE
    # =====================================================================
    sim_in, sim_out = h.sim_zone(-1400, 8000)
    h.lk(st_pin0.outputs["Geometry"], sim_in.inputs["Geometry"])

    # dt per substep and force displacements (g + wind) * dt^2
    dt_sub = h.ma('DIVIDE', -1200, -400,
                  sim_in.outputs["Delta Time"], g["Substeps"],
                  label="dt / substeps")
    dt2 = h.ma('MULTIPLY', -1000, -400,
               dt_sub.outputs["Value"], dt_sub.outputs["Value"], label="dt^2")

    # wind turbulence: animated 4D noise, centered and scaled
    stime = h.n("GeometryNodeInputSceneTime", -1400, -800)
    pos_w = h.n("GeometryNodeInputPosition", -1400, -1000)
    noise = h.n("ShaderNodeTexNoise", -1200, -850, label="wind noise",
                noise_dimensions='4D')
    h.lk(pos_w.outputs["Position"], noise.inputs["Vector"])
    h.lk(stime.outputs["Seconds"], noise.inputs["W"])
    noise.inputs["Scale"].default_value = 1.2

    centered = h.vm('SUBTRACT', -1000, -850, noise.outputs["Color"],
                    (0.5, 0.5, 0.5))
    turb_amt = h.ma('MULTIPLY', -1000, -1050, g["Wind Turbulence"], 2.0)
    turb = h.vscale(-800, -850, centered.outputs["Vector"],
                    turb_amt.outputs["Value"], label="turbulence")
    wind_total = h.vm('ADD', -800, -600, g["Wind"], turb.outputs["Vector"])
    force = h.vm('ADD', -600, -500, g["Gravity"],
                 wind_total.outputs["Vector"], label="g + wind")
    fdisp = h.vscale(-400, -500, force.outputs["Vector"],
                     dt2.outputs["Value"], label="force * dt^2")

    # collider geometry, in the modifier object's space
    obj_info = h.n("GeometryNodeObjectInfo", -1200, -1300,
                   transform_space='RELATIVE')
    h.lk(g["Collision Object"], obj_info.inputs["Object"])

    # ---------------------------------------------------------------------
    #  SUBSTEP LOOP
    # ---------------------------------------------------------------------
    r1_in, r1_out = h.repeat_zone(-800, 6800, label="Substeps")
    h.lk(sim_in.outputs["Geometry"], r1_in.inputs["Geometry"])
    h.lk(g["Substeps"], r1_in.inputs["Iterations"])

    prev1 = h.named('FLOAT_VECTOR', A_PREV, -600, -700)
    pos1 = h.n("GeometryNodeInputPosition", -600, -900)
    vel = h.vm('SUBTRACT', -400, -700,
               pos1.outputs["Position"], prev1.outputs["Attribute"],
               label="velocity")
    veld = h.vscale(-200, -700, vel.outputs["Vector"], g["Damping"],
                    label="damped vel")
    p_step = h.vm('ADD', 0, -700, pos1.outputs["Position"],
                  veld.outputs["Vector"])
    p_new = h.vm('ADD', 200, -700, p_step.outputs["Vector"],
                 fdisp.outputs["Vector"], label="integrated pos")

    st_prev1 = h.store('FLOAT_VECTOR', 'POINT', A_PREV, -400, 100,
                       geo=r1_in.outputs["Geometry"],
                       value=pos1.outputs["Position"])

    pin1 = h.named('BOOLEAN', A_PIN, -200, 300)
    npin1 = h.bmath('NOT', 0, 300, pin1.outputs["Attribute"])

    sp_verlet = h.n("GeometryNodeSetPosition", 200, 100, label="Verlet step")
    h.lk(st_prev1.outputs["Geometry"], sp_verlet.inputs["Geometry"])
    h.lk(npin1.outputs["Boolean"], sp_verlet.inputs["Selection"])
    h.lk(p_new.outputs["Vector"], sp_verlet.inputs["Position"])

    # ---------------------------------------------------------------------
    #  CONSTRAINT LOOP
    # ---------------------------------------------------------------------
    r2_in, r2_out = h.repeat_zone(500, 6400, label="Constraint iters")
    h.lk(sp_verlet.outputs["Geometry"], r2_in.inputs["Geometry"])
    h.lk(g["Iterations"], r2_in.inputs["Iterations"])

    st_acc0 = h.store('FLOAT_VECTOR', 'POINT', A_ACCUM, 800, 100,
                      geo=r2_in.outputs["Geometry"])  # resets to (0,0,0)

    ev2 = h.n("GeometryNodeInputMeshEdgeVertices", 800, -600)
    delta = h.vm('SUBTRACT', 1000, -600,
                 ev2.outputs["Position 2"], ev2.outputs["Position 1"],
                 label="edge delta")
    dist = h.vm('LENGTH', 1200, -600, delta.outputs["Vector"])
    rest2 = h.named('FLOAT', A_REST, 1200, -800)
    stretch = h.ma('SUBTRACT', 1400, -600,
                   dist.outputs["Value"], rest2.outputs["Attribute"],
                   label="stretch")
    dsafe = h.ma('MAXIMUM', 1400, -800, dist.outputs["Value"], 1e-6)
    ratio = h.ma('DIVIDE', 1600, -600,
                 stretch.outputs["Value"], dsafe.outputs["Value"])
    k_half = h.ma('MULTIPLY', 1600, -900, g["Stiffness"], 0.5)
    factor = h.ma('MULTIPLY', 1800, -600,
                  ratio.outputs["Value"], k_half.outputs["Value"])
    corr = h.vscale(2000, -600, delta.outputs["Vector"],
                    factor.outputs["Value"], label="edge correction")

    st_corr = h.store('FLOAT_VECTOR', 'EDGE', A_CORR, 1100, 100,
                      geo=st_acc0.outputs["Geometry"],
                      value=corr.outputs["Vector"])

    # ---- valence gather loop: scatter edge corrections to points ----
    r3_in, r3_out = h.repeat_zone(1400, 3400, label="Valence gather")
    h.lk(st_corr.outputs["Geometry"], r3_in.inputs["Geometry"])
    h.lk(g["Max Valence"], r3_in.inputs["Iterations"])

    eov = h.n("GeometryNodeEdgesOfVertex", 1700, -400)
    h.lk(r3_in.outputs["Iteration"], eov.inputs["Sort Index"])

    valid = h.cmp('INT', 'LESS_THAN', 1950, -300,
                  r3_in.outputs["Iteration"], eov.outputs["Total"],
                  label="k < valence")

    corr_named = h.named('FLOAT_VECTOR', A_CORR, 1700, -650)
    si_corr = h.sample('FLOAT_VECTOR', 'EDGE', 1950, -650,
                       r3_in.outputs["Geometry"],
                       corr_named.outputs["Attribute"],
                       eov.outputs["Edge Index"], label="sample corr")

    ev3 = h.n("GeometryNodeInputMeshEdgeVertices", 1700, -900)
    si_v1 = h.sample('INT', 'EDGE', 1950, -900,
                     r3_in.outputs["Geometry"],
                     ev3.outputs["Vertex Index 1"],
                     eov.outputs["Edge Index"], label="sample v1 idx")

    self_idx = h.n("GeometryNodeInputIndex", 1950, -1100)
    is_v1 = h.cmp('INT', 'EQUAL', 2200, -900,
                  si_v1.outputs["Value"], self_idx.outputs["Index"],
                  label="am I v1?")

    sgn = h.n("GeometryNodeSwitch", 2450, -900, label="sign",
              input_type='FLOAT')
    h.lk(is_v1.outputs["Result"], sgn.inputs["Switch"])
    sgn.inputs["False"].default_value = -1.0
    sgn.inputs["True"].default_value = 1.0

    mask = h.ma('MULTIPLY', 2700, -700,
                sgn.outputs["Output"], valid.outputs["Result"],
                label="sign * valid")
    contrib = h.vscale(2900, -700, si_corr.outputs["Value"],
                       mask.outputs["Value"], label="contribution")

    acc_prev = h.named('FLOAT_VECTOR', A_ACCUM, 2900, -400)
    acc_new = h.vm('ADD', 3100, -400,
                   acc_prev.outputs["Attribute"], contrib.outputs["Vector"])

    st_acc = h.store('FLOAT_VECTOR', 'POINT', A_ACCUM, 3150, 100,
                     geo=r3_in.outputs["Geometry"],
                     value=acc_new.outputs["Vector"])
    h.lk(st_acc.outputs["Geometry"], r3_out.inputs["Geometry"])

    # ---- apply accumulated corrections (Jacobi-averaged by valence) ----
    # Summing all edge corrections at full strength explodes at
    # high-valence vertices (the 16-spoke hub); dividing by the edge
    # count keeps the relaxation stable everywhere.
    acc_f = h.named('FLOAT_VECTOR', A_ACCUM, 3600, -300)
    eov_pt = h.n("GeometryNodeEdgesOfVertex", 3600, -700, label="valence")
    val_max = h.ma('MAXIMUM', 3750, -700, eov_pt.outputs["Total"], 1.0)
    inv_val = h.ma('DIVIDE', 3900, -700, 1.0, val_max.outputs["Value"],
                   label="1 / valence")
    acc_avg = h.vscale(3750, -300, acc_f.outputs["Attribute"],
                       inv_val.outputs["Value"], label="averaged corr")

    pin2 = h.named('BOOLEAN', A_PIN, 3600, -500)
    npin2 = h.bmath('NOT', 3800, -500, pin2.outputs["Attribute"])

    sp_relax = h.n("GeometryNodeSetPosition", 3950, 100, label="Relax")
    h.lk(r3_out.outputs["Geometry"], sp_relax.inputs["Geometry"])
    h.lk(npin2.outputs["Boolean"], sp_relax.inputs["Selection"])
    h.lk(acc_avg.outputs["Vector"], sp_relax.inputs["Offset"])

    # ---- collision projection ----
    pos_c = h.n("GeometryNodeInputPosition", 4100, -300)

    src_pos = h.n("GeometryNodeInputPosition", 4100, -600)
    sns_p = h.n("GeometryNodeSampleNearestSurface", 4350, -500,
                label="hit position", data_type='FLOAT_VECTOR')
    h.lk(obj_info.outputs["Geometry"], sns_p.inputs["Mesh"])
    h.lk(src_pos.outputs["Position"], sns_p.inputs["Value"])
    h.lk(pos_c.outputs["Position"], sns_p.inputs["Sample Position"])

    src_nrm = h.n("GeometryNodeInputNormal", 4100, -900)
    sns_n = h.n("GeometryNodeSampleNearestSurface", 4350, -850,
                label="hit normal", data_type='FLOAT_VECTOR')
    h.lk(obj_info.outputs["Geometry"], sns_n.inputs["Mesh"])
    h.lk(src_nrm.outputs["Normal"], sns_n.inputs["Value"])
    h.lk(pos_c.outputs["Position"], sns_n.inputs["Sample Position"])

    rel = h.vm('SUBTRACT', 4650, -500,
               pos_c.outputs["Position"], sns_p.outputs["Value"])
    sdist = h.vm('DOT_PRODUCT', 4850, -500,
                 rel.outputs["Vector"], sns_n.outputs["Value"],
                 label="signed dist")

    pen = h.cmp('FLOAT', 'LESS_THAN', 5050, -400,
                sdist.outputs["Value"], g["Collision Offset"],
                label="penetrating")
    nlen = h.vm('LENGTH', 4650, -900, sns_n.outputs["Value"])
    hasgeo = h.cmp('FLOAT', 'GREATER_THAN', 4850, -900,
                   nlen.outputs["Value"], 0.1, label="collider valid")

    and1 = h.bmath('AND', 5250, -500, pen.outputs["Result"],
                   hasgeo.outputs["Result"])
    and2 = h.bmath('AND', 5400, -500, and1.outputs["Boolean"],
                   g["Enable Collision"])
    collide = h.bmath('AND', 5550, -500, and2.outputs["Boolean"],
                      npin2.outputs["Boolean"])

    noff = h.vscale(5050, -700, sns_n.outputs["Value"],
                    g["Collision Offset"])
    target = h.vm('ADD', 5250, -700,
                  sns_p.outputs["Value"], noff.outputs["Vector"],
                  label="surface target")

    sp_col = h.n("GeometryNodeSetPosition", 5500, 100, label="Collide")
    h.lk(sp_relax.outputs["Geometry"], sp_col.inputs["Geometry"])
    h.lk(collide.outputs["Boolean"], sp_col.inputs["Selection"])
    h.lk(target.outputs["Vector"], sp_col.inputs["Position"])

    # friction: pull prev_pos toward the contact point to kill sliding
    prev_c = h.named('FLOAT_VECTOR', A_PREV, 5500, -900)
    dpf = h.vm('SUBTRACT', 5700, -900,
               target.outputs["Vector"], prev_c.outputs["Attribute"])
    dpfs = h.vscale(5900, -900, dpf.outputs["Vector"], g["Friction"])
    prev_new = h.vm('ADD', 6100, -900,
                    prev_c.outputs["Attribute"], dpfs.outputs["Vector"])

    st_fric = h.store('FLOAT_VECTOR', 'POINT', A_PREV, 5850, 100,
                      geo=sp_col.outputs["Geometry"],
                      value=prev_new.outputs["Vector"],
                      sel=collide.outputs["Boolean"])
    h.lk(st_fric.outputs["Geometry"], r2_out.inputs["Geometry"])

    h.lk(r2_out.outputs["Geometry"], r1_out.inputs["Geometry"])

    # ---------------------------------------------------------------------
    #  TEARING (once per frame, after all substeps)
    # ---------------------------------------------------------------------
    ev4 = h.n("GeometryNodeInputMeshEdgeVertices", 6800, -400)
    d4 = h.vm('DISTANCE', 7000, -400,
              ev4.outputs["Position 1"], ev4.outputs["Position 2"])
    rest4 = h.named('FLOAT', A_REST, 7000, -650)
    limit = h.ma('MULTIPLY', 7200, -550,
                 rest4.outputs["Attribute"], g["Tear Threshold"],
                 label="tear limit")
    over = h.cmp('FLOAT', 'GREATER_THAN', 7400, -450,
                 d4.outputs["Value"], limit.outputs["Value"],
                 label="overstretched")
    tear_sel = h.bmath('AND', 7600, -450, over.outputs["Result"],
                       g["Enable Tearing"])

    del_edges = h.n("GeometryNodeDeleteGeometry", 7200, 100,
                    label="TEAR", domain='EDGE', mode='EDGE_FACE')
    h.lk(r1_out.outputs["Geometry"], del_edges.inputs["Geometry"])
    h.lk(tear_sel.outputs["Boolean"], del_edges.inputs["Selection"])

    eov2 = h.n("GeometryNodeEdgesOfVertex", 7400, -800)
    iszero = h.cmp('INT', 'EQUAL', 7600, -800,
                   eov2.outputs["Total"], 0, label="orphaned")
    loose_sel = h.bmath('AND', 7800, -800, iszero.outputs["Result"],
                        g["Remove Loose Points"])

    del_pts = h.n("GeometryNodeDeleteGeometry", 7600, 100,
                  label="Remove loose", domain='POINT')
    h.lk(del_edges.outputs["Geometry"], del_pts.inputs["Geometry"])
    h.lk(loose_sel.outputs["Boolean"], del_pts.inputs["Selection"])

    h.lk(del_pts.outputs["Geometry"], sim_out.inputs["Geometry"])
    h.lk(sim_out.outputs["Geometry"], go.inputs["Geometry"])
    return nt


SOLVER_VERSION = 2


def ensure_solver_group():
    nt = bpy.data.node_groups.get(GROUP_SOLVER)
    if nt is not None:
        if nt.get("swf_version", 0) >= SOLVER_VERSION:
            return nt
        nt.name = GROUP_SOLVER + ".old"  # keep, but retire the stale build
    nt = _build_group()
    nt["swf_version"] = SOLVER_VERSION
    return nt


def apply_solver(obj, collider=None, report=None):
    """Add the solver modifier to obj, bind the pin attribute, set collider."""
    group = ensure_solver_group()
    mod = obj.modifiers.new("SWF Tearing Solver", 'NODES')
    mod.node_group = group

    pin_id = input_identifier(group, "Pin Vertices")
    if pin_id:
        try:
            mod[pin_id + "_use_attribute"] = True
            mod[pin_id + "_attribute_name"] = A_PIN
        except (KeyError, TypeError):
            if report:
                report({'WARNING'},
                       "Couldn't auto-bind pins — set 'Pin Vertices' to "
                       "attribute '%s' manually." % A_PIN)

    if collider is not None:
        coll_id = input_identifier(group, "Collision Object")
        if coll_id:
            try:
                mod[coll_id] = collider
            except (KeyError, TypeError):
                pass
    return mod


# ============================================================================
#  Operators
# ============================================================================

class SWF_OT_add_tearing_solver(Operator):
    """Add the tearing web solver to the active object.
    If another mesh is also selected, it becomes the collider."""
    bl_idname = "swf.add_tearing_solver"
    bl_label = "Add Tearing Solver"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == 'MESH'

    def execute(self, context):
        obj = context.object
        others = [o for o in context.selected_objects
                  if o is not obj and o.type == 'MESH']
        collider = others[0] if others else None
        apply_solver(obj, collider, self.report)
        if collider:
            self.report({'INFO'}, "Collider set to '%s'" % collider.name)
        return {'FINISHED'}


class SWF_OT_pin_vertices(Operator):
    """Write the current Edit Mode vertex selection into the pin attribute"""
    bl_idname = "swf.pin_vertices"
    bl_label = "Pin Selected"
    bl_options = {'REGISTER', 'UNDO'}

    action: bpy.props.EnumProperty(
        items=[('PIN', "Pin Selected", ""),
               ('UNPIN', "Unpin Selected", ""),
               ('CLEAR', "Clear All Pins", "")],
        default='PIN')

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == 'MESH'

    def execute(self, context):
        obj = context.object
        was_edit = (obj.mode == 'EDIT')
        if was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')

        me = obj.data
        attr = me.attributes.get(A_PIN)
        if (attr is None or attr.domain != 'POINT'
                or attr.data_type != 'BOOLEAN'):
            if attr is not None:
                me.attributes.remove(attr)
            attr = me.attributes.new(A_PIN, 'BOOLEAN', 'POINT')

        if self.action == 'CLEAR':
            for d in attr.data:
                d.value = False
        else:
            val = (self.action == 'PIN')
            for v in me.vertices:
                if v.select:
                    attr.data[v.index].value = val

        if was_edit:
            bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}


classes = (SWF_OT_add_tearing_solver, SWF_OT_pin_vertices)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
