# Strandify: turns the (simulated, possibly torn) edge mesh into renderable
# silk geometry — curves with noisy radius swept into tubes — plus optional
# dew droplets instanced along the strands.
#
# Add this modifier AFTER the tearing solver so torn strands render too.

import bpy
from bpy.types import Operator

from .constants import GROUP_STRANDIFY
from .nodeutils import H, sock_in, minmax_sockets, input_identifier
from .materials import (ensure_silk_material, ensure_dew_material,
                        ensure_tension_material)


def _build_group():
    nt = bpy.data.node_groups.new(GROUP_STRANDIFY, "GeometryNodeTree")
    iface = nt.interface
    h = H(nt)

    iface.new_socket(name="Geometry", in_out='INPUT',
                     socket_type='NodeSocketGeometry')
    sock_in(iface, "Strand Radius",      'NodeSocketFloat', 0.0008,
            0.00005, 0.01)
    sock_in(iface, "Radius Variation",   'NodeSocketFloat', 0.4, 0.0, 1.0)
    sock_in(iface, "Profile Resolution", 'NodeSocketInt',   6, 3, 16)
    sock_in(iface, "Silk Material",      'NodeSocketMaterial')
    sock_in(iface, "Show Tension",       'NodeSocketBool',  False)
    sock_in(iface, "Tension Material",   'NodeSocketMaterial')
    sock_in(iface, "Enable Dew",         'NodeSocketBool',  True)
    sock_in(iface, "Dew Spacing",        'NodeSocketFloat', 0.06, 0.005, 1.0)
    sock_in(iface, "Dew Amount",         'NodeSocketFloat', 0.25, 0.0, 1.0)
    sock_in(iface, "Dew Size",           'NodeSocketFloat', 0.004,
            0.0005, 0.05)
    sock_in(iface, "Dew Material",       'NodeSocketMaterial')
    sock_in(iface, "Seed",               'NodeSocketInt',   0, 0, 100000)
    iface.new_socket(name="Geometry", in_out='OUTPUT',
                     socket_type='NodeSocketGeometry')

    gi = h.n("NodeGroupInput", -1200, 0)
    go = h.n("NodeGroupOutput", 1400, 0)
    g = gi.outputs

    # ---- strands -----------------------------------------------------------
    m2c = h.n("GeometryNodeMeshToCurve", -900, 100)
    h.lk(g["Geometry"], m2c.inputs["Mesh"])

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

    setrad = h.n("GeometryNodeSetCurveRadius", -600, 100)
    h.lk(m2c.outputs["Curve"], setrad.inputs["Curve"])
    h.lk(radius.outputs["Value"], setrad.inputs["Radius"])

    circle = h.n("GeometryNodeCurvePrimitiveCircle", -300, -100,
                 label="profile")
    h.lk(g["Profile Resolution"], circle.inputs["Resolution"])
    circle.inputs["Radius"].default_value = 1.0

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

    # ---- dew ---------------------------------------------------------------
    c2p = h.n("GeometryNodeCurveToPoints", -300, -700, mode='LENGTH')
    h.lk(setrad.outputs["Curve"], c2p.inputs["Curve"])
    h.lk(g["Dew Spacing"], c2p.inputs["Length"])

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

    ico = h.n("GeometryNodeMeshIcoSphere", 0, -1200, label="droplet")
    ico.inputs["Radius"].default_value = 1.0
    ico.inputs["Subdivisions"].default_value = 2

    rs = h.n("FunctionNodeRandomValue", 0, -1450, data_type='FLOAT')
    smn, smx = minmax_sockets(rs)
    smn.default_value = 0.6
    smx.default_value = 1.4
    seed2 = h.ma('ADD', -200, -1450, g["Seed"], 1.0)
    h.lk(seed2.outputs["Value"], rs.inputs["Seed"])
    dsize = h.ma('MULTIPLY', 200, -1400,
                 rs.outputs["Value"], g["Dew Size"])

    iop = h.n("GeometryNodeInstanceOnPoints", 300, -800)
    h.lk(delete.outputs["Geometry"], iop.inputs["Points"])
    h.lk(ico.outputs["Mesh"], iop.inputs["Instance"])
    h.lk(dsize.outputs["Value"], iop.inputs["Scale"])

    real = h.n("GeometryNodeRealizeInstances", 550, -800)
    h.lk(iop.outputs["Instances"], real.inputs["Geometry"])

    dewmat = h.n("GeometryNodeSetMaterial", 750, -800)
    h.lk(real.outputs["Geometry"], dewmat.inputs["Geometry"])
    h.lk(g["Dew Material"], dewmat.inputs["Material"])

    dew_switch = h.n("GeometryNodeSwitch", 950, -500,
                     input_type='GEOMETRY', label="dew on/off")
    h.lk(g["Enable Dew"], dew_switch.inputs["Switch"])
    h.lk(dewmat.outputs["Geometry"], dew_switch.inputs["True"])

    join = h.n("GeometryNodeJoinGeometry", 1150, 0)
    h.lk(dew_switch.outputs["Output"], join.inputs["Geometry"])
    h.lk(silk.outputs["Geometry"], join.inputs["Geometry"])
    h.lk(join.outputs["Geometry"], go.inputs["Geometry"])
    return nt


STRANDIFY_VERSION = 2


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


classes = (SWF_OT_add_strandify,)


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
