# Materials: silk (Fresnel-driven transparent/glossy mix added to a dusty
# translucent Principled base — grazing-angle visibility like real cobweb)
# and dew (simple water glass).

import bpy

from .constants import MAT_SILK, MAT_DEW


def ensure_silk_material():
    mat = bpy.data.materials.get(MAT_SILK)
    if mat:
        return mat
    mat = bpy.data.materials.new(MAT_SILK)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    n, lk = nt.nodes.new, nt.links.new

    out = n("ShaderNodeOutputMaterial"); out.location = (800, 0)
    add = n("ShaderNodeAddShader"); add.location = (600, 0)

    # branch 1: grazing-angle sheen
    fresnel = n("ShaderNodeFresnel"); fresnel.location = (0, 300)
    fresnel.inputs["IOR"].default_value = 1.15
    transp = n("ShaderNodeBsdfTransparent"); transp.location = (0, 150)
    glossy = n("ShaderNodeBsdfGlossy"); glossy.location = (0, 0)
    glossy.inputs["Roughness"].default_value = 0.25
    mix = n("ShaderNodeMixShader"); mix.location = (300, 150)
    lk(fresnel.outputs["Fac"], mix.inputs["Fac"])
    lk(transp.outputs["BSDF"], mix.inputs[1])
    lk(glossy.outputs["BSDF"], mix.inputs[2])
    lk(mix.outputs["Shader"], add.inputs[0])

    # branch 2: dusty translucent silk body
    pr = n("ShaderNodeBsdfPrincipled"); pr.location = (0, -350)
    pr.inputs["Base Color"].default_value = (0.909, 0.894, 0.862, 1.0)
    pr.inputs["Roughness"].default_value = 0.6
    for name in ("Transmission Weight", "Transmission"):
        if name in pr.inputs:
            pr.inputs[name].default_value = 0.3
            break

    noise = n("ShaderNodeTexNoise"); noise.location = (-500, -550)
    noise.inputs["Scale"].default_value = 180.0
    bump = n("ShaderNodeBump"); bump.location = (-250, -550)
    bump.inputs["Strength"].default_value = 0.08
    lk(noise.outputs["Fac"], bump.inputs["Height"])
    lk(bump.outputs["Normal"], pr.inputs["Normal"])
    lk(pr.outputs["BSDF"], add.inputs[1])

    lk(add.outputs["Shader"], out.inputs["Surface"])
    return mat


def ensure_dew_material():
    mat = bpy.data.materials.get(MAT_DEW)
    if mat:
        return mat
    mat = bpy.data.materials.new(MAT_DEW)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (300, 0)
    glass = nt.nodes.new("ShaderNodeBsdfGlass"); glass.location = (0, 0)
    glass.inputs["IOR"].default_value = 1.33
    glass.inputs["Roughness"].default_value = 0.02
    nt.links.new(glass.outputs["BSDF"], out.inputs["Surface"])
    return mat
