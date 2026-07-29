# Materials: silk (Fresnel-driven transparent/glossy mix added to a dusty
# translucent Principled base — grazing-angle visibility like real cobweb),
# dew (simple water glass), tension (solver heatmap) and synth-web (the
# synthetic shear-thinning polymer fibre — see ensure_synth_material).

import bpy

from .constants import MAT_SILK, MAT_DEW, MAT_TENSION, MAT_SYNTH, A_TENSION


def _first_input(node, *names):
    """First socket present out of `names` — Principled renamed half its
    inputs in 4.x (Transmission -> Transmission Weight, etc.), so every
    optional slot is looked up by trying both spellings."""
    for name in names:
        if name in node.inputs:
            return node.inputs[name]
    return None


def _mix_io(node):
    """(Factor, A, B, Result) of a Mix node for its current data_type.

    Mix carries one A/B/Result triple per data type and disables the ones
    it is not using, so a lookup by name would always land on the Float
    variant. Same enabled-socket trick nodeutils uses for Compare."""
    ins = [s for s in node.inputs if s.enabled]
    fac = next(s for s in ins if s.name == "Factor")
    a, b = [s for s in ins if s.name in ("A", "B")][:2]
    res = next(s for s in node.outputs if s.enabled and s.name == "Result")
    return fac, a, b, res


def _set_input(node, value, *names):
    sock = _first_input(node, *names)
    if sock is None:
        return None
    try:
        sock.default_value = value
    except (TypeError, ValueError):
        # e.g. Sheen Tint went float -> color between versions
        return None
    return sock


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


def ensure_tension_material():
    """Emissive heatmap driven by the solver's swf_tension attribute:
    deep blue at rest -> green -> yellow -> red just before tearing."""
    mat = bpy.data.materials.get(MAT_TENSION)
    if mat:
        return mat
    mat = bpy.data.materials.new(MAT_TENSION)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    n, lk = nt.nodes.new, nt.links.new

    out = n("ShaderNodeOutputMaterial"); out.location = (700, 0)

    attr = n("ShaderNodeAttribute"); attr.location = (-400, 0)
    attr.attribute_name = A_TENSION
    attr.attribute_type = 'GEOMETRY'

    ramp = n("ShaderNodeValToRGB"); ramp.location = (-150, 0)
    cr = ramp.color_ramp
    cr.elements[0].position = 0.0
    cr.elements[0].color = (0.02, 0.08, 0.45, 1.0)   # rest: deep blue
    e = cr.elements.new(0.45); e.color = (0.05, 0.75, 0.25, 1.0)  # green
    e = cr.elements.new(0.72); e.color = (0.95, 0.85, 0.05, 1.0)  # yellow
    cr.elements[-1].position = 1.0
    cr.elements[-1].color = (1.0, 0.03, 0.01, 1.0)   # about to tear: red

    emit = n("ShaderNodeEmission"); emit.location = (350, 0)
    lk(attr.outputs["Fac"], ramp.inputs["Fac"])
    lk(ramp.outputs["Color"], emit.inputs["Color"])

    # hotter strands glow brighter: strength = 1 + tension * 4
    boost = n("ShaderNodeMath"); boost.location = (100, -200)
    boost.operation = 'MULTIPLY_ADD'
    boost.inputs[1].default_value = 4.0
    boost.inputs[2].default_value = 1.0
    lk(attr.outputs["Fac"], boost.inputs[0])
    lk(boost.outputs["Value"], emit.inputs["Strength"])

    lk(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def ensure_synth_material():
    """Synthetic web-fluid: a shear-thinning polymer that stays liquid under
    pressure in the cartridge and air-cures the instant it clears the nozzle.

    Shaded as a nylon-like monofilament rather than natural silk — an
    IOR 1.53 cured skin (coat) over a milky, faintly translucent core,
    smooth where silk is dusty. Two details sell the chemistry:

      * high-frequency noise in the bump — the fibre knits from a bundle of
        extruded filaments, so the surface is textured, not glassy
      * a chalk bloom at grazing angles (Layer Weight -> Facing) driving both
        base colour and roughness — the ester breakdown that powders the web
        and dissolves it an hour or two after it is fired
    """
    mat = bpy.data.materials.get(MAT_SYNTH)
    if mat:
        return mat
    mat = bpy.data.materials.new(MAT_SYNTH)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    n, lk = nt.nodes.new, nt.links.new

    out = n("ShaderNodeOutputMaterial"); out.location = (700, 0)
    pr = n("ShaderNodeBsdfPrincipled"); pr.location = (350, 0)

    # powder bloom: 0 head-on, 1 at grazing angles
    lw = n("ShaderNodeLayerWeight"); lw.location = (-700, 250)
    lw.inputs["Blend"].default_value = 0.35
    powder = lw.outputs["Facing"]

    col = n("ShaderNodeMix"); col.location = (-350, 250)
    col.data_type = 'RGBA'
    cfac, ca, cb, cres = _mix_io(col)
    ca.default_value = (0.855, 0.885, 0.945, 1.0)   # cured polymer core
    cb.default_value = (0.970, 0.965, 0.945, 1.0)   # ester powder bloom
    lk(powder, cfac)
    lk(cres, pr.inputs["Base Color"])

    # roughness: smooth extruded polymer, dulled where it has powdered over
    rnoise = n("ShaderNodeTexNoise"); rnoise.location = (-900, -50)
    rnoise.inputs["Scale"].default_value = 260.0
    rmap = n("ShaderNodeMapRange"); rmap.location = (-700, -50)
    rmap.inputs["To Min"].default_value = 0.10
    rmap.inputs["To Max"].default_value = 0.24
    lk(rnoise.outputs["Fac"], rmap.inputs["Value"])
    rmix = n("ShaderNodeMix"); rmix.location = (-350, -50)
    rmix.data_type = 'FLOAT'
    rfac, ra, rb, rres = _mix_io(rmix)
    rb.default_value = 0.55
    lk(powder, rfac)
    lk(rmap.outputs["Result"], ra)
    lk(rres, pr.inputs["Roughness"])

    # micro-texture of the knitted filament bundle
    bnoise = n("ShaderNodeTexNoise"); bnoise.location = (-900, -400)
    bnoise.inputs["Scale"].default_value = 620.0
    bnoise.inputs["Detail"].default_value = 8.0
    bump = n("ShaderNodeBump"); bump.location = (-350, -400)
    bump.inputs["Strength"].default_value = 0.05
    lk(bnoise.outputs["Fac"], bump.inputs["Height"])
    lk(bump.outputs["Normal"], pr.inputs["Normal"])

    _set_input(pr, 1.53, "IOR")                       # nylon
    _set_input(pr, 0.18, "Transmission Weight", "Transmission")
    _set_input(pr, 0.15, "Subsurface Weight", "Subsurface")
    _set_input(pr, (0.004, 0.004, 0.006), "Subsurface Radius")
    _set_input(pr, 0.45, "Coat Weight", "Clearcoat")  # air-cured skin
    _set_input(pr, 0.05, "Coat Roughness", "Clearcoat Roughness")
    _set_input(pr, 1.60, "Coat IOR")
    _set_input(pr, 0.25, "Sheen Weight", "Sheen")
    _set_input(pr, 0.30, "Sheen Roughness")
    _set_input(pr, 0.60, "Specular IOR Level", "Specular")

    lk(pr.outputs["BSDF"], out.inputs["Surface"])
    return mat
