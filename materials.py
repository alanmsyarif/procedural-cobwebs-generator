# Materials: silk (Fresnel-driven transparent/glossy mix added to a dusty
# translucent Principled base — grazing-angle visibility like real cobweb),
# dew (simple water glass), tension (solver heatmap) and synth-web (the
# synthetic shear-thinning polymer fibre — see ensure_synth_material).

import bpy

from .constants import (MAT_SILK, MAT_DEW, MAT_TENSION, MAT_SYNTH,
                        A_TENSION, A_SHOT)


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


SYNTH_VERSION = 2


def ensure_synth_material():
    """Synthetic web-fluid, shaded to the published chemistry.

    The fluid is shear-thinning: virtually solid at rest in the cartridge,
    turned fluid only by the shearing force of being fired — which is why
    the shooter never clogs. On contact with air the long-chain polymer
    knits into a tough, flexible, nylon-related fibre.

    Two properties are time-dependent, and both are driven here by how long
    a thread has been in the air (`swf_shot_t` against the current frame,
    the same per-point fire time the shot reveal uses):

      * adhesion falls off rapidly once exposed. Fresh fibre still reads wet
        and tacky — glossy, a thicker cured skin, more light through it. As
        it sets it goes matte and opaque.
      * imbibed esters break the solid down into powder after an hour or
        two. That is far past any shot animation, so it lands here as a
        chalk bloom that only starts once the fibre has set, strongest at
        grazing angles where a dusting would catch the light.

    "Cure Frames" sets how long that takes on screen — 36 frames by
    default, which is a second and a half at 24fps. Comic-accurate would be
    hours; this is the readable version.

    Shader nodes have no scene-time input, so the current frame arrives
    through a driver on the "Arachne Frame" value node.
    """
    mat = bpy.data.materials.get(MAT_SYNTH)
    if mat is not None and mat.get("swf_version", 0) >= SYNTH_VERSION:
        return mat
    # Rebuilt in place rather than renamed aside: every object and modifier
    # socket already pointing at this datablock then picks up the new look,
    # where a ".old" rename would leave existing scenes on the previous
    # build. Hand edits to the node tree are lost on a version bump.
    if mat is None:
        mat = bpy.data.materials.new(MAT_SYNTH)
    mat.use_nodes = True
    nt = mat.node_tree
    if nt.animation_data is not None:
        nt.animation_data_clear()      # drop the previous build's driver
    nt.nodes.clear()
    n, lk = nt.nodes.new, nt.links.new

    def mixf(x, y, a, b, fac, label=""):
        """Float mix, accepting sockets or constants for A and B."""
        m = n("ShaderNodeMix"); m.location = (x, y); m.data_type = 'FLOAT'
        if label:
            m.label = label
        f, sa, sb, res = _mix_io(m)
        for sock, val in ((sa, a), (sb, b)):
            if isinstance(val, bpy.types.NodeSocket):
                lk(val, sock)
            else:
                sock.default_value = val
        lk(fac, f)
        return res

    out = n("ShaderNodeOutputMaterial"); out.location = (900, 0)
    pr = n("ShaderNodeBsdfPrincipled"); pr.location = (550, 0)

    # ---- time in air ------------------------------------------------------
    now = n("ShaderNodeValue"); now.location = (-1500, 400)
    now.name = "Arachne Frame"; now.label = "current frame"
    try:
        fcu = nt.driver_add('nodes["Arachne Frame"].outputs[0].default_value')
        # driver_add seeds a generator modifier that would ignore the
        # expression entirely
        for mod in list(fcu.modifiers):
            fcu.modifiers.remove(mod)
        fcu.driver.type = 'SCRIPTED'
        fcu.driver.expression = "frame"
    except Exception:
        # no driver: the whole web reads as fully cured, which is the
        # sensible fallback rather than a broken material
        now.outputs[0].default_value = 1e6

    shot = n("ShaderNodeAttribute"); shot.location = (-1500, 200)
    shot.attribute_name = A_SHOT
    shot.attribute_type = 'GEOMETRY'

    age = n("ShaderNodeMath"); age.location = (-1250, 300)
    age.operation = 'SUBTRACT'; age.label = "frames in air"
    lk(now.outputs[0], age.inputs[0])
    lk(shot.outputs["Fac"], age.inputs[1])

    cure = n("ShaderNodeValue"); cure.location = (-1250, 100)
    cure.name = "Cure Frames"; cure.label = "cure frames"
    cure.outputs[0].default_value = 36.0

    setf = n("ShaderNodeMapRange"); setf.location = (-1050, 250)
    setf.label = "0 = just fired, 1 = set"
    setf.inputs["From Min"].default_value = 0.0
    lk(cure.outputs[0], setf.inputs["From Max"])
    lk(age.outputs["Value"], setf.inputs["Value"])
    dry = setf.outputs["Result"]

    # ---- powder: only once set, and only at grazing angles ----------------
    lw = n("ShaderNodeLayerWeight"); lw.location = (-1050, 600)
    lw.inputs["Blend"].default_value = 0.35
    powder = n("ShaderNodeMath"); powder.location = (-800, 600)
    powder.operation = 'MULTIPLY'; powder.label = "ester bloom"
    lk(lw.outputs["Facing"], powder.inputs[0])
    lk(dry, powder.inputs[1])

    col = n("ShaderNodeMix"); col.location = (-350, 550)
    col.data_type = 'RGBA'
    cfac, ca, cb, cres = _mix_io(col)
    ca.default_value = (0.855, 0.885, 0.945, 1.0)   # knitted polymer
    cb.default_value = (0.970, 0.965, 0.945, 1.0)   # ester powder
    lk(powder.outputs["Value"], cfac)
    lk(cres, pr.inputs["Base Color"])

    # ---- roughness: wet and glossy -> set and matte -> powdered -----------
    rnoise = n("ShaderNodeTexNoise"); rnoise.location = (-1050, -100)
    rnoise.inputs["Scale"].default_value = 260.0
    rmap = n("ShaderNodeMapRange"); rmap.location = (-800, -100)
    rmap.inputs["To Min"].default_value = 0.14
    rmap.inputs["To Max"].default_value = 0.30
    lk(rnoise.outputs["Fac"], rmap.inputs["Value"])
    # tacky fibre is still wet, so it starts far glossier than it ends
    set_rough = mixf(-500, -100, 0.05, rmap.outputs["Result"], dry,
                     label="wet -> set")
    lk(mixf(-200, -100, set_rough, 0.62, powder.outputs["Value"],
            label="powdered"), pr.inputs["Roughness"])

    # ---- adhesion fading with exposure ------------------------------------
    # a tacky thread carries a wet skin and passes more light; a set one is
    # a dry opaque fibre
    coat = mixf(-500, 250, 0.85, 0.30, dry, label="wet skin")
    trans = mixf(-500, 100, 0.32, 0.08, dry, label="clouding over")
    sheen = mixf(-500, -300, 0.10, 0.35, dry, label="dry fuzz")

    # ---- micro-texture of the knitted filament bundle ---------------------
    bnoise = n("ShaderNodeTexNoise"); bnoise.location = (-1050, -500)
    bnoise.inputs["Scale"].default_value = 620.0
    bnoise.inputs["Detail"].default_value = 8.0
    bump = n("ShaderNodeBump"); bump.location = (-500, -500)
    bump.inputs["Strength"].default_value = 0.05
    lk(bnoise.outputs["Fac"], bump.inputs["Height"])
    lk(bump.outputs["Normal"], pr.inputs["Normal"])

    _set_input(pr, 1.53, "IOR")                       # nylon-related
    _set_input(pr, 0.15, "Subsurface Weight", "Subsurface")
    _set_input(pr, (0.004, 0.004, 0.006), "Subsurface Radius")
    _set_input(pr, 0.05, "Coat Roughness", "Clearcoat Roughness")
    _set_input(pr, 1.60, "Coat IOR")
    _set_input(pr, 0.30, "Sheen Roughness")
    _set_input(pr, 0.60, "Specular IOR Level", "Specular")
    for value, names in ((coat, ("Coat Weight", "Clearcoat")),
                         (trans, ("Transmission Weight", "Transmission")),
                         (sheen, ("Sheen Weight", "Sheen"))):
        sock = _first_input(pr, *names)
        if sock is not None:
            lk(value, sock)

    lk(pr.outputs["BSDF"], out.inputs["Surface"])
    mat["swf_version"] = SYNTH_VERSION
    return mat
