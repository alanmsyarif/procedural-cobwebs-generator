# N-panel UI.

import os

import bpy
from bpy.types import Operator, Panel

from .constants import A_EMIT, A_SHOT, BUILD, P_EMITTER
from .generator import live_ready
from .gpu_solver import backend_reason, gpu_backend_available

_DISK = {"mtime": None, "build": BUILD}


def _wrap(text, width):
    """Break a message into panel-width lines. Blender clips a label at the
    region edge rather than wrapping it, so a raw exception string would
    lose its tail — which is the half that says what went wrong."""
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        lines.append(line)
    return lines or [""]


def disk_build():
    """The BUILD string sitting in constants.py on disk, which is not what
    is running if the files were replaced without reloading scripts —
    Python holds on to the modules it imported at startup. Comparing the
    two is the only way to catch that from inside Blender."""
    path = os.path.join(os.path.dirname(__file__), "constants.py")
    try:
        mtime = os.path.getmtime(path)
        if mtime != _DISK["mtime"]:
            _DISK["mtime"] = mtime
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("BUILD"):
                        _DISK["build"] = line.split("=", 1)[1].strip().strip(
                            '"').strip("'")
                        break
    except Exception:
        pass
    return _DISK["build"]


class ARN_OT_diagnose(Operator):
    """Write a report on the active web — which build is loaded, what the
    solver is bound to, where each burst's muzzle sits — into a text
    datablock (open it in the Scripting workspace)"""
    bl_idname = "arachne.diagnose"
    bl_label = "Diagnose Web"

    def execute(self, context):
        import sys
        import numpy as np
        from . import gpu_native, gpu_solver

        L = []
        obj = context.object
        L.append("Arachne build %s" % BUILD)
        L.append("module: %s" % sys.modules[__name__].__file__)
        L.append("scene frame: %d" % context.scene.frame_current)
        if obj is None or obj.type != 'MESH' or not obj.get("swf_web"):
            L.append("!! active object is not an Arachne web — select the "
                     "web object and run again")
        else:
            me = obj.data
            g = getattr(obj, "swf_gpu", None)
            L.append("web: %s  verts=%d  modifiers=%s"
                     % (obj.name, len(me.vertices),
                        [m.name for m in obj.modifiers]))
            L.append("solver enabled: %s   stick_follow: %s   collider: %s"
                     % (getattr(g, "enabled", None),
                        getattr(g, "stick_follow", None),
                        getattr(getattr(g, "collider", None), "name", None)))
            L.append("mesh binding: %s = %r   %s attribute: %s"
                     % (P_EMITTER, me.get(P_EMITTER), A_EMIT,
                        A_EMIT in me.attributes))
            st = gpu_solver._STATES.get(obj.name)
            if st is None:
                L.append("solver state: NONE (never stepped — play forward "
                         "from the start frame)")
            else:
                L.append("solver state: emitter=%s  collider hosts=%s  "
                         "follow=%s  latch=%s  stepped_to=%s  n=%d"
                         % (getattr(st.emitter, "name", None),
                            [o.name for o in st.stick_objs],
                            st.follow, st.latch, st.last_frame, st.n))
                try:
                    w = gpu_native._read(st.posA, st.n, 4)[:, 3]
                    L.append("pin states: free=%d pinned=%d %s"
                             % (int((w < 0.5).sum()),
                                int(np.isclose(w, 1.0).sum()),
                                "  ".join(
                                    "ride[%s]=%d" % (o.name,
                                                     int(np.isclose(w, 2.0 + k)
                                                         .sum()))
                                    for k, o in enumerate(st.stick_objs))))
                except Exception as ex:
                    L.append("pin states unreadable: %s" % ex)
            emit = bpy.data.objects.get(me.get(P_EMITTER) or "")
            if A_EMIT in me.attributes and emit is not None:
                n = len(me.vertices)
                mz = np.zeros(n, bool)
                me.attributes[A_EMIT].data.foreach_get("value", mz)
                sh = np.zeros(n, np.float32)
                me.attributes[A_SHOT].data.foreach_get("value", sh)
                pos = np.empty(n * 3, np.float32)
                src = ("swf_gpu_pos" if "swf_gpu_pos" in me.attributes
                       else None)
                if src:
                    me.attributes[src].data.foreach_get("vector", pos)
                else:
                    me.vertices.foreach_get("co", pos)
                pos = pos.reshape(-1, 3)
                E = np.array(obj.matrix_world.inverted_safe()
                             @ emit.matrix_world.translation)
                L.append("muzzles: %d, distance to %s right now:"
                         % (int(mz.sum()), emit.name))
                for f in sorted(set(np.round(sh[mz]))):
                    i = int(np.flatnonzero(mz & (np.round(sh) == f))[0])
                    L.append("   fired frame %7.1f -> %.3f m"
                             % (f, float(np.linalg.norm(pos[i] - E))))

        text = "\n".join(L)
        print(text)
        name = "Arachne Diagnostic"
        txt = bpy.data.texts.get(name) or bpy.data.texts.new(name)
        txt.clear()
        txt.write(text)
        self.report({'INFO'},
                    "Written to Text block '%s' (Scripting workspace)" % name)
        return {'FINISHED'}


def _draw_shot_status(col, context, p):
    """Everything that decides whether a Web Shot sticks to its emitter,
    stated on the panel. Each of these has been the answer at some point,
    and none of them is visible anywhere else."""
    from . import gpu_solver

    box = col.box()
    box.scale_y = 0.8
    disk = disk_build()
    if disk != BUILD:
        box.label(text="RELOAD SCRIPTS: running %s, on disk %s"
                       % (BUILD, disk), icon='ERROR')

    ob = context.object
    if (ob is None or ob.type != 'MESH' or not ob.get("swf_web")
            or ob.data.attributes.get(A_SHOT) is None):
        box.label(text="Select the web to see its status.", icon='INFO')
        return

    me = ob.data
    bound = me.get(P_EMITTER)
    if p.shot_emitter is None:
        box.label(text="No emitter set.", icon='INFO')
    elif bound is None or A_EMIT not in me.attributes:
        box.label(text="Not bound — press Generate Web", icon='ERROR')
    elif bound not in bpy.data.objects:
        box.label(text="Bound to '%s', which no longer exists" % bound,
                  icon='ERROR')
    else:
        box.label(text="Stuck to: %s" % bound, icon='CHECKMARK')

    g = getattr(ob, "swf_gpu", None)
    if g is None or not g.enabled:
        box.label(text="GPU solver off — Stick To Emitter needs it",
                  icon='ERROR')
    else:
        st = gpu_solver._STATES.get(ob.name)
        if st is None:
            box.label(text="Solver idle — play forward a frame", icon='INFO')
        else:
            box.label(text="Solver: frame %s, %s"
                           % (st.last_frame,
                              "anchored" if st.emitter is not None
                              else "no anchors bound"),
                      icon='PLAY' if st.emitter is not None else 'ERROR')


class ARN_PT_main(Panel):
    bl_label = "Arachne"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Arachne"

    def draw(self, context):
        layout = self.layout
        p = context.scene.swf_web

        box = layout.box()
        box.label(text="Generate", icon='OUTLINER_OB_CURVES')
        col = box.column(align=True)
        col.prop(p, "mode", text="")
        col.separator()
        if p.mode == 'CHAOS':
            col.prop(p, "cobweb_initial")
            col.prop(p, "cobweb_spiders")
            col.prop(p, "cobweb_steps")
            col.prop(p, "cobweb_jump")
            col.prop(p, "cobweb_spread")
            col.prop(p, "cobweb_clump")
            col.prop(p, "cobweb_bridge")
            col.prop(p, "radius", text="Anchor Span")
            col.prop(p, "cobweb_sag")
            col.prop(p, "jitter")
            col.prop(p, "cobweb_detail")
            col.prop(p, "seed")
            col.label(text="Select anchor geometry first.", icon='INFO')
        elif p.mode == 'SHOT':
            col.prop(p, "shot_emitter")
            aimc = p.shot_aim_collection
            row = col.row()
            row.enabled = aimc is None       # the collection overrides it
            row.prop(p, "shot_aim")
            col.prop(p, "shot_aim_collection")
            if aimc is not None:
                n = len([o for o in aimc.objects
                         if o is not p.shot_emitter and not o.get("swf_web")])
                if n == 0:
                    col.label(text="Aim Collection is empty.", icon='ERROR')
                elif n > p.shot_bursts:
                    # unreached members are ignored entirely, hit test
                    # included — say so rather than letting it look broken
                    col.label(text="%d targets, Bursts %d — only the first"
                                   % (n, p.shot_bursts), icon='ERROR')
                    col.label(text="%d used, rest ignored."
                                   % p.shot_bursts, icon='BLANK1')
                else:
                    col.label(text="%d targets, one per burst in turn." % n,
                              icon='INFO')
            row = col.row()
            row.enabled = p.shot_emitter is not None
            row.prop(p, "shot_stick_emitter")
            col.separator()
            col.prop(p, "shot_count")
            col.prop(p, "shot_bursts")
            row = col.row()
            row.enabled = p.shot_bursts > 1
            row.prop(p, "shot_burst_gap")
            col.prop(p, "shot_start")
            col.prop(p, "shot_interval")
            col.prop(p, "shot_speed")
            # an aim target sets the reach itself, far enough to arrive and
            # no farther, so Range has nothing to say
            row = col.row()
            row.enabled = p.shot_aim is None and aimc is None
            row.prop(p, "shot_range")
            col.prop(p, "shot_spread")
            col.separator()
            col.prop(p, "shot_clot")
            col.prop(p, "shot_clot_size")
            col.prop(p, "shot_clot_twist")
            row = col.row()
            row.enabled = p.shot_clot > 0.0
            row.prop(p, "shot_clot_smooth")
            col.separator()
            col.prop(p, "shot_tangle")
            col.prop(p, "jitter")
            col.prop(p, "detail")
            col.prop(p, "seed")

            col.separator()
            adv = col.box()
            adv.prop(p, "shot_advanced", emboss=False,
                     icon='TRIA_DOWN' if p.shot_advanced else 'TRIA_RIGHT')
            if p.shot_advanced:
                sub = adv.column(align=True)
                sub.prop(p, "shot_arc")
                sub.prop(p, "shot_whip")
                sub.prop(p, "shot_slack")
                sub.separator()
                sub.prop(p, "shot_splat")
                spl = sub.column(align=True)
                spl.enabled = p.shot_splat > 0
                spl.prop(p, "shot_splat_size")
                spl.prop(p, "shot_splat_web")

            col.label(text="Select what the shots hit.", icon='INFO')
            col.label(text="Shots fly once the solver or strands are on.",
                      icon='INFO')
            _draw_shot_status(col, context, p)
        else:
            col.prop(p, "radials")
            col.prop(p, "rings")
            col.prop(p, "radius")
            col.prop(p, "hub_factor")
            col.prop(p, "jitter")
            col.prop(p, "spiral_sag")
            col.prop(p, "damage")
            col.prop(p, "asymmetry")
            col.prop(p, "tangles")
            col.prop(p, "detail")
            col.separator()
            col.prop(p, "anchors")
            col.prop(p, "anchor_extend")
            col.prop(p, "seed")
            col.prop(p, "plane")
        col.separator()
        col.operator("arachne.generate_web", icon='ADD')
        row = col.row(align=True)
        row.prop(p, "live", toggle=True, icon='MOD_TIME')
        if p.live:
            if p.live_obj is None:
                col.label(text="Generate a web to tweak it live.",
                          icon='INFO')
            elif live_ready(p):
                row.label(text="", icon='CHECKMARK')
            else:
                # rebuilding would swap the mesh out from under the running
                # sim, so live update stands down. Point at the way back in
                # rather than just saying no: hiding the apply modifier in
                # the viewport keeps every solver setting intact.
                row.label(text="", icon='LOCKED')
                col.label(text="Solver on. Hide the GPU Apply modifier",
                          icon='INFO')
                col.label(text="in the viewport to tweak live.",
                          icon='BLANK1')

        box = layout.box()
        box.label(text="GPU Solver", icon='MEMORY')
        col = box.column(align=True)
        if not gpu_backend_available():
            # say which of the two it is, wrapped — an exception text is
            # longer than one panel row
            warn = col.box()
            warn.scale_y = 0.7
            for i, line in enumerate(_wrap(backend_reason(), 38)):
                warn.label(text=line, icon='ERROR' if i == 0 else 'BLANK1')
            col.operator("arachne.reset_gpu", text="Clear Error",
                         icon='FILE_REFRESH')
        col.operator("arachne.add_gpu_solver", icon='PLAY')
        obj = context.object
        if obj and obj.type == 'MESH' and obj.swf_gpu.enabled:
            g = obj.swf_gpu
            col.separator()
            col.prop(g, "tension")
            col.prop(g, "resist_compression")
            col.prop(g, "stiffness")
            col.prop(g, "damping")
            col.prop(g, "iterations")
            col.prop(g, "substeps")
            col.prop(g, "pre_warm")
            col.prop(g, "deteriorate")
            col.prop(g, "seed")
            col.separator()
            col.prop(g, "gravity")
            col.prop(g, "wind")
            col.prop(g, "turbulence")
            col.separator()
            col.prop(g, "enable_collision")
            col.prop(g, "collision_shape", text="")
            if g.collision_shape == 'MESH_SDF' or g.collider_collection:
                col.prop(g, "sdf_resolution")
            col.prop(g, "collider")
            # A bounding sphere around a box reaches sqrt(3)/2 of the side
            # from its centre while the faces sit at half the side, so the
            # collision surface bulges well past anything the web is stuck
            # to and shoves those anchors off it. Harmless for a roughly
            # round collider, fatal for the flat thing a shot just hit.
            if (g.collider is not None and g.collision_shape == 'SPHERE'
                    and not g.collider_collection):
                warn = col.box()
                warn.scale_y = 0.7
                warn.label(text="Bounding sphere reaches past flat faces",
                           icon='ERROR')
                warn.label(text="Shot at this? Use Mesh (SDF) or clear it.")
            col.prop(g, "collider_collection")
            col.prop(g, "collision_offset")
            col.prop(g, "friction")
            col.prop(g, "stickiness")
            col.prop(g, "stick_follow")
            col.separator()
            col.prop(g, "pull_collider")
            if g.pull_collider:
                single = g.collider is not None and not g.collider_collection
                rb = getattr(g.collider, "rigid_body", None) if single else None
                sub = col.column(align=True)
                sub.enabled = single
                sub.prop(g, "collider_static")
                sub.prop(g, "pull_strength")
                # mass and friction belong to Bullet — surfaced here so the
                # dials that matter are in one place, not two tabs apart
                if rb is not None:
                    mov = sub.column(align=True)
                    mov.enabled = not g.collider_static
                    mov.prop(rb, "mass")
                    mov.prop(rb, "friction")
                    mov.prop(rb, "linear_damping", text="Drag")
                col.operator("arachne.setup_pull", icon='PHYSICS')
                # the web can only pull through anchors welded to the
                # collider, and those exist only while it rides it
                if not single:
                    col.label(text="Needs a single Collider object.",
                              icon='ERROR')
                elif rb is None:
                    col.label(text="Run Set Up Rigid Body Pull.",
                              icon='ERROR')
                elif not g.stick_follow:
                    col.label(text="Turn on Stuck Follows Collider.",
                              icon='ERROR')
            col.separator()
            col.prop(g, "enable_tearing")
            col.prop(g, "tear_threshold")
            if g.enable_tearing:
                # Tearing measures stretch against rest length, and Tension
                # above 1 shrinks rest — so a thread starts out already
                # part-way to the threshold and the number here buys less
                # than it reads. Spell out what is actually left.
                from .gpu_native import rest_slack
                slack = rest_slack(g.tension)
                start = 1.0 / max(slack, 1e-6)
                if start > 1.0:
                    margin = g.tear_threshold / start
                    warn = col.box()
                    warn.scale_y = 0.7
                    warn.label(
                        text="Pre-tension starts threads at %.2f stretch"
                             % start,
                        icon='INFO' if margin > 1.5 else 'ERROR')
                    warn.label(text="Tears at %.2fx built, not %.2fx"
                                    % (g.tear_threshold * slack,
                                       g.tear_threshold))
            col.separator()
            row = col.row(align=True)
            row.operator("arachne.reset_gpu", icon='FILE_REFRESH')
            row.operator("arachne.remove_gpu_solver", text="Remove", icon='X')
        col.separator()
        col.label(text="Anchors (Edit Mode):")
        row = col.row(align=True)
        row.operator("arachne.pin_vertices", text="Pin").action = 'PIN'
        row.operator("arachne.pin_vertices", text="Unpin").action = 'UNPIN'
        col.operator("arachne.pin_vertices",
                     text="Clear All Pins").action = 'CLEAR'

        box = layout.box()
        box.label(text="Render", icon='CURVES')
        col = box.column(align=True)
        col.operator("arachne.add_strandify")
        col.separator()
        row = col.row(align=True)
        row.operator("arachne.bake_dew", icon='RENDER_ANIMATION')
        row.operator("arachne.free_dew_bake", text="Free", icon='X')
        col.label(text="Bake before rendering animations.", icon='INFO')

        col = layout.column(align=True)
        col.label(text="Play from frame 1 to simulate.", icon='INFO')
        row = col.row(align=True)
        row.label(text="Build %s" % BUILD)
        row.operator("arachne.diagnose", text="", icon='CONSOLE')


classes = (ARN_OT_diagnose, ARN_PT_main)


def _safe_register(cls):
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
