# N-panel UI and the one-click full setup (lite: generator + GPU solver).

import bpy
from bpy.types import Operator, Panel

from .generator import build_web_object
from .gpu_solver import enable_gpu_solver, gpu_backend_available
from .strandify import apply_strandify


class ARN_OT_full_setup(Operator):
    """Generate a web with the GPU solver and strandify already applied.
    If a mesh is selected when you click, it becomes the collider
    (and, in Chaotic Cobweb mode, the anchor geometry)"""
    bl_idname = "arachne.full_setup"
    bl_label = "Create Web + Sim + Strands"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        env = [o for o in context.selected_objects if o.type == 'MESH']
        # Web Shot strands already stick where they hit, and the surface
        # they were fired at makes a terrible bounding-sphere collider —
        # its sphere swallows the whole web and blows it apart. Leave the
        # collider unset and let the user pick a moving one.
        collider = None if context.scene.swf_web.mode == 'SHOT' else (
            env[0] if env else None)
        obj = build_web_object(context, context.scene.swf_web, env)
        if obj is None:
            self.report({'ERROR'},
                        "Nothing to build — Chaotic Cobweb needs selected "
                        "mesh geometry to anchor to, Web Shot needs at "
                        "least one valid shot.")
            return {'CANCELLED'}
        # this web gets a solver + strandify — don't let Live Update
        # rebuild it (that would change topology under the solver)
        context.scene.swf_web.live_obj = None
        if gpu_backend_available():
            enable_gpu_solver(context, obj, collider)
        else:
            self.report({'WARNING'},
                        "GPU compute unavailable — web created without "
                        "simulation.")
        apply_strandify(obj)
        self.report({'INFO'}, "Web ready — play from frame 1.")
        return {'FINISHED'}


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
            col.prop(p, "spiral_sag", text="Thread Sag")
            col.prop(p, "jitter")
            col.prop(p, "detail")
            col.prop(p, "seed")
            col.label(text="Select anchor geometry first.", icon='INFO')
        elif p.mode == 'SHOT':
            col.prop(p, "shot_emitter")
            col.prop(p, "shot_aim")
            col.separator()
            col.prop(p, "shot_count")
            col.prop(p, "shot_start")
            col.prop(p, "shot_interval")
            col.prop(p, "shot_speed")
            col.prop(p, "shot_range")
            col.prop(p, "shot_spread")
            col.separator()
            col.prop(p, "shot_clot")
            col.prop(p, "shot_clot_size")
            col.prop(p, "shot_clot_twist")
            col.prop(p, "shot_arc")
            col.prop(p, "shot_whip")
            col.prop(p, "shot_slack")
            col.separator()
            col.prop(p, "shot_splat")
            col.prop(p, "shot_splat_size")
            col.prop(p, "shot_splat_web")
            col.separator()
            col.prop(p, "shot_tangle")
            col.prop(p, "jitter")
            col.prop(p, "detail")
            col.prop(p, "seed")
            col.label(text="Select what the shots hit.", icon='INFO')
            col.label(text="Shots fly once the solver or strands are on.",
                      icon='INFO')
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
            if p.live_obj is not None:
                row.label(text="", icon='CHECKMARK')
            else:
                col.label(text="Generate a web to tweak it live.",
                          icon='INFO')
        col.operator("arachne.full_setup", icon='PLAY')

        box = layout.box()
        box.label(text="GPU Solver", icon='MEMORY')
        col = box.column(align=True)
        if not gpu_backend_available():
            col.label(text="GPU compute unavailable.", icon='ERROR')
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
            col.prop(g, "collider_collection")
            col.prop(g, "collision_offset")
            col.prop(g, "friction")
            col.prop(g, "stickiness")
            col.prop(g, "stick_follow")
            col.separator()
            col.prop(g, "enable_tearing")
            col.prop(g, "tear_threshold")
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


classes = (ARN_OT_full_setup, ARN_PT_main)


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
