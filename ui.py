# N-panel UI and the one-click full setup (lite: generator + GPU solver).

import bpy
from bpy.types import Operator, Panel

from .generator import build_web_object
from .gpu_solver import enable_gpu_solver, gpu_backend_available
from .strandify import apply_strandify


class SWF_OT_full_setup(Operator):
    """Generate a web with the GPU solver and strandify already applied.
    If a mesh is selected when you click, it becomes the collider
    (and, in Chaotic Cobweb mode, the anchor geometry)"""
    bl_idname = "swf.full_setup"
    bl_label = "Create Web + Sim + Strands"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        env = [o for o in context.selected_objects if o.type == 'MESH']
        collider = env[0] if env else None
        obj = build_web_object(context, context.scene.swf_web, env)
        if obj is None:
            self.report({'ERROR'},
                        "Chaotic Cobweb needs selected mesh geometry to "
                        "anchor to.")
            return {'CANCELLED'}
        if gpu_backend_available():
            enable_gpu_solver(context, obj, collider)
        else:
            self.report({'WARNING'},
                        "GPU compute unavailable — web created without "
                        "simulation.")
        apply_strandify(obj)
        self.report({'INFO'}, "Web ready — play from frame 1.")
        return {'FINISHED'}


class SWF_PT_main(Panel):
    bl_label = "Spider Web Forge"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Web Forge"

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
            col.prop(p, "radius", text="Anchor Span")
            col.prop(p, "spiral_sag", text="Thread Sag")
            col.prop(p, "jitter")
            col.prop(p, "detail")
            col.prop(p, "seed")
            col.label(text="Select anchor geometry first.", icon='INFO')
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
        col.operator("swf.generate_web", icon='ADD')
        col.operator("swf.full_setup", icon='PLAY')

        box = layout.box()
        box.label(text="GPU Solver", icon='MEMORY')
        col = box.column(align=True)
        if not gpu_backend_available():
            col.label(text="GPU compute unavailable.", icon='ERROR')
        col.operator("swf.add_gpu_solver", icon='PLAY')
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
            if g.collision_shape == 'MESH_SDF':
                col.prop(g, "sdf_resolution")
            col.prop(g, "collider")
            col.prop(g, "collision_offset")
            col.prop(g, "friction")
            col.separator()
            col.prop(g, "enable_tearing")
            col.prop(g, "tear_threshold")
            col.separator()
            row = col.row(align=True)
            row.operator("swf.reset_gpu", icon='FILE_REFRESH')
            row.operator("swf.remove_gpu_solver", text="Remove", icon='X')
        col.separator()
        col.label(text="Anchors (Edit Mode):")
        row = col.row(align=True)
        row.operator("swf.pin_vertices", text="Pin").action = 'PIN'
        row.operator("swf.pin_vertices", text="Unpin").action = 'UNPIN'
        col.operator("swf.pin_vertices",
                     text="Clear All Pins").action = 'CLEAR'

        box = layout.box()
        box.label(text="Render", icon='CURVES')
        col = box.column(align=True)
        col.operator("swf.add_strandify")
        col.separator()
        row = col.row(align=True)
        row.operator("swf.bake_dew", icon='RENDER_ANIMATION')
        row.operator("swf.free_dew_bake", text="Free", icon='X')
        col.label(text="Bake before rendering animations.", icon='INFO')

        col = layout.column(align=True)
        col.label(text="Play from frame 1 to simulate.", icon='INFO')


classes = (SWF_OT_full_setup, SWF_PT_main)


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
