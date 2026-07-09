# N-panel UI and the one-click full setup.

import bpy
from bpy.types import Operator, Panel

from .generator import build_web_object
from .solver import apply_solver
from .strandify import apply_strandify


class SWF_OT_full_setup(Operator):
    """Generate a web with the solver and strandify already applied.
    If a mesh is selected when you click, it becomes the collider."""
    bl_idname = "swf.full_setup"
    bl_label = "Create Web + Sim + Strands"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        collider = next(
            (o for o in context.selected_objects if o.type == 'MESH'), None)
        obj = build_web_object(context, context.scene.swf_web)
        apply_solver(obj, collider, self.report)
        apply_strandify(obj)
        msg = "Web ready — play from frame 1."
        if collider:
            msg += " Collider: '%s'." % collider.name
        else:
            msg += " Set a Collision Object in the solver modifier."
        self.report({'INFO'}, msg)
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
        col.prop(p, "radials")
        col.prop(p, "rings")
        col.prop(p, "radius")
        col.prop(p, "hub_factor")
        col.prop(p, "jitter")
        col.separator()
        col.prop(p, "anchors")
        col.prop(p, "anchor_extend")
        col.prop(p, "subdiv")
        col.prop(p, "seed")
        col.prop(p, "plane")
        col.separator()
        col.operator("swf.generate_web", icon='ADD')
        col.operator("swf.full_setup", icon='PLAY')

        box = layout.box()
        box.label(text="Tearing Solver", icon='PHYSICS')
        col = box.column(align=True)
        col.operator("swf.add_tearing_solver")
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

        col = layout.column(align=True)
        col.label(text="Play from frame 1 to simulate.", icon='INFO')
        col.label(text="Strandify goes AFTER the solver.")


classes = (SWF_OT_full_setup, SWF_PT_main)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
