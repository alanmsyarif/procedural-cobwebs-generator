# Procedural orb-web generator.
#
# Builds an edge-only orb web: hub spokes, radials, a continuous Archimedean
# spiral (shared vertices at every radial/spiral intersection — structurally
# correct for the solver), and extended anchor threads whose endpoints are
# written into the `swf_pin` boolean attribute that the tearing solver binds
# automatically.

import math
import random

import bpy
import bmesh
from bpy.props import (
    IntProperty, FloatProperty, EnumProperty,
)
from bpy.types import Operator, PropertyGroup

from .constants import A_PIN


class SWF_WebProps(PropertyGroup):
    radials: IntProperty(
        name="Radials", default=16, min=3, max=64,
        description="Number of radial threads")
    rings: IntProperty(
        name="Spiral Turns", default=12, min=2, max=60,
        description="Number of turns in the capture spiral")
    radius: FloatProperty(
        name="Radius", default=1.0, min=0.05, max=50.0,
        subtype='DISTANCE', description="Web radius")
    hub_factor: FloatProperty(
        name="Hub Size", default=0.08, min=0.01, max=0.5,
        description="Hub radius as a fraction of the web radius")
    jitter: FloatProperty(
        name="Irregularity", default=0.15, min=0.0, max=1.0,
        description="Organic irregularity of angles, radii and plane offset")
    anchors: IntProperty(
        name="Anchor Threads", default=5, min=1, max=16,
        description="Number of anchor threads extended past the rim")
    anchor_extend: FloatProperty(
        name="Anchor Length", default=0.35, min=0.05, max=3.0,
        description="Anchor thread length as a fraction of the web radius")
    subdiv: IntProperty(
        name="Sag Subdivisions", default=1, min=0, max=3,
        description="Edge subdivisions so strands can sag and bend "
                    "in the solver")
    seed: IntProperty(name="Seed", default=0, min=0)
    plane: EnumProperty(
        name="Plane",
        items=[('XZ', "XZ (vertical)",
                "Vertical web — sags naturally under -Z gravity"),
               ('XY', "XY (horizontal)", "Flat web in the ground plane")],
        default='XZ')


def build_web_object(context, p):
    """Create the web object from properties. Returns the new object."""
    rnd = random.Random(p.seed)
    bm = bmesh.new()
    # tag layer: operators like subdivide_edges reallocate BMesh data and
    # invalidate BMVert references, so anchors are marked in custom data
    # (which survives operators) and re-collected by tag afterwards
    tag = bm.verts.layers.int.new("swf_tag")

    R, N = p.radials, p.rings
    hub = p.radius * p.hub_factor
    total = N * R

    angles = [
        2.0 * math.pi * j / R + rnd.uniform(-1.0, 1.0) * p.jitter * math.pi / R
        for j in range(R)
    ]

    def place(x, y):
        off = rnd.uniform(-1.0, 1.0) * p.jitter * 0.02 * p.radius
        if p.plane == 'XZ':
            return (x, off, y)
        return (x, y, off)

    # vertex grid: ring i, radial j — radius grows continuously with
    # k = i*R + j, which turns the ring stack into one true spiral
    grid = []
    for i in range(N):
        row = []
        for j in range(R):
            k = i * R + j
            t = k / max(total - 1, 1)
            r = hub + (p.radius - hub) * t
            r *= 1.0 + rnd.uniform(-1.0, 1.0) * p.jitter * 0.15
            row.append(bm.verts.new(
                place(math.cos(angles[j]) * r, math.sin(angles[j]) * r)))
        grid.append(row)

    center = bm.verts.new(place(0.0, 0.0))

    # spiral edges (continuous through the ring wrap)
    prev = None
    for i in range(N):
        for j in range(R):
            v = grid[i][j]
            if prev is not None:
                bm.edges.new((prev, v))
            prev = v

    # hub spokes + radial edges
    for j in range(R):
        bm.edges.new((center, grid[0][j]))
        for i in range(N - 1):
            bm.edges.new((grid[i][j], grid[i + 1][j]))

    # anchor threads past the rim; endpoints get pinned
    anchor_ends = []
    step = max(1, R // p.anchors)
    for j in range(0, R, step):
        if len(anchor_ends) >= p.anchors:
            break
        dx, dy = math.cos(angles[j]), math.sin(angles[j])
        prev_v = grid[N - 1][j]
        segs = 3
        for s in range(1, segs + 1):
            rr = p.radius * (1.0 + p.anchor_extend * s / segs)
            v = bm.verts.new(place(dx * rr, dy * rr))
            bm.edges.new((prev_v, v))
            prev_v = v
        prev_v[tag] = 1
        anchor_ends.append(prev_v)

    # subdivide so the solver can create sag between intersections
    if p.subdiv > 0:
        bmesh.ops.subdivide_edges(
            bm, edges=list(bm.edges), cuts=p.subdiv, use_grid_fill=False)

    bm.verts.index_update()
    # re-fetch the tag layer and re-collect anchors: references in
    # anchor_ends may be stale after subdivide_edges
    tag = bm.verts.layers.int.get("swf_tag") or tag
    pin_indices = {v.index for v in bm.verts if v[tag] == 1}
    bm.verts.layers.int.remove(tag)

    me = bpy.data.meshes.new("SpiderWeb")
    bm.to_mesh(me)
    bm.free()

    attr = me.attributes.new(A_PIN, 'BOOLEAN', 'POINT')
    for idx in pin_indices:
        attr.data[idx].value = True

    obj = bpy.data.objects.new("SpiderWeb", me)
    context.collection.objects.link(obj)
    for o in context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    return obj


class SWF_OT_generate_web(Operator):
    """Generate a procedural orb web (anchor endpoints pre-pinned)"""
    bl_idname = "swf.generate_web"
    bl_label = "Generate Web"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        build_web_object(context, context.scene.swf_web)
        return {'FINISHED'}


classes = (SWF_WebProps, SWF_OT_generate_web)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.swf_web = bpy.props.PointerProperty(type=SWF_WebProps)


def unregister():
    del bpy.types.Scene.swf_web
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
