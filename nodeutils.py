# Node-tree construction helpers.
#
# All socket access is by NAME (or by enabled-socket lookup for nodes with
# per-type socket sets like Compare and Random Value) so the builders survive
# the socket-index reshuffles introduced in Blender 5.x.

import bpy


def compare_ab(node):
    """(A, B) input sockets of a Compare node for its current data_type."""
    ab = [s for s in node.inputs if s.enabled and s.name in ("A", "B")]
    return ab[0], ab[1]


def minmax_sockets(node):
    """(Min, Max) input sockets of a Random Value node for its data_type."""
    mm = [s for s in node.inputs if s.enabled and s.name in ("Min", "Max")]
    return mm[0], mm[1]


class H:
    """Helper namespace bound to one node tree."""

    def __init__(self, nt):
        self.nt = nt
        self.nodes = nt.nodes
        self.links = nt.links

    # -- basics ------------------------------------------------------------
    def n(self, idname, x, y, label="", **props):
        node = self.nodes.new(idname)
        node.location = (x, y)
        if label:
            node.label = label
        for k, v in props.items():
            setattr(node, k, v)
        return node

    def lk(self, out_sock, in_sock):
        self.links.new(out_sock, in_sock)

    def _feed(self, sock, v):
        if v is None:
            return
        if isinstance(v, bpy.types.NodeSocket):
            self.lk(v, sock)
        else:
            sock.default_value = v

    # -- math --------------------------------------------------------------
    def ma(self, op, x, y, a=None, b=None, label=""):
        node = self.n("ShaderNodeMath", x, y, label=label, operation=op)
        self._feed(node.inputs[0], a)
        self._feed(node.inputs[1], b)
        return node

    def vm(self, op, x, y, a=None, b=None, label=""):
        node = self.n("ShaderNodeVectorMath", x, y, label=label, operation=op)
        self._feed(node.inputs[0], a)
        self._feed(node.inputs[1], b)
        return node

    def vscale(self, x, y, vec, scale, label=""):
        node = self.n("ShaderNodeVectorMath", x, y, label=label,
                      operation='SCALE')
        self._feed(node.inputs[0], vec)
        self._feed(node.inputs["Scale"], scale)
        return node

    def cmp(self, dtype, op, x, y, a=None, b=None, label=""):
        node = self.n("FunctionNodeCompare", x, y, label=label,
                      data_type=dtype, operation=op)
        sa, sb = compare_ab(node)
        self._feed(sa, a)
        self._feed(sb, b)
        return node

    def bmath(self, op, x, y, a=None, b=None):
        node = self.n("FunctionNodeBooleanMath", x, y, operation=op)
        if a is not None:
            self.lk(a, node.inputs[0])
        if b is not None:
            self.lk(b, node.inputs[1])
        return node

    # -- attributes ----------------------------------------------------------
    def named(self, dtype, name, x, y):
        node = self.n("GeometryNodeInputNamedAttribute", x, y,
                      label=name, data_type=dtype)
        node.inputs["Name"].default_value = name
        return node

    def store(self, dtype, domain, name, x, y, geo=None, value=None, sel=None):
        node = self.n("GeometryNodeStoreNamedAttribute", x, y,
                      label="Store " + name, data_type=dtype, domain=domain)
        node.inputs["Name"].default_value = name
        if geo is not None:
            self.lk(geo, node.inputs["Geometry"])
        self._feed(node.inputs["Value"], value)
        if sel is not None:
            self.lk(sel, node.inputs["Selection"])
        return node

    def sample(self, dtype, domain, x, y, geo, value, index, label=""):
        node = self.n("GeometryNodeSampleIndex", x, y, label=label,
                      data_type=dtype, domain=domain)
        self.lk(geo, node.inputs["Geometry"])
        self.lk(value, node.inputs["Value"])
        self.lk(index, node.inputs["Index"])
        return node

    # -- zones ---------------------------------------------------------------
    def sim_zone(self, x_in, x_out, y=0):
        s_in = self.n("GeometryNodeSimulationInput", x_in, y)
        s_out = self.n("GeometryNodeSimulationOutput", x_out, y)
        s_in.pair_with_output(s_out)
        if not s_out.state_items:
            s_out.state_items.new('GEOMETRY', "Geometry")
        return s_in, s_out

    def repeat_zone(self, x_in, x_out, y=0, label=""):
        r_in = self.n("GeometryNodeRepeatInput", x_in, y, label=label)
        r_out = self.n("GeometryNodeRepeatOutput", x_out, y)
        r_in.pair_with_output(r_out)
        if not r_out.repeat_items:
            r_out.repeat_items.new('GEOMETRY', "Geometry")
        return r_in, r_out


def sock_in(iface, name, stype, default=None, minv=None, maxv=None):
    s = iface.new_socket(name=name, in_out='INPUT', socket_type=stype)
    if default is not None:
        try:
            s.default_value = default
        except (AttributeError, TypeError):
            pass
    for attr, val in (("min_value", minv), ("max_value", maxv)):
        if val is not None:
            try:
                setattr(s, attr, val)
            except AttributeError:
                pass
    return s


def input_identifier(group, sock_name):
    """Interface identifier of a group input socket (for modifier IDProps)."""
    for item in group.interface.items_tree:
        if (getattr(item, "item_type", "") == 'SOCKET'
                and item.in_out == 'INPUT' and item.name == sock_name):
            return item.identifier
    return None
