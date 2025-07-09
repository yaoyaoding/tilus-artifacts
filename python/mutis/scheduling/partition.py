from typing import List, Dict, Optional, Union
from collections import defaultdict, deque
from hidet.ir.expr import Var
from mutis.ir.graph import Graph, Operator
from mutis.ir.schedule import Partition, Schedule


def list_union(lists: List) -> List:
    ret = []
    for lst in lists:
        for item in lst:
            if all(item is not v for v in ret):
                ret.append(item)
    return ret


def topological_sort(members: List[Union[Operator, Partition]]) -> List[Union[Operator, Partition]]:
    indices = list(range(len(members)))
    edges = {u: [] for u in indices}

    node2idx: Dict[Operator, int] = {}
    for idx, member in enumerate(members):
        if isinstance(member, Operator):
            node2idx[member] = idx
        else:
            for node in member.nodes:
                node2idx[node] = idx
    nodes = list(node2idx)

    for v_node in nodes:
        v_idx = node2idx[v_node]
        for tensor in v_node.inputs:
            u_node = tensor.producer
            if u_node not in node2idx:
                continue
            u_idx = node2idx[u_node]
            if u_idx == v_idx or v_idx in edges[u_idx]:
                continue
            edges[u_idx].append(v_idx)

    in_degree: Dict[int, int] = {idx: 0 for idx in indices}
    for u in edges:
        for v in edges[u]:
            in_degree[v] += 1
    order = []
    queue = deque([u for u in indices if in_degree[u] == 0])
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in edges[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(order) != len(indices):
        raise RuntimeError('Loop detected during topological sorting')

    return [members[i] for i in order]


def partition_according_to(nodes, sch: Schedule, sub_partition_kind: str):
    axis2nodes: Dict[Optional[Var], List[Operator]] = defaultdict(list)
    node2axis: Dict[Operator, Optional[Var]] = {}
    tensor2tile = sch.graph_tile.tensor2tile
    axis_kind_map = sch.graph_tile.axis_kind

    # add the operator that create the sub_partition_kind tile axes
    for axis, op in sch.graph_tile.creator_map.items():
        if axis_kind_map[axis] == sub_partition_kind and op in nodes:
            if op in node2axis:
                raise NotImplementedError('A tensor has been tiled with multiple reduce axes')
            axis2nodes[axis].append(op)
            node2axis[op] = axis

    # add the operator that produce a tensor with given tiles
    for node in nodes:
        if node.output is None:
            continue
        tile_axes = list_union(tensor2tile[node.output].tile_axes)
        for axis in tile_axes:
            if axis_kind_map[axis] == sub_partition_kind:
                if node in node2axis:
                    raise NotImplementedError('A tensor has been tiled with multiple reduce axes')
                axis2nodes[axis].append(node)
                node2axis[node] = axis
    members: List[Union[Operator, Partition]] = []
    for axis, axis_nodes in axis2nodes.items():
        members.append(partition_graph_for_axes(axis_nodes, sch, [axis]))
    for node in nodes:
        if node not in node2axis:
            members.append(node)

    members = topological_sort(members)
    return members


def partition_graph_for_axes(nodes: List[Operator], sch: Schedule, axes: List[Var]) -> Partition:
    kinds: List[str] = [sch.graph_tile.axis_kind[axis] for axis in axes]
    assert len(set(kinds)) == 1, 'All axes must have the same kind, got {}'.format(set(kinds))
    kind = kinds[0]

    if kind == 'block':
        sub_partition_kind = (
            'inter_block_reduce'
            if sch.graph_tile.inter_block_reduce_axes
            else 'reduce'
            if sch.graph_tile.reduce_axes
            else 'unroll'
        )
        members = partition_according_to(nodes, sch=sch, sub_partition_kind=sub_partition_kind)
        return Partition(members=members, axes=axes, kind=kind)
    elif kind == 'inter_block_reduce':
        assert len(axes) == 1
        sub_partition_kind = 'reduce' if sch.graph_tile.reduce_axes else 'unroll'
        members = partition_according_to(nodes, sch=sch, sub_partition_kind=sub_partition_kind)
        return Partition(members=members, axes=axes, kind=kind)
    elif kind == 'reduce':
        assert len(axes) == 1
        members = partition_according_to(nodes, sch=sch, sub_partition_kind='unroll')
        return Partition(members=members, axes=axes, kind=kind)
    elif kind == 'unroll':
        assert len(axes) == 1
        members = topological_sort(nodes)
        return Partition(members=members, axes=axes, kind=kind)
    else:
        raise NotImplementedError()


def partition_graph(graph: Graph, schedules: List[Schedule]) -> List[Schedule]:
    for sch in schedules:
        sch.partition = partition_graph_for_axes(graph.nodes, sch=sch, axes=sch.graph_tile.block_axes)
    return schedules
