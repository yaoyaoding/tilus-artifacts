from typing import List, Dict, Tuple

from mutis.ir import Tensor, Graph, Operator
from mutis.ir.analyzers import analyze_usage
from mutis.ir.schedule import Schedule
from mutis.ir.tile import TensorTile, GraphTile
from mutis.scheduling.scheduler import resolve_scheduler


def get_anchor_operator(graph: Graph) -> Operator:
    from mutis.ops.arithmatic import ElementwiseBinary, ElementwiseUnary, BroadcastElementwiseBinary
    from mutis.ops.attention import Attention
    from mutis.ops.ldst import Store, Load
    from mutis.ops.matmul import Matmul
    from mutis.ops.transform import Cast

    exclusive_operators = (Attention, Matmul)
    operators_in_order = (
        Attention,
        Matmul,
        ElementwiseBinary,
        ElementwiseUnary,
        BroadcastElementwiseBinary,
        Store,
        Load,
        Cast,
    )

    nodes = graph.nodes.copy()
    for node in nodes:
        if not isinstance(node, operators_in_order):
            raise NotImplementedError(type(node))

    exclusive_operators = [node for node in nodes if isinstance(node, exclusive_operators)]
    if len(exclusive_operators) > 1:
        raise NotImplementedError('More than one exclusive operators: {}'.format(exclusive_operators))
    if len(exclusive_operators) == 1:
        return exclusive_operators[0]

    nodes = sorted(nodes, key=lambda nd: operators_in_order.index(nd.__class__))
    return nodes[0]


def generate_initial_graph_tiles(graph: Graph) -> List[GraphTile]:
    anchor = get_anchor_operator(graph)
    scheduler = resolve_scheduler(anchor)
    return scheduler.tile(graph, anchor)


def propagate_graph_tile(graph: Graph, graph_tile: GraphTile):
    queue: List[Tensor] = []
    usage: Dict[Tensor, List[Tuple[Operator, int]]] = analyze_usage(graph)

    for node in graph.nodes:
        tensor = node.output
        if tensor is None or tensor not in graph_tile.tensor2tile:
            continue
        queue.append(tensor)

    def propagate(u: Tensor, v: Tensor, u_idx: int, v_idx: int, sets: List[List[Tuple[int, int]]]):
        if v in graph_tile.tensor2tile:
            # todo: check whether the existing tiling matches
            return
        u_tile = graph_tile.tensor2tile[u]
        v_tile = TensorTile(v.shape)
        for st in sets:
            for u_dim in range(len(u.shape)):
                if (u_idx, u_dim) not in st:
                    continue
                for v_dim in range(len(v.shape)):
                    if (v_idx, v_dim) not in st:
                        continue
                    for axis in u_tile.tile_axes[u_dim]:
                        v_tile.tile(v_dim, axis, graph_tile.tile_size_map[axis], graph_tile.axis_kind[axis])
        graph_tile.tensor2tile[v] = v_tile
        queue.append(v)

    while queue:
        u: Tensor = queue.pop()

        # expand to the producer's inputs
        op = u.producer
        sets: List[List[Tuple[int, int]]] = op.tile_propagation_sets()
        for i, v in enumerate(op.inputs):
            u_idx = len(op.inputs)
            v_idx = i
            propagate(u, v, u_idx, v_idx, sets)

        # expand to the output of the consumer op and other inputs
        for consumer, i in usage[u]:
            sets = consumer.tile_propagation_sets()
            for j, v in enumerate(consumer.inputs):
                u_idx = i
                v_idx = j
                if i == j:
                    # self
                    continue
                propagate(u, v, u_idx, v_idx, sets)
            if consumer.output:
                u_idx = i
                v_idx = len(consumer.inputs)
                propagate(u, consumer.output, u_idx, v_idx, sets)


def sort_tile_axes(graph: Graph, graph_tile: GraphTile):
    kind_rank = {'block': 0, 'inter_block_reduce': 1, 'reduce': 2, 'unroll': 3}
    for tensor, tile in graph_tile.tensor2tile.items():
        pairs = [(axis, kind) for axis, kind in zip(tile.linear_tile_axes, tile.axes_kind)]
        pairs.sort(key=lambda pair: (kind_rank[pair[1]], graph_tile.tile_size_map[pair[0]]))
        tile.linear_tile_axes = [pair[0] for pair in pairs]
        tile.axes_kind = [pair[1] for pair in pairs]


def tile_graph(graph: Graph) -> List[Schedule]:
    anchor = get_anchor_operator(graph)
    graph_tiles = generate_initial_graph_tiles(graph)

    schedules: List[Schedule] = []
    for graph_tile in graph_tiles:
        propagate_graph_tile(graph, graph_tile)
        sort_tile_axes(graph, graph_tile)
        schedules.append(Schedule(anchor_op=anchor, graph_tile=graph_tile))

    return schedules
