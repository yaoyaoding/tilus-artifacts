from typing import List, Dict, Tuple

from hidet.ir.expr import Var
from hidet.ir.type import DataType
from hidet.utils import prod
from mutis.ir.graph import Graph, Operator, Tensor
from mutis.ir.layout import spatial, repeat, expand
from mutis.ir.schedule import Schedule, Variants
from mutis.ir.tile import GraphTile, TensorTile, BlockMapping
from mutis.ir.layout import Layout
from mutis.ir.analyzers.usage_analyzer import analyze_usage
from mutis.ops.transform import Cast
from mutis.ops.matmul import Matmul
from mutis.ops.arithmatic import ElementwiseBinary, ElementwiseUnary, BroadcastElementwiseBinary
from mutis.ops.ldst import Load, Store
from mutis.vm.ir.shared_layout import shared_repeat, shared_compose, shared_column_repeat
from mutis.scheduling import tune
from mutis.scheduling.scheduler import BaseScheduler, register_scheduler
from mutis.utils import cdiv, gcd


@register_scheduler(Load)
class LoadScheduler(BaseScheduler):
    @tune.space(
        0,  #
        {
            'stages, pipeline': [
                # ['gmem->regs', 'ldg', None], #
                # ['gmem->smem->regs', 2]
                ['gmem->smem->regs', 3]
            ]
        },
    )
    @tune.space(
        2,
        {
            'stages, pipeline': [
                # ['gmem->regs', None],
                # ['gmem->smem->regs', 2],
                ['gmem->smem->regs', 3],
                ['gmem->smem->regs', 4],
            ]
        },
    )
    def _determine_variant_for_reduce_load(self, stages: str, pipeline, op: Operator, sch: Schedule):
        new_sch = sch.copy()
        new_sch.variants.set_variant(op, 'stages', stages)
        new_sch.variants.set_variant(op, 'pipeline', pipeline)
        return new_sch

    def _auto_layout(self, m: int, n: int, threads: int, order: int) -> Layout:
        if order == 0:
            dims = [0, 1]
        else:
            dims = [1, 0]

        spatial_shape = [1, 1]
        remain_shape = [m, n]
        remain_threads = threads
        for dim in dims:
            spatial_shape[dim] = gcd(remain_threads, remain_shape[dim])
            remain_threads //= spatial_shape[dim]
            remain_shape[dim] //= spatial_shape[dim]
        ret = repeat(*remain_shape).spatial(*spatial_shape)
        return ret

    def _determine_simt_loading_a_regs_layout(self, dtype: DataType, m: int, k: int, threads: int) -> Layout:
        # shared memory is column major order
        if dtype.nbits % 8 != 0 or 128 % dtype.nbits != 0:
            return spatial(threads // k, k).repeat(m // (threads // k), 1)
        vec = gcd(128 // dtype.nbits, m)
        used_threads = gcd(threads, m * k // vec)
        return self._auto_layout(m // vec, k, used_threads, order=0).repeat(vec, 1)

    def _determine_simt_loading_b_regs_layout(self, dtype: DataType, k: int, n: int, threads: int) -> Layout:
        # shared memory is row major order
        if dtype.nbits % 8 != 0 or 128 % dtype.nbits != 0:
            return repeat(1, n // (threads // k)).spatial(k, threads // k)
        vec = gcd(128 // dtype.nbits, n)
        used_threads = gcd(threads, k * n // vec)
        return self._auto_layout(k, n // vec, used_threads, order=1).repeat(1, vec)

    def annotate_simt_load_hint(self, graph: Graph, op: Operator, sch: Schedule):
        if sch.variants.get_variant(op, 'stages') != 'gmem->smem->regs':
            return
        pass_through = (Cast, ElementwiseBinary, ElementwiseUnary, BroadcastElementwiseBinary)
        usage: Dict[Tensor, List[Tuple[Operator, int]]] = analyze_usage(graph)

        found_consumers: List[Tuple[Operator, int]] = []
        consumers: List[Tuple[Operator, int]] = [(op, idx) for op, idx in usage[op.output]]
        visited = set(consumers)

        while consumers:
            u, idx = consumers.pop()
            if not isinstance(u, pass_through):
                found_consumers.append((u, idx))
                continue
            if u.output is None:
                continue
            for v, idx in usage[u.output]:
                if v not in visited:
                    consumers.append((v, idx))
                    visited.add((v, idx))

        if len(found_consumers) != 1:
            return
        consumer_op, consumer_input_idx = found_consumers[0]
        if not isinstance(consumer_op, Matmul):
            return

        if consumer_op not in sch.variants.op2variant:
            return

        matmul_variant = sch.variants.op2variant[consumer_op]
        if matmul_variant['inst'] != 'simt':
            return

        tensor_tile = sch.tile_of(op.output)
        reduce_axes = [
            axis for kind, axis in zip(tensor_tile.axes_kind, tensor_tile.linear_tile_axes) if kind == 'reduce'
        ]
        if len(reduce_axes) != 1:
            return
        reduce_axis = reduce_axes[0]
        if op.output.elem_type.nbits % 8 != 0:
            return
        shared_shape: List[int] = sch.tile_of(op.output).tiled_shape(stop_axis=reduce_axis, include_stop_axis=True)
        assert all(s == 1 for s in shared_shape[:-2])
        num_threads = sch.num_warps * 32
        if consumer_input_idx == 0:
            # a
            sch.variants.set_variant(op, name='shared_layout_hint', value=shared_column_repeat(*shared_shape))
            m, k = shared_shape[-2:]
            assert num_threads % k == 0
            assert m % (num_threads // k) == 0
            sch.variants.set_variant(
                op,
                name='g2s_layout_hint',
                value=expand(
                    self._determine_simt_loading_a_regs_layout(op.output.elem_type, m, k, num_threads),
                    dims=list(range(len(shared_shape) - 2)),
                ),
            )
        elif consumer_input_idx == 1:
            # b
            sch.variants.set_variant(op, name='shared_layout_hint', value=shared_repeat(*shared_shape))
            k, n = shared_shape[-2:]
            assert num_threads % k == 0
            assert n % (num_threads // k) == 0
            sch.variants.set_variant(
                op,
                name='g2s_layout_hint',
                value=expand(
                    self._determine_simt_loading_b_regs_layout(op.output.elem_type, k, n, num_threads),
                    dims=list(range(len(shared_shape) - 2)),
                ),
            )
        else:
            assert False

    def determine_variant(self, graph: Graph, op: Operator, sch: Schedule) -> List[Schedule]:
        tensor: Tensor = op.output
        tensor_tile: TensorTile = sch.tile_of(tensor)

        reduce_axes: List[Var] = []

        for axis in tensor_tile.linear_tile_axes:
            if sch.graph_tile.axis_kind[axis] == 'reduce':
                reduce_axes.append(axis)

        if len(reduce_axes) > 1:
            # support when needed, for now, we do not encounter this case in real world applications
            raise NotImplementedError()
        elif len(reduce_axes) == 0:
            # load outside any loop, use plain loading
            new_sch = sch.copy()
            new_sch.variants.set_variant(op, 'stages', 'gmem->regs')
            new_sch.variants.set_variant(op, 'pipeline', None)
            return [new_sch]

        # load inside a loop
        reduce_axis = reduce_axes[0]
        other_loads: List[Load] = []  # other loads in the same reduce axis
        for node in graph.nodes:
            if isinstance(node, Load) and reduce_axis in sch.tile_of(node.output).linear_tile_axes:
                other_loads.append(node)
        other_loads_with_variants = [
            other_load
            for other_load in other_loads
            if other_load in sch.variants.op2variant and sch.variants.op2variant[other_load]['stages'] != 'gmem->regs'
        ]
        schedules = []
        if other_loads_with_variants:
            # loads in the same reduce loop should share the same variant
            new_sch = sch.copy()
            other_load = other_loads_with_variants[0]
            new_sch.variants.set_variant(op, 'stages', new_sch.variants.get_variant(other_load, 'stages'))
            new_sch.variants.set_variant(op, 'pipeline', new_sch.variants.get_variant(other_load, 'pipeline'))
            schedules.append(new_sch)
        else:
            schedules.extend(tune.extract(self._determine_variant_for_reduce_load, kwargs={'op': op, 'sch': sch}))

        # annotate hint
        updated_schedules = []
        for schedule in schedules:
            try:
                self.annotate_simt_load_hint(graph, op, schedule)
            except AssertionError:
                pass
            else:
                updated_schedules.append(schedule)
        return updated_schedules


@register_scheduler(Store)
class StoreScheduler(BaseScheduler):
    @tune.space(0, {'last_dim_tile': [64]})
    @tune.space(1, {'last_dim_tile': [64, 128, 256]})
    @tune.space(2, {'last_dim_tile': [64, 128, 256]})
    def _tile(self, op: Store, last_dim_tile: int):
        graph_tile = GraphTile()
        x = op.inputs[0]

        for i in range(len(x.shape)):
            tile_size = last_dim_tile if i == len(x.shape) - 1 else 1
            axis = graph_tile.generate_tile_axis(
                tile_size=tile_size, num_tiles=cdiv(x.shape[i], tile_size), kind='block', creator=op
            )
            graph_tile.tile(x, dim=i, axis=axis)
        graph_tile.block_mapping = BlockMapping.default_mapping(graph_tile)
        return graph_tile

    def tile(self, graph, op: Store) -> List[GraphTile]:
        return tune.extract(self._tile, kwargs={'op': op})

    @tune.space(0, {'vector_size': [1]})
    @tune.space(2, {'vector_size': [1, 2, 4]})
    def _determine_layout_as_anchor(self, op: Operator, sch: Schedule, vector_size: int) -> List[Schedule]:
        x = op.inputs[0]

        tile = sch.tile_of(x)
        tiled_shape: List[int] = tile.tiled_shape()

        tune.check(tiled_shape[-1] % vector_size == 0)

        repeat_shape = [1 if i != len(tiled_shape) - 1 else vector_size for i in range(len(tiled_shape))]
        spatial_shape = [s if i != len(tiled_shape) - 1 else s // vector_size for i, s in enumerate(tiled_shape)]

        tune.check(prod(spatial_shape) % 32 == 0)

        new_sch = sch.copy()
        new_sch.num_warps = cdiv(prod(spatial_shape), 32)
        new_sch.variants = Variants({})
        new_sch.layouts = {x: spatial(*spatial_shape).repeat(*repeat_shape)}
        return new_sch

    def determine_layout_as_anchor(self, graph: Graph, op: Operator, sch: Schedule) -> List[Schedule]:
        return tune.extract(self._determine_layout_as_anchor, kwargs={'op': op, 'sch': sch})
