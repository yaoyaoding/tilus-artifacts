from typing import List, Optional, Tuple

from hidet.utils import prod
from mutis.ir.graph import Operator
from mutis.ir.layout import Layout, spatial, squeeze, reduce, expand, repeat
from mutis.ir.schedule import Schedule, Variants
from mutis.ir.tile import GraphTile, BlockMapping
from mutis.ir.graph import Graph, Tensor
from mutis.ops.arithmatic import ElementwiseUnary, ElementwiseBinary, BroadcastElementwiseBinary
from mutis.scheduling import tune
from mutis.scheduling.scheduler import BaseScheduler, register_scheduler
from mutis.utils import cdiv, same_list


class ElementwiseBaseScheduler(BaseScheduler):
    @tune.space(0, {'last_dim_tile': [64]})
    @tune.space(1, {'last_dim_tile': [64, 128, 256]})
    @tune.space(2, {'last_dim_tile': [64, 128, 256]})
    def _tile(self, op: Operator, last_dim_tile: int):
        graph_tile = GraphTile()
        for i in range(len(op.output.shape)):
            tile_size = last_dim_tile if i == len(op.output.shape) - 1 else 1
            axis = graph_tile.generate_tile_axis(
                tile_size=tile_size, num_tiles=cdiv(op.output.shape[i], tile_size), kind='block', creator=op
            )
            for tensor in op.inputs + [op.output]:
                j = i - len(op.output.shape) + len(tensor.shape)
                if j >= 0:
                    graph_tile.tile(tensor, dim=j, axis=axis)
        graph_tile.block_mapping = BlockMapping.default_mapping(graph_tile)
        return graph_tile

    def tile(self, graph, op: Operator) -> List[GraphTile]:
        return tune.extract(self._tile, kwargs={'op': op})

    @tune.space(0, {'vector_size': [1]})
    @tune.space(2, {'vector_size': [1, 2, 4, 8, 16]})
    def _determine_layout_as_anchor(self, op: Operator, sch: Schedule, vector_size: int) -> List[Schedule]:
        x = op.output

        tile = sch.tile_of(x)
        tiled_shape: List[int] = tile.tiled_shape()

        tune.check(tiled_shape[-1] % vector_size == 0)

        repeat_shape = [1 if i != len(tiled_shape) - 1 else vector_size for i in range(len(tiled_shape))]
        spatial_shape = [s if i != len(tiled_shape) - 1 else s // vector_size for i, s in enumerate(tiled_shape)]

        new_sch = sch.copy()
        new_sch.num_warps = cdiv(prod(spatial_shape), 32)
        new_sch.variants = Variants({})
        new_sch.layouts = {x: spatial(*spatial_shape).repeat(*repeat_shape)}
        return new_sch

    def determine_layout_as_anchor(self, graph: Graph, op: Operator, sch: Schedule) -> List[Schedule]:
        return tune.extract(self._determine_layout_as_anchor, kwargs={'op': op, 'sch': sch})

    def propagate_layout(self, op: Operator, sch: Schedule, layouts: List[Optional[Layout]]) -> List[Optional[Layout]]:
        anchor_layout: Optional[Layout] = None
        for layout in layouts:
            if layout and (anchor_layout is None or prod(anchor_layout.shape) < prod(layout.shape)):
                anchor_layout = layout

        assert anchor_layout is not None

        new_layouts = []
        tensors: List[Tensor] = op.inputs + [op.output]
        shapes: List[List[int]] = [sch.tile_of(tensor).tiled_shape() for tensor in tensors]
        for shape, layout in zip(shapes, layouts):
            if layout:
                new_layouts.append(layout)
            else:
                layout = anchor_layout
                # squeeze or expand to make the rank consistent
                if len(shape) > len(layout.shape):
                    layout = expand(layout, list(range(len(shape) - len(layout.shape))))
                elif len(shape) < len(layout.shape):
                    reduce_dims = [i for i in range(len(layout.shape) - len(shape)) if layout.shape[i] > 1]
                    if reduce_dims:
                        layout = reduce(layout, reduce_dims, squeeze_dims=False)
                    squeeze_dims = list(range(len(layout.shape) - len(shape)))
                    layout = squeeze(layout, squeeze_dims)

                dims = list(range(len(shape)))

                # repeat
                repeat_dims = [i for i in dims if shape[i] > layout.shape[i]]
                if repeat_dims:
                    assert [shape[i] % layout.shape[i] == 0 for i in repeat_dims]
                    repeat_shape = [shape[i] // layout.shape[i] for i in repeat_dims]
                    layout = repeat(*repeat_shape) * layout

                # reduce
                reduce_dims = [i for i in dims if shape[i] < layout.shape[i]]
                if reduce_dims:
                    assert [shape[i] == 1 for i in reduce_dims]
                    layout = reduce(layout, reduce_dims, squeeze_dims=False)

                assert same_list(shape, layout.shape)
                new_layouts.append(layout)
        return new_layouts


@register_scheduler(ElementwiseUnary)
class ElementwiseUnaryScheduler(ElementwiseBaseScheduler):
    pass


@register_scheduler(ElementwiseBinary)
class ElementwiseBinaryScheduler(ElementwiseBaseScheduler):
    pass


@register_scheduler(BroadcastElementwiseBinary)
class BroadcastElementwiseBinaryScheduler(ElementwiseBaseScheduler):
    pass
