from typing import List, Union, Optional, Dict, Iterator, Tuple
import functools
from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, Var
from mutis.ir.graph import Tensor, Operator
from mutis.ir.layout import Layout


class BlockMapping:
    def __init__(
        self,
        hardware_axes: List[Var],
        hardware_num_blocks: List[Expr],
        predicate: Expr,
        virtual_axes_values: Dict[Var, Expr],
    ):
        # the hardware block axes
        self.hardware_axes: List[Var] = hardware_axes
        # the extent of each hardware axis
        self.hardware_num_blocks: List[Expr] = hardware_num_blocks
        # whether the given hardware block axes should participate the computation
        self.predicate: Expr = predicate
        # when predicate evaluates to True, how each virtual axis (block axes and inter block reduce axes) are
        # calculated given the hardware axes
        self.virtual_axes_values: Dict[Var, Expr] = virtual_axes_values

    @staticmethod
    def default_mapping(graph_tile):
        assert isinstance(graph_tile, GraphTile)
        from mutis.vm.ir.impl.program import ThreadBlockMapping_default_mapping

        return ThreadBlockMapping_default_mapping(graph_tile)


class TensorTile:
    def __init__(self, shape: List[Expr]):
        self.orig_shape: List[Expr] = list(shape)
        self.tile_axes: List[List[Var]] = [[] for _ in range(len(shape))]
        self.tile_sizes: List[List[int]] = [[] for _ in range(len(shape))]
        self.linear_tile_axes: List[Var] = []
        self.axes_kind: List[str] = []

    def __str__(self):
        from hidet.ir.tools import IRPrinter

        printer = IRPrinter()

        items = [
            'orig_shape=[{}]'.format(printer(self.orig_shape)),
            'tile_axes=[{}]'.format(
                ', '.join('[{}]'.format(', '.join(str(printer(axis)) for axis in axes)) for axes in self.tile_axes)
            ),
            'tile_sizes=[{}]'.format(
                ', '.join('[{}]'.format(', '.join(str(size) for size in sizes)) for sizes in self.tile_sizes)
            ),
            'linear_tile_axes=[{}]'.format(', '.join(str(printer(axis)) for axis in self.linear_tile_axes)),
            'axes_kind=[{}]'.format(', '.join(self.axes_kind)),
        ]
        return 'tensor_tile({})'.format(', '.join(items))

    def tile(self, dim: int, axis: Var, tile_size: int, kind: str):
        self.tile_axes[dim].append(axis)
        self.tile_sizes[dim].append(tile_size)
        self.linear_tile_axes.append(axis)
        self.axes_kind.append(kind)

    def _extract_tiles_until(
        self, start_axis: Optional[Var], stop_axis: Optional[Var], include_start_axis: bool, include_stop_axis: bool
    ) -> Tuple[List[List[Var]], List[List[int]]]:
        consider_axes: List[Var] = []
        started: bool = False
        for axis in self.linear_tile_axes:
            started = started or start_axis is None or start_axis is axis
            if not started or (started and axis is start_axis and not include_start_axis):
                continue
            if axis is stop_axis:
                if include_stop_axis:
                    consider_axes.append(axis)
                break
            consider_axes.append(axis)

        tile_axes: List[List[Var]] = []
        tile_sizes: List[List[int]] = []
        for axes, sizes in zip(self.tile_axes, self.tile_sizes):
            tile_axes.append([axis for axis, size in zip(axes, sizes) if axis in consider_axes])
            tile_sizes.append([size for axis, size in zip(axes, sizes) if axis in consider_axes])
        return tile_axes, tile_sizes

    def tiled_shape(
        self,
        start_axis: Optional[Var] = None,
        stop_axis: Optional[Var] = None,
        include_start_axis=True,
        include_stop_axis=True,
    ) -> List[int]:
        tile_axes, tile_sizes = self._extract_tiles_until(start_axis, stop_axis, include_start_axis, include_stop_axis)
        return [sizes[-1] for sizes in tile_sizes]

    def tile_indices(
        self,
        start_axis: Optional[Var] = None,
        stop_axis: Optional[Var] = None,
        include_start_axis=True,
        include_stop_axis=True,
    ) -> List[Expr]:
        tile_axes, tile_sizes = self._extract_tiles_until(start_axis, stop_axis, include_start_axis, include_stop_axis)
        indices = []
        for tile_axes, tile_sizes in zip(tile_axes, tile_sizes):
            assert all(a >= b for a, b in zip(tile_sizes[:-1], tile_sizes[1:]))
            tile_strides = [a // tile_sizes[-1] for a in tile_sizes]
            items = [axis * stride for axis, stride in zip(tile_axes, tile_strides)]
            indices.append(sum(items, start=int32.zero))
        return indices

    def tile_offsets(
        self,
        start_axis: Optional[Var] = None,
        stop_axis: Optional[Var] = None,
        include_start_axis=True,
        include_stop_axis=True,
    ) -> List[Expr]:
        tile_axes, tile_sizes = self._extract_tiles_until(start_axis, stop_axis, include_start_axis, include_stop_axis)
        offsets: List[Expr] = []
        for tile_axes, tile_sizes in zip(tile_axes, tile_sizes):
            items = [axis * size for axis, size in zip(tile_axes, tile_sizes)]
            offsets.append(sum(items, start=int32.zero))
        return offsets


class GraphTile:
    def __init__(self):
        self.block_axes: List[Var] = []
        self.inter_block_reduce_axes: List[Var] = []
        self.reduce_axes: List[Var] = []
        self.unroll_axes: List[Var] = []
        self.creator_map: Dict[Var, Operator] = {}
        self.axis_kind: Dict[Var, str] = {}
        self.tile_size_map: Dict[Var, int] = {}
        self.num_tiles_map: Dict[Var, Union[Expr, int]] = {}
        self.tensor2tile: Dict[Tensor, TensorTile] = {}
        self.block_mapping: Optional[BlockMapping] = None

    def copy(self):
        gt = GraphTile()
        gt.block_axes = list(self.block_axes)
        gt.inter_block_reduce_axes = list(self.inter_block_reduce_axes)
        gt.reduce_axes = list(self.reduce_axes)
        gt.unroll_axes = list(self.unroll_axes)
        gt.creator_map = dict(self.creator_map)
        gt.axis_kind = dict(self.axis_kind)
        gt.tile_size_map = dict(self.tile_size_map)
        gt.num_tiles_map = dict(self.num_tiles_map)
        gt.tensor2tile = dict(self.tensor2tile)
        gt.block_mapping = self.block_mapping
        return gt

    def get_var_name(self, kind: str):
        field = '{}_axes'.format(kind)
        assert hasattr(self, field), field
        axes = getattr(self, field)
        kind2prefix = {'block': 'b', 'inter_block_reduce': 'br', 'reduce': 'r', 'unroll': 'u'}
        prefix = kind2prefix[kind]
        return '{}{}'.format(prefix, len(axes))

    def generate_tile_axis(self, tile_size: int, num_tiles: Expr, kind: str, creator: Operator) -> Var:
        axis = Var(self.get_var_name(kind), type=int32)
        if kind == 'block':
            self.block_axes.append(axis)
        elif kind == 'inter_block_reduce':
            self.inter_block_reduce_axes.append(axis)
        elif kind == 'reduce':
            self.reduce_axes.append(axis)
        elif kind == 'unroll':
            self.unroll_axes.append(axis)
        else:
            raise NotImplementedError()
        self.creator_map[axis] = creator
        self.axis_kind[axis] = kind
        self.tile_size_map[axis] = tile_size
        self.num_tiles_map[axis] = num_tiles
        return axis

    def tile(self, tensor: Tensor, dim: int, axis: Var):
        assert axis in self.tile_size_map, 'Please use the axis generated by `generate_tile_axis`'
        if dim < 0:
            dim += len(tensor.shape)
        if tensor not in self.tensor2tile:
            self.tensor2tile[tensor] = TensorTile(tensor.shape)
        self.tensor2tile[tensor].tile(dim, axis, self.tile_size_map[axis], self.axis_kind[axis])
