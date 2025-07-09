from __future__ import annotations

import functools
from typing import Optional, Union, List, Tuple, cast

from hidet.ir.expr import Var, Expr, logical_and, is_constant
from hidet.ir.type import DataType
from hidet.ir.dtypes import uint32
from hidet.ir.utils.index_transform import index_serialize, index_add, index_multiply, index_divide, index_deserialize
from mutis.backends.codegen import register_emitter, BaseEmitter, NotSupportedEmitter
from mutis.ir.layout import Layout
from mutis.ir.layout import repeat, spatial, greedy_decompose, flatten
from mutis.ops.ldst import Load
from mutis.utils import prod, idiv, is_power_of_two, gcd
from mutis.vm.ir.inst import Instruction, LoadGlobalInst, ViewInst
from mutis.vm.ir.stmt import Stmt, SeqStmt
from mutis.vm.ir.weight_transform import WeightTransform, WeightLayoutTransform
from mutis.vm.ir.builder import VirtualMachineBuilder


class TransformLoadBaseEmitter(BaseEmitter):
    """
    Issue a weight transformation for the loaded tensor to make the data loading more efficient.

    Given a load operator with a specific layout, we will first determine a transformed layout with the same number
    of threads and local size. And the transformed layout will be used to load the data from the layout transformed
    tensor on global memory with maximum efficiency. After that, the tile with transformed layout will be viewed as
    a tile with original layout.

    A concrete example:
    tensor in global memory: float8[14336, 4096] with strides [4096, 1]
    layout: spatial(1, 4).repeat(1, 4).repeat(2, 1).column_spatial(4, 8).repeat(4, 1)

    We will first decompose the original layout into two layouts layout = lhs_layout * rhs_layout such that
    1. rhs_layout has 32 threads
    2. rhs_layout has n elements such that n * bits_of(dtype) // 8 is a multiple of 16, 8, 4, 2, or 1. We want the n as
       large as possible.

    In above example, we will decompose the layout into

        layout = lhs_layout * rhs_layout
        lhs_layout = spatial(1, 4).repeat(1, 2)
        rhs_layout = repeat(1, 2).repeat(2, 1).column_spatial(4, 8).repeat(4, 1)

    where rhs_layout has 32 threads, each thread has 16 bytes of elements. The rhs_layout has shape [32, 16]

    We will issue a weight transformation so that each tile of [32, 16] elements in the float8[144336, 4096] tensor
    will have the following layout:

        [tile 0, tile 1, tile 2, ..., tile ((144336 / 32) * (4096 / 16))]

    where each tile will have 32 * 16 elements. The layout of each tile will be

        [elements reads by thread 0, elements reads by thread 1, ..., elements reads by thread 31]

    With above global memory layout transformation, we can use vanilla LoadGlobalInst to load the data into a tile with

        transformed_layout = spatial(num_warps).repeat(warp_repeats).spatial(warp_size).repeat(n)
        where num_warps = 4, warp_repeats = 2, warp_size = 32, n = 16 in above example.

    After the data is loaded in the transformed layout, we will view the loaded tile as a tile with the original layout.
    """

    def __init__(self, codegen, op, variant):
        super().__init__(codegen, op, variant)
        self.op: Load = cast(Load, op)
        self.dtype: DataType = self.op.dtype

        self.original_layout: Layout = self.tensor2layout[self.op.output]
        self.lhs_layout: Optional[Layout] = None
        self.rhs_layout: Optional[Layout] = None
        self.transformed_dtype: Optional[DataType] = None
        self.transformed_lhs_layout: Optional[Layout] = None
        self.transformed_rhs_layout: Optional[Layout] = None
        self.transformed_layout: Optional[Layout] = None
        self.layout_resolution()

        self.weight_shape: List[int] = [int(a) for a in self.op.shape]
        self.num_tiles: List[int] = index_divide(self.weight_shape, self.rhs_layout.shape)
        # self.lhs_shape: List[int] = self.lhs_layout.shape
        self.transform: Optional[WeightTransform] = None
        self.init_weight_transform()

    def supports_transform(self):
        op: Load = cast(Load, self.op)
        ptr = op.ptr
        # the loaded tensor must be a weight tensor with constant shape and strides
        if not (isinstance(ptr, Var) and ptr in self.graph.params and self.graph.param2attrs[ptr].is_weight):
            raise NotSupportedEmitter()
        if any(not is_constant(v) for v in op.shape) or any(not is_constant(v) for v in op.strides):
            raise NotSupportedEmitter()

        # the right side must have warp_size threads
        warp_size = 32
        if self.rhs_layout.num_workers != warp_size:
            raise NotSupportedEmitter()

        # the shape of the weight tensor must be perfectly divided by the rhs_layout
        if any(a % b != 0 for a, b in zip(op.shape, self.rhs_layout.shape)):
            raise NotSupportedEmitter()

        # if not self.rhs_layout.is_simple():
        #     raise NotSupportedEmitter()

        if not self.original_layout.is_simple():
            raise NotSupportedEmitter()

    def emit_weight_transform(self) -> Optional[Tuple[Var, List[WeightTransform]]]:
        load = cast(Load, self.op)
        ptr = load.ptr
        assert isinstance(ptr, Var)
        return ptr, [self.transform]

    def layout_resolution(self):
        # get the output layout which has the same number of workers and local_size as the in_layout
        # but the elements for each worker is continuous with a vectorized number of elements

        load_max_nbits = 128  # lds128
        warp_size = 32

        if is_power_of_two(self.dtype.nbits):
            self.lhs_layout, self.rhs_layout = greedy_decompose(
                self.original_layout, rhs_max_local_size=load_max_nbits // self.dtype.nbits, rhs_max_workers=warp_size
            )
            self.transformed_dtype = self.dtype
            self.transformed_rhs_layout = spatial(self.rhs_layout.num_workers).repeat(self.rhs_layout.local_size)
        elif self.dtype.nbits * self.original_layout.local_size % 32 == 0:
            self.lhs_layout, self.rhs_layout = greedy_decompose(self.original_layout, rhs_max_workers=warp_size)
            if self.dtype.nbits * self.rhs_layout.local_size % 32 != 0:
                raise NotSupportedEmitter()
            # the number of uint32 in rhs layout
            num_uint32_elements = self.dtype.nbits * self.rhs_layout.local_size // 32
            # we can only load [4, 2, 1] uint32 elements at once. the larger, the better
            uint32_vec_size = gcd(num_uint32_elements, 4)
            uint32_num_vecs = num_uint32_elements // uint32_vec_size

            self.transformed_dtype = uint32
            self.transformed_rhs_layout = (
                repeat(uint32_num_vecs).spatial(self.rhs_layout.num_workers).repeat(uint32_vec_size)
            )
        else:
            raise NotSupportedEmitter()

        # if self.lhs_layout.is_simple():
        #     self.transformed_lhs_layout = repeat(self.lhs_layout.local_size).spatial(self.lhs_layout.num_workers)
        # else:
        self.transformed_lhs_layout = flatten(self.lhs_layout)
        self.transformed_layout = self.transformed_lhs_layout * self.transformed_rhs_layout

        # check support
        self.supports_transform()

    def init_weight_transform(self):
        op = cast(Load, self.op)

        self.transform = WeightLayoutTransform(
            dtype=self.dtype,
            shape=self.weight_shape,
            strides=[int(s) for s in op.strides],
            original_layout=self.rhs_layout,
            transformed_dtype=self.transformed_dtype,
            transformed_layout=self.transformed_rhs_layout,
        )

    def f_ptr(self, outer_tile_indices: List[Expr], axes: List[Expr]):
        """
        Given the outer_tile_indices and axis, return the pointer to the data in the global memory, where
        `outer_tile_indices` are the indices of the operator's output tile while `axis` is the axis of the transformed
        layout.
        """
        lhs_global = axes[0] // self.transformed_rhs_layout.shape[0]
        rhs_global = axes[0] % self.transformed_rhs_layout.shape[0]
        intra_tile_indices = index_deserialize(lhs_global, shape=self.lhs_layout.shape)
        tile_indices = index_add(index_multiply(outer_tile_indices, self.lhs_layout.shape), intra_tile_indices)
        tile_index = index_serialize(tile_indices, shape=self.num_tiles)
        tile_offset = tile_index * self.transformed_rhs_layout.shape[0]
        return tile_offset + rhs_global

    def f_mask(self, outer_tile_indices: List[Expr], axes: List[Expr]):
        outer_tile_offsets = index_multiply(outer_tile_indices, self.tensor2tile[self.op.output].tiled_shape())
        lhs_global = axes[0] // self.transformed_rhs_layout.shape[0]
        intra_tile_indices = index_deserialize(lhs_global, shape=self.lhs_layout.shape)
        intra_tile_offsets = index_multiply(intra_tile_indices, self.rhs_layout.shape)
        tile_offsets = index_add(outer_tile_offsets, intra_tile_offsets)
        return functools.reduce(logical_and, [a < b for a, b in zip(tile_offsets, self.op.shape)])


@register_emitter(Load, priority=1, variant={'stages': 'gmem->regs'})
class LoadEmitter(TransformLoadBaseEmitter):
    """
    load the data with weight transform
    """

    def emit(self) -> Union[Stmt, Instruction]:
        load = cast(Load, self.op)

        def f_ptr(axes: List[Var]) -> Expr:
            outer_tile_indices = self.tensor2tile[load.output].tile_indices()
            return self.f_ptr(outer_tile_indices, axes)

        def f_mask(axes: List[Var]) -> Expr:
            outer_tile_indices = self.tensor2tile[load.output].tile_indices()
            return self.f_mask(outer_tile_indices, axes)

        vb = VirtualMachineBuilder()
        loaded = vb.load_global(
            dtype=self.transformed_dtype, layout=self.transformed_layout, ptr=load.ptr, f_offset=f_ptr, f_mask=f_mask
        )
        viewed = vb.view(loaded, layout=self.original_layout, dtype=self.dtype)
        if load.cast_dtype and load.cast_dtype != load.dtype:
            viewed = vb.cast(viewed, dtype=load.cast_dtype)
        self.codegen.tensor2value[load.output] = viewed
        return vb.finish()
