from __future__ import annotations

from typing import Optional, Union, Dict, Any, cast, List

from hidet.ir.expr import Var, Expr, logical_and
from hidet.ir.type import pointer_type, DataType
from hidet.ir.dtypes import uint8, uint32, int32, boolean
from hidet.ir.utils.index_transform import index_serialize, index_sum, index_multiply, index_add
from hidet.ir.primitives.cuda.vars import threadIdx
from mutis.ir.graph import Operator
from mutis.ir.tile import TensorTile, GraphTile
from mutis.ir.layout import Layout
from mutis.vm.ir.inst import Instruction, AllocateInst, ElementwiseBinaryInst, SimtDotInst
from mutis.vm.ir.inst import ShuffleDownInst, ShuffleUpInst, CastInst, AllocateScalarInst, AllocateGlobalInst
from mutis.vm.ir.builder import VirtualMachineBuilder
from mutis.vm.ir.stmt import Stmt, SeqStmt
from mutis.vm.ir.value import RegisterValue
from mutis.backends.codegen import register_emitter, ReduceEmitter, InterBlockReduceEmitter, NotSupportedEmitter
from mutis.ops.matmul import Matmul
from mutis.utils import floor_log2, prod, is_power_of_two
from mutis.backends.emitters.utils import access_efficient_layout


@register_emitter(Matmul, priority=1)
class MatmulSimtEmitter(InterBlockReduceEmitter, ReduceEmitter):
    def __init__(self, codegen, op: Operator, variant: Dict[str, Any]):
        super().__init__(codegen, op, variant)
        self.op: Matmul = cast(Matmul, op)
        self.accumulator: Optional[RegisterValue] = None
        self.output: Optional[RegisterValue] = None

        # used for inter-block reduction
        self.num_block_reduce_tiles: Optional[Expr] = None
        self.block_indices: Optional[List[Expr]] = None
        self.block_shape: Optional[List[Expr]] = None
        self.block_offsets: Optional[List[Expr]] = None
        self.block_index: Optional[Expr] = None
        self.tiled_shape: Optional[List[int]] = None
        self.intermediate: Optional[Var] = None
        self.semaphore: Optional[Var] = None

        if variant['inst'] != 'simt':
            raise NotSupportedEmitter()

    def intermediate_nbytes(self, br_axis: Var) -> Expr:
        tensor_tile: TensorTile = self.tensor2tile[self.op.output]
        layout: Layout = self.tensor2layout[self.op.output]
        self.tiled_shape: List[int] = tensor_tile.tiled_shape()
        self.block_indices = tensor_tile.tile_indices()
        self.block_offsets = tensor_tile.tile_offsets()
        self.block_shape = [(a + (b - 1)) // b for a, b in zip(self.op.output.shape, self.tiled_shape)]
        self.block_index = index_serialize(self.block_indices, self.block_shape)
        self.num_block_reduce_tiles: Expr = self.num_tiles_map[br_axis]
        return prod(self.block_shape) * (layout.num_workers * layout.local_size * self.op.output.elem_type.nbytes)

    def semaphore_nbytes(self) -> Expr:
        return prod(self.block_shape) * int32.nbytes

    def emit_inter_block_reduce_prologue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        dtype = self.op.output.elem_type
        self.intermediate = Var('intermediate', type=~dtype)
        self.semaphore = Var('semaphore', type=~uint32)
        alloc_intermediate_inst = AllocateGlobalInst(self.intermediate, nbytes=self.intermediate_nbytes(axis))
        alloc_semaphore_inst = AllocateGlobalInst(self.semaphore, nbytes=self.semaphore_nbytes(), require_clean=True)
        return SeqStmt([alloc_intermediate_inst, alloc_semaphore_inst])

    def emit_reduce_prologue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        op = cast(Matmul, self.op)
        inst = AllocateInst.create(
            dtype=op.acc_dtype, layout=self.tensor2layout[op.output], f_init=lambda axes: op.acc_dtype.zero
        )
        self.accumulator = inst.output
        return inst

    def emit(self) -> Union[Stmt, Instruction]:
        assert self.accumulator is not None
        inst = SimtDotInst(
            d=self.accumulator,
            a=self.tensor2value[self.op.inputs[0]].as_register_value(),
            b=self.tensor2value[self.op.inputs[1]].as_register_value(),
            c=self.accumulator,
            warp_spatial=self.variant['warp_spatial'],
            warp_repeat=self.variant['warp_repeat'],
            thread_spatial=self.variant['simt_thread_spatial'],
            thread_repeat=self.variant['simt_thread_repeat'],
        )
        return inst

    def emit_reduce_epilogue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        op: Matmul = cast(Matmul, self.op)
        warp_spatial = self.variant['warp_spatial']
        seq = []

        # cast the accumulator (e.g., fp32) to a lower precision (e.g., fp16) and store it to output
        if self.op.acc_dtype != self.op.out_dtype:
            self.output = RegisterValue(dtype=op.out_dtype, layout=self.tensor2layout[op.output])
            seq.append(CastInst(self.output, self.accumulator))
        else:
            self.output = self.accumulator

        if warp_spatial[2] > 1:
            # reduce over warps along the k axis
            # example:
            # when warp_spatial=(1, 2, 4) where m=1, n=2, k=4 and ranks=[1, 2, 0]
            # the emitted instructions will be:
            #   temp = allocate(...)
            #   shuffle_down(temp, output, mask=0b11111111, delta=4, width=8)
            #   binary_add(output, temp, output)
            #   shuffle_down(temp, output, cond=0b00001111, delta=2, width=4)
            #   binary_add(output, temp, output)

            # reduce
            output = self.output
            temp = RegisterValue(dtype=output.dtype, layout=output.layout)
            seq.append(AllocateInst(temp, axes=None, init=None))
            width: int = warp_spatial[0] * warp_spatial[1] * warp_spatial[2]
            assert is_power_of_two(warp_spatial[2])
            reduce_repeat = floor_log2(warp_spatial[2])
            masks, deltas, widths = [], [], []
            for i in range(reduce_repeat):
                masks.append((1 << (width >> i)) - 1)
                deltas.append(width >> (i + 1))
                widths.append(width >> i)
            for mask, delta, width in zip(masks, deltas, widths):
                seq.append(ShuffleDownInst(output=temp, x=output, mask=mask, delta=delta, width=width))
                seq.append(ElementwiseBinaryInst(output=output, x=output, y=temp, op='+'))

            # propagate back
            for mask, delta, width in reversed(list(zip(masks, deltas, widths))):
                seq.append(ShuffleUpInst(output=output, x=output, mask=mask, delta=delta, width=width))

        self.tensor2value[op.output] = self.output

        return SeqStmt(seq)

    def wait_semaphore(self, vb, value: Expr):
        semephore_value = vb.allocate_scalar('semaphore_value', scalar_type=int32, init=-int32.one)
        with vb.while_loop(boolean.true):
            with vb.if_then(threadIdx.x == 0):
                vb.assign_scalar(semephore_value, vb.load_scalar(ptr=self.semaphore + self.block_index, sync='acquire'))
            cond = vb.syncthreads_or(semephore_value == value)
            with vb.if_then(cond):
                vb.brk()

    def release_semaphore(self, vb, value: Expr):
        with vb.if_then(threadIdx.x == 0):
            # set the semaphore to the new value
            vb.store_scalar(ptr=self.semaphore + self.block_index, value=value, sync='release')

    def emit_inter_block_reduce_epilogue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        vb = VirtualMachineBuilder()
        dtype: DataType = self.op.output.elem_type
        original_x = self.tensor2value[self.op.output].as_register_value()
        original_layout = original_x.layout
        c_block_size = original_layout.num_workers * original_layout.local_size

        def get_block_offset():
            block_strides: List[Expr] = [
                prod(self.block_shape[i + 1 :]) * c_block_size for i in range(len(self.block_shape))
            ]
            block_offset = index_sum(index_multiply(self.block_indices, block_strides))
            return block_offset

        # whether the tile is full inside the c matrix
        full_inside = logical_and(
            *[
                (self.block_indices[i] + 1) * self.tiled_shape[i] <= self.op.output.shape[i]
                for i in range(len(self.block_shape))
            ]
        )

        with vb.if_then(full_inside):
            # when the tile is full inside the c matrix, we store with flattened intermediate layout
            intermediate_layout = access_efficient_layout(original_layout, element_nbytes=original_x.dtype.nbytes)
            intermediate_x = vb.view(original_x, layout=intermediate_layout)

            with vb.if_then(axis > 0):
                self.wait_semaphore(vb, value=axis)
                other_intermediate = vb.load_global(
                    dtype=dtype,
                    layout=intermediate_layout,
                    ptr=self.intermediate,
                    f_offset=lambda axes: get_block_offset() + axes[0],
                    f_mask=lambda axes: boolean.true,
                )
                vb.add(intermediate_x, other_intermediate, out=intermediate_x)

            with vb.if_then(axis + 1 < self.num_block_reduce_tiles):
                vb.store_global(
                    intermediate_x,
                    ptr=self.intermediate,
                    f_offset=lambda axes: get_block_offset() + axes[0],
                    f_mask=lambda axes: boolean.true,
                )
                self.release_semaphore(vb, value=axis + 1)
                vb.exit()
            with vb.otherwise():
                self.release_semaphore(vb, value=int32.zero)

        with vb.otherwise():
            intermediate_x = original_x
            intermediate_layout = original_layout

            def f_offset(axes):
                block_offset = get_block_offset()
                return block_offset + index_serialize(axes, self.tiled_shape)

            def f_mask(axes):
                indices = index_add(self.block_offsets, axes)
                return logical_and(*[index < size for index, size in zip(indices, self.op.output.shape)])

            # when the tile is not full inside the c matrix, we store with original layout
            with vb.if_then(axis > 0):
                self.wait_semaphore(vb, value=axis)
                other_intermediate = vb.load_global(
                    dtype=dtype,
                    layout=intermediate_layout,
                    ptr=self.intermediate,
                    f_offset=lambda axes: f_offset(axes),
                    f_mask=lambda axes: f_mask(axes),
                )
                vb.add(intermediate_x, other_intermediate, out=intermediate_x)

            with vb.if_then(axis + 1 < self.num_block_reduce_tiles):
                vb.store_global(
                    intermediate_x,
                    ptr=self.intermediate,
                    f_offset=lambda axes: f_offset(axes),
                    f_mask=lambda axes: f_mask(axes),
                )
                self.release_semaphore(vb, value=axis + 1)
                vb.exit()
            with vb.otherwise():
                self.release_semaphore(vb, value=int32.zero)

        return vb.finish()
