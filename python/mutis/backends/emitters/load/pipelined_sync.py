from __future__ import annotations

from typing import Optional, Union, Dict, Any, List, Type, Tuple, cast
import functools

from hidet.ir.dtypes import int32, boolean
from hidet.ir.expr import Var, Expr, Constant, logical_and, index_vars
from hidet.ir.type import DataType
from hidet.ir.mapping import TaskMapping
from hidet.ir.utils.index_transform import index_multiply, index_sum, index_add, index_deserialize
from hidet.ir.utils.index_transform import index_serialize, index_divide
from hidet.ir.tools import rewrite
from mutis.backends.codegen import (
    register_emitter,
    GroupReduceEmitter,
    ReduceEmitter,
    GroupUnrollEmitter,
    UnrollEmitter,
    NotSupportedEmitter,
)
from mutis.ir.layout import Layout, squeeze, repeat, expand, simplify, divide, auto_repeat_spatial
from mutis.ir.tile import TensorTile
from mutis.ops.ldst import Load
from mutis.vm.ir.inst import AllocateSharedInst, FreeSharedInst, ViewSharedInst
from mutis.vm.ir.inst import Instruction, ViewInst, CopyAsyncCommitGroupInst, CopyAsyncWaitGroupInst
from mutis.vm.ir.inst import SyncThreadsInst, AllocateScalarInst, AssignScalarInst, CopyAsyncInst, LoadMatrixInst
from mutis.vm.ir.inst import LoadSharedInst
from mutis.vm.ir.stmt import Stmt, SeqStmt, ForStmt
from mutis.vm.ir.value import SharedLayout, SharedValue, RegisterValue
from mutis.vm.ir.builder import VirtualMachineBuilder
from mutis.utils import prod, cdiv, idiv, gcd
from mutis.target import gpgpu_any
from mutis.vm.ir.shared_layout import shared_repeat, shared_compose, shared_column_repeat
from mutis.backends.emitters.load.transformed import TransformLoadBaseEmitter, WeightTransform


class PipelinedSyncLoadGroupReduceEmitter(GroupReduceEmitter, GroupUnrollEmitter):
    """

    emit the following code

    # reduce prologue
    ... # allocate shared memory
    ... # allocate register memory
    for stage in range(num_stages - 1):
        ... # preload_global for `stage`
        ... # store_shared for `stage`
    SyncThreads()

    current_stage = 0
    preload_stage = num_stages - 1
    for k in range(num_tiles):
        /* reduce body prologue */
        ...  # preload_global for `preload_stage`

        for regs_stage in range(num_regs_stages - 1):
            ... # preload_shared for `regs_stage`

        current_regs_stage = 0
        preload_regs_stage = num_regs_stages - 1

        for u in range(num_unrolls):
            ...  # preload_shared for `preload_regs_stage` with unroll loop `u + num_regs_stages - 1`

            ...  # using `current_regs_stage` of register memory

            current_regs_stage = (current_regs_stage + 1) % num_regs_stages
            preload_regs_stage = (preload_regs_stage + 1) % num_regs_stages

        /* reduce body epilogue */
        ...  # store_shared for `preload_stage` (perform the optional cast here)

        preload_stage = (preload_stage + 1) % num_stages
        current_stage = (current_stage + 1) % num_stages
        SyncThreads()

    /* reduce epilogue */
    ...  # free shared memory

    """

    def __init__(self, emitters: List[PipelinedSyncLoadEmitter]):
        super().__init__(emitters)
        self.emitters: List[PipelinedSyncLoadEmitter] = emitters

        self.num_stages: Optional[int] = None
        self.num_regs_stages: Optional[int] = None

        self.current_stage: Optional[Var] = None
        self.preload_stage: Optional[Var] = None

        self.current_regs_stage: Optional[Var] = None
        self.preload_regs_stage: Optional[Var] = None

    def init(self):
        assert all(emitter.num_stages == self.emitters[0].num_stages for emitter in self.emitters)
        self.num_stages: int = self.emitters[0].num_stages

        assert all(emitter.num_regs_stages == self.emitters[0].num_regs_stages for emitter in self.emitters)
        self.num_regs_stages: int = self.emitters[0].num_regs_stages

    def emit_reduce_prologue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        vb = VirtualMachineBuilder()

        # allocate shared memory
        for emitter in self.emitters:
            vb.append(emitter.allocate_loading_register_memory(axis))
        for emitter in self.emitters:
            vb.append(emitter.allocate_shared_memory(axis))

        # preload the first `num_stages - 1` stages
        with vb.for_range(extent=self.num_stages - 1) as stage:
            for emitter in self.emitters:
                vb.append(emitter.preload_global(k_iter=stage, stage=stage))
                vb.append(emitter.store_shared(k_iter=stage, stage=stage))
        vb.syncthreads()

        # define index variables
        self.current_stage = vb.allocate_scalar('current_stage', int32, init=int32(0))
        self.preload_stage = vb.allocate_scalar('preload_stage', int32, init=int32(self.num_stages - 1))

        return vb.finish()

    def emit_reduce_body_prologue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        vb = VirtualMachineBuilder()

        # set the current_stage so that the `emit(...)` method can know the current stage
        for emitter in self.emitters:
            emitter.current_stage = self.current_stage

        # preload the next stage from global memory to register
        for emitter in self.emitters:
            vb.append(emitter.preload_global(k_iter=axis + (self.num_stages - 1), stage=self.preload_stage))

        return vb.finish()

    def emit_unroll_prologue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        vb = VirtualMachineBuilder()

        # allocate register memory
        for emitter in self.emitters:
            vb.append(emitter.allocate_register_memory(axis))

        # preload `num_regs_stages - 1` stages
        with vb.for_range(extent=self.num_regs_stages - 1) as regs_stage:
            for emitter in self.emitters:
                vb.append(emitter.preload_shared(u_iter=regs_stage, regs_stage=regs_stage))

        # define index variables
        self.current_regs_stage = vb.allocate_scalar('current_regs_stage', int32, init=int32(0))
        self.preload_regs_stage = vb.allocate_scalar('preload_regs_stage', int32, init=int32(self.num_regs_stages - 1))

        return vb.finish()

    def emit_unroll_body_prologue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        vb = VirtualMachineBuilder()

        # set the current_regs_stage so that the `emit(...)` method can know the current register stage
        for emitter in self.emitters:
            emitter.current_regs_stage = self.current_regs_stage

        # preload the next stage from shared memory to register
        for emitter in self.emitters:
            vb.append(
                emitter.preload_shared(u_iter=axis + (self.num_regs_stages - 1), regs_stage=self.preload_regs_stage)
            )

        return vb.finish()

    def emit_unroll_body_epilogue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        vb = VirtualMachineBuilder()

        # update stage index variables
        vb.assign_scalar(self.current_regs_stage, (self.current_regs_stage + 1) % self.num_regs_stages)
        vb.assign_scalar(self.preload_regs_stage, (self.preload_regs_stage + 1) % self.num_regs_stages)

        return vb.finish()

    def emit_unroll_epilogue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        pass  # do nothing

    def emit_reduce_body_epilogue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        vb = VirtualMachineBuilder()

        # store the register of the next stage to shared memory
        for emitter in self.emitters:
            vb.append(emitter.store_shared(k_iter=axis + (self.num_stages - 1), stage=self.preload_stage))

        # update stage index variables
        vb.assign_scalar(self.current_stage, (self.current_stage + 1) % self.num_stages)
        vb.assign_scalar(self.preload_stage, (self.preload_stage + 1) % self.num_stages)
        vb.syncthreads()

        return vb.finish()

    def emit_reduce_epilogue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        seq = []
        for emitter in self.emitters:
            seq.append(emitter.free_shared_memory())
        return SeqStmt(seq)


class PipelinedSyncLoadEmitter(ReduceEmitter, UnrollEmitter):
    def __init__(self, codegen, op: Load, variant: Dict[str, Any]):
        super().__init__(codegen, op, variant)
        self.op: Load = op
        self.num_stages: int = int(variant['pipeline'])
        self.num_regs_stages: int = int(variant.get('regs_stages', 2))
        self.tile: TensorTile = self.tensor2tile[op.output]
        reduce_axes = [
            axis for axis in self.tile.linear_tile_axes if self.codegen.schedule.graph_tile.axis_kind[axis] == 'reduce'
        ]
        unroll_axes = [
            axis for axis in self.tile.linear_tile_axes if self.codegen.schedule.graph_tile.axis_kind[axis] == 'unroll'
        ]
        if len(reduce_axes) != 1:
            raise NotSupportedEmitter()
        self.k_axis: Var = reduce_axes[0]
        self.u_axis: Var = unroll_axes[0]

        self.current_stage: Optional[Expr] = None  # will be set before emit(...) was called
        self.current_regs_stage: Optional[Expr] = None  # will be set before emit(...) was called

    @staticmethod
    def emitter_group_class() -> Optional[Type[GroupReduceEmitter]]:
        return PipelinedSyncLoadGroupReduceEmitter

    def allocate_loading_register_memory(self, axis: Var) -> Union[Stmt, Instruction]:
        raise NotImplementedError()

    def allocate_shared_memory(self, axis: Var) -> Union[Stmt, Instruction]:
        raise NotImplementedError()

    def allocate_register_memory(self, axis: Var) -> Union[Stmt, Instruction]:
        raise NotImplementedError()

    def preload_global(self, k_iter: Expr, stage: Expr) -> Union[Stmt, Instruction]:
        raise NotImplementedError()

    def store_shared(self, k_iter: Expr, stage: Expr) -> Union[Stmt, Instruction]:
        raise NotImplementedError()

    def preload_shared(self, u_iter: Expr, regs_stage: Expr) -> Union[Stmt, Instruction]:
        raise NotImplementedError()

    def free_shared_memory(self) -> Union[Stmt, Instruction]:
        raise NotImplementedError()


@register_emitter(
    Load, priority=2, variant={'stages': 'gmem->smem->regs'}, target=gpgpu_any  # depends on cp.async in sm80 nv gpus
)
class PipelinedSyncDefaultLoadEmitter(PipelinedSyncLoadEmitter):
    """
    1. determine shared memory layout
    2. determine gmem->smem loading task mapping
    """

    def __init__(self, codegen, op: Load, variant: Dict[str, Any]):
        super().__init__(codegen, op, variant)
        self.shared_layout: Optional[SharedLayout] = None
        self.g2s_regs_layout: Optional[Layout] = None
        self.regs_layout: Layout = self.tensor2layout[op.output]

        self.load_dtype: DataType = op.dtype
        self.output_dtype: DataType = op.cast_dtype if op.cast_dtype is not None else op.dtype

        self.determine_shared_layout()

        self.g2s_regs_value: Optional[RegisterValue] = None
        self.shared_value: Optional[SharedValue] = None
        self.regs_value: Optional[RegisterValue] = None

    def determine_shared_layout(self):
        if not self.variant['pipeline'] == 2:
            raise NotSupportedEmitter()

        if self.variant.get('shared_layout_hint', None):
            self.shared_layout = self.variant['shared_layout_hint']
            assert self.variant['g2s_layout_hint'] is not None
            self.g2s_regs_layout = self.variant['g2s_layout_hint']
            return

        shared_shape: List[int] = self.tile.tiled_shape(stop_axis=self.k_axis, include_stop_axis=True)
        self.shared_layout = shared_repeat(*shared_shape)
        self.g2s_regs_layout = auto_repeat_spatial(num_threads=self.codegen.schedule.num_warps * 32, shape=shared_shape)

    def allocate_loading_register_memory(self, axis: Var) -> Union[Stmt, Instruction]:
        vb = VirtualMachineBuilder()
        self.g2s_regs_value = vb.allocate(dtype=self.load_dtype, layout=self.g2s_regs_layout)
        return vb.finish()

    def allocate_shared_memory(self, axis: Var) -> Union[Stmt, Instruction]:
        vb = VirtualMachineBuilder()
        self.shared_value = vb.allocate_shared(
            dtype=self.output_dtype, shared_layout=self.shared_layout.prepend_dim(extent=2)
        )
        return vb.finish()

    def allocate_register_memory(self, axis: Var) -> Union[Stmt, Instruction]:
        vb = VirtualMachineBuilder()
        layout = repeat(self.num_regs_stages, *[1 for _ in self.regs_layout.shape]) * expand(self.regs_layout, dims=[0])
        self.regs_value = vb.allocate(dtype=self.output_dtype, layout=layout)
        return vb.finish()

    def preload_global(self, k_iter: Expr, stage: Expr) -> Union[Stmt, Instruction]:
        vb = VirtualMachineBuilder()

        tile: TensorTile = self.tensor2tile[self.op.output]
        tile_offsets: List[Expr] = tile.tile_offsets(stop_axis=self.k_axis, include_stop_axis=True)

        def f_offset(axes: List[Var]) -> Expr:
            intra_offsets: List[Expr] = [int32.zero for i in range(len(tile_offsets) - len(axes))] + axes
            global_offsets: List[Expr] = index_add(tile_offsets, intra_offsets)
            global_offset: Expr = index_sum(index_multiply(global_offsets, self.op.strides))
            global_offset = rewrite(global_offset, {self.k_axis: k_iter})
            return global_offset

        def f_mask(axes: List[Var]) -> Expr:
            intra_offsets: List[Expr] = [int32.zero for i in range(len(tile_offsets) - len(axes))] + axes
            global_offsets: List[Expr] = index_add(tile_offsets, intra_offsets)
            mask: Expr = logical_and(
                *[global_offset < extent for global_offset, extent in zip(global_offsets, self.op.shape)]
            )
            mask = rewrite(mask, {self.k_axis: k_iter})
            return mask

        vb.load_global(
            dtype=self.load_dtype,
            layout=self.g2s_regs_layout,
            ptr=self.op.ptr,
            f_offset=f_offset,
            f_mask=f_mask,
            out=self.g2s_regs_value,
        )

        return vb.finish()

    def preload_shared(self, u_iter: Expr, regs_stage: Expr) -> Union[Stmt, Instruction]:
        vb = VirtualMachineBuilder()
        tensor = self.op.output

        # slice the current stage shared buffer
        viewed_shared = vb.view_shared(x=self.shared_value, indices=[self.current_stage], layout=self.shared_layout)

        # slice the current stage register buffer to store to
        target_regs = vb.view(
            self.regs_value, layout=self.regs_layout, local_offset=regs_stage * self.regs_layout.local_size
        )

        # load from gmem to smem
        offsets = self.tensor2tile[tensor].tile_offsets(start_axis=self.k_axis, include_start_axis=False)
        offsets = rewrite(offsets, {self.u_axis: u_iter})
        vb.load_shared(
            src=viewed_shared, register_layout=self.tensor2layout[tensor], offsets=offsets, output=target_regs
        )

        return vb.finish()

    def emit(self) -> Union[Stmt, Instruction]:
        vb = VirtualMachineBuilder()
        self.tensor2value[self.op.output] = vb.view(
            self.regs_value, layout=self.regs_layout, local_offset=self.current_regs_stage * self.regs_layout.local_size
        )
        return vb.finish()

    def store_shared(self, k_iter: Expr, stage: Expr) -> Union[Stmt, Instruction]:
        vb = VirtualMachineBuilder()

        shared = vb.view_shared(x=self.shared_value, indices=[stage], layout=self.shared_layout)
        regs = self.g2s_regs_value
        if self.load_dtype != self.output_dtype:
            regs = vb.cast(regs, dtype=self.output_dtype)
        vb.store_shared(dst=shared, src=regs)

        return vb.finish()

    def free_shared_memory(self) -> Union[Stmt, Instruction]:
        vb = VirtualMachineBuilder()
        vb.free_shared(self.shared_value)
        return vb.finish()
