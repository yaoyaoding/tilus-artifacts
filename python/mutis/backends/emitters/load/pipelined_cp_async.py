from __future__ import annotations

from typing import Optional, Union, Dict, Any, List, Type, Tuple, cast
import functools

from hidet.ir.dtypes import int32, boolean
from hidet.ir.expr import Var, Expr, Constant, logical_and, index_vars
from hidet.ir.type import DataType
from hidet.ir.utils.index_transform import index_multiply, index_sum, index_add, index_deserialize
from hidet.ir.utils.index_transform import index_serialize, index_divide
from hidet.ir.tools import rewrite
from mutis.backends.codegen import register_emitter, GroupReduceEmitter, ReduceEmitter, NotSupportedEmitter
from mutis.ir.layout import Layout, squeeze, simplify, divide
from mutis.ir.tile import TensorTile
from mutis.ops.ldst import Load
from mutis.vm.ir.inst import AllocateSharedInst, FreeSharedInst, ViewSharedInst
from mutis.vm.ir.inst import Instruction, ViewInst, CopyAsyncCommitGroupInst, CopyAsyncWaitGroupInst
from mutis.vm.ir.inst import SyncThreadsInst, AllocateScalarInst, AssignScalarInst, CopyAsyncInst, LoadMatrixInst
from mutis.vm.ir.inst import LoadSharedInst
from mutis.vm.ir.stmt import Stmt, SeqStmt, ForStmt
from mutis.vm.ir.value import SharedLayout, SharedValue
from mutis.vm.ir.builder import VirtualMachineBuilder
from mutis.utils import prod, cdiv, idiv
from mutis.target import nvgpu_sm80
from mutis.backends.emitters.load.transformed import TransformLoadBaseEmitter, WeightTransform


class PipelinedCopyAsyncLoadGroupReduceEmitter(GroupReduceEmitter):
    """

    emit the following code

    # reduce prologue
    ... # allocate shared memory
    for stage in range(num_stages - 1):
        ... # preload for `stage`
        CopyAsyncCommitGroup()
    CopyAsyncWaitGroup(remain=num_stages - 2)
    SyncThreads()

    current_stage = 0
    preload_stage = num_stages - 1
    for k in range(num_tiles):
        # emit load inst
        ...

        # reduce body epilogue
        ... # preload for `preload_stage`
        CopyAsyncCommitGroup()
        CopyAsyncWaitGroup(remain=num_stage - 1)
        SyncThreads()

    # reduce epilogue
    ...  # free shared memory

    """

    def __init__(self, emitters: List[PipelinedCopyAsyncLoadEmitter]):
        super().__init__(emitters)
        self.emitters: List[PipelinedCopyAsyncLoadEmitter] = emitters

        self.num_stages: Optional[int] = None
        self.current_stage: Optional[Var] = None
        self.preload_stage: Optional[Var] = None

    def emit_reduce_prologue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        seq = []

        assert all(emitter.num_stages == self.emitters[0].num_stages for emitter in self.emitters)
        self.num_stages = self.emitters[0].num_stages

        # allocate shared memory
        for emitter in self.emitters:
            seq.append(emitter.allocate_shared_memory(axis))

        # preload the first `num_stages - 1` stages
        stage = Var('stage', int32)
        preload_seq = []
        for emitter in self.emitters:
            preload_seq.append(emitter.preload(k_iter=stage, stage=stage))
        preload_seq.append(CopyAsyncCommitGroupInst())
        seq.append(
            ForStmt(iter_var=stage, extent=int32(self.num_stages - 1), body=SeqStmt(preload_seq), unroll_factor=-1)
        )

        # wait and sync
        seq.append(CopyAsyncWaitGroupInst(n=int32(self.num_stages - 2)))
        seq.append(SyncThreadsInst())

        # define index vars
        alloc_current_stage_inst = AllocateScalarInst.create('current_stage', int32, init=int32(0))
        alloc_preload_stage_inst = AllocateScalarInst.create('preload_stage', int32, init=int32(self.num_stages - 1))
        seq.append(alloc_current_stage_inst)
        seq.append(alloc_preload_stage_inst)
        self.current_stage = alloc_current_stage_inst.var
        self.preload_stage = alloc_preload_stage_inst.var
        return SeqStmt(seq)

    def emit_reduce_body_prologue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        for emitter in self.emitters:
            emitter.current_stage = self.current_stage
        return None

    def emit_reduce_body_epilogue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        seq = []

        # preload the next stage
        for emitter in self.emitters:
            seq.append(emitter.preload(k_iter=axis + (self.num_stages - 1), stage=self.preload_stage))
        seq.append(CopyAsyncCommitGroupInst())
        seq.append(CopyAsyncWaitGroupInst(n=int32(self.num_stages - 2)))
        seq.append(SyncThreadsInst())

        # update current stage and preload stage
        seq.append(AssignScalarInst(self.current_stage, (self.current_stage + 1) % self.num_stages))
        seq.append(AssignScalarInst(self.preload_stage, (self.preload_stage + 1) % self.num_stages))
        return SeqStmt(seq)

    def emit_reduce_epilogue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        seq = []
        for emitter in self.emitters:
            seq.append(emitter.free_shared_memory())
        return SeqStmt(seq)


class PipelinedCopyAsyncLoadEmitter(ReduceEmitter):
    def __init__(self, codegen, op: Load, variant: Dict[str, Any]):
        super().__init__(codegen, op, variant)
        self.op: Load = op
        self.dtype: DataType = op.dtype
        self.num_stages: int = int(variant['pipeline'])
        self.tile: TensorTile = self.tensor2tile[op.output]
        reduce_axes = [
            axis for axis in self.tile.linear_tile_axes if self.codegen.schedule.graph_tile.axis_kind[axis] == 'reduce'
        ]
        if len(reduce_axes) != 1:
            raise NotSupportedEmitter()
        self.k_axis: Var = reduce_axes[0]

        self.current_stage: Optional[Expr] = None  # will be set before emit(...) was called

    @staticmethod
    def emitter_group_class() -> Optional[Type[GroupReduceEmitter]]:
        return PipelinedCopyAsyncLoadGroupReduceEmitter

    def allocate_shared_memory(self, axis: Var) -> Union[Stmt, Instruction]:
        raise NotImplementedError()

    def preload(self, k_iter: Expr, stage: Expr) -> Union[Stmt, Instruction]:
        raise NotImplementedError()

    def free_shared_memory(self) -> Union[Stmt, Instruction]:
        raise NotImplementedError()


@register_emitter(
    Load, priority=5, variant={'stages': 'gmem->smem->regs'}, target=nvgpu_sm80  # depends on cp.async in sm80 nv gpus
)
class PipelinedCopyAsyncLoadMatrixEmitter(PipelinedCopyAsyncLoadEmitter):
    def __init__(self, codegen, op: Load, variant: Dict[str, Any]):
        super().__init__(codegen, op, variant)
        # analyze results
        self.shared_shape: Optional[List[int]] = None
        self.shared_layout: Optional[SharedLayout] = None
        self.ldmatrix_layout: Optional[Layout] = None

        if self.dtype.nbits < 8:
            raise NotSupportedEmitter()

        self.determine_shared_layout()
        self.determine_ldmatrix_layouts()
        self.check_cp_async_validity()

        # emitted code related
        self.current_stage: Optional[Expr] = None
        self.shared_buf: Optional[SharedValue] = None

    def determine_ldmatrix_layouts(self):
        # normalize the layout into 2 dims
        layout: Layout = self.tensor2layout[self.op.output]

        if len(layout.shape) < 2:
            raise NotSupportedEmitter()
        if len(layout.shape) > 2:
            if any(s != 1 for s in layout.shape[:-2]):
                raise NotSupportedEmitter()
            layout = simplify(squeeze(layout, dims=list(range(len(layout.shape) - 2))))
        self.ldmatrix_layout = layout

        # check whether the layout can be loaded via ldmatrix
        for nbytes, trans, atom_layout in LoadMatrixInst.ldmatrix_configs:
            if nbytes != self.dtype.nbytes:
                continue
            outer: Optional[Layout] = divide(layout, atom_layout)
            if outer is not None:
                return
        raise NotSupportedEmitter()

    def determine_shared_layout(self):
        """
        repeat(rows, cols).swizzle(dim=1, log_step=*)  (all position are mod by 8)
        0   0 1   0 1 2 3   0 1 2 3 4 5 6 7
        1   2 3   4 5 6 7   1 0 3 2 5 4 7 6
        2   4 5   1 0 3 2   2 3 0 1 6 7 4 5
        3   6 7   5 4 7 6   3 2 1 0 7 6 5 4
        4   1 0   2 3 0 1   4 5 6 7 0 1 2 3
        5   3 2   6 7 4 5   5 4 7 6 1 0 3 2
        6   5 4   3 2 1 0   6 7 4 5 2 3 0 1
        7   7 6   7 6 5 4   7 6 5 4 3 2 1 0

        each above number is a vector with 16 bytes
        """
        tiled_shape: List[int] = self.tile.tiled_shape(stop_axis=self.k_axis, include_stop_axis=True)
        assert all(s == 1 for s in tiled_shape[:-2])
        rows, columns = tiled_shape[-2:]
        if 16 * 8 % self.dtype.nbits != 0:
            raise NotSupportedEmitter()
        vec_elements = 16 * 8 // self.dtype.nbits
        columns = idiv(columns, vec_elements)

        assert rows % 8 == 0

        srepeat = SharedLayout.repeat
        scompose = SharedLayout.compose

        if columns % 8 == 0:
            # most efficient for cp.async
            self.shared_layout = srepeat(rows, columns).swizzle(dim=1, regards_dim=0, log_step=0)
        elif columns % 4 == 0:
            self.shared_layout = scompose(
                srepeat(1, columns // 4), srepeat(rows, 4).swizzle(dim=1, regards_dim=0, log_step=1)
            )
        elif columns % 2 == 0:
            self.shared_layout = scompose(
                srepeat(1, columns // 2), srepeat(rows, 2).swizzle(dim=1, regards_dim=0, log_step=2)
            )
        else:
            # most not efficient for cp.async
            self.shared_layout = scompose(
                srepeat(1, columns), srepeat(rows, 1).swizzle(dim=1, regards_dim=0, log_step=3)
            )
        self.shared_layout = scompose(self.shared_layout, srepeat(1, vec_elements)).simplify()
        self.shared_shape = self.shared_layout.shape

    def check_cp_async_validity(self):
        from hidet.transforms.rule_based_simplifier import RuleBasedSimplifier, BoundInfo
        from mutis.ir.analyzers.value_analyzer import analyze_info, TensorInfo

        # get shared info
        shared_info: TensorInfo = analyze_info(
            shape=self.shared_layout.shape, axes=self.shared_layout.axes, var2info={}, expr=self.shared_layout.offset
        )

        # get global info
        tile: TensorTile = self.tensor2tile[self.op.output]
        tile_offsets: List[Expr] = tile.tile_offsets(stop_axis=self.k_axis, include_stop_axis=True)
        cp_async_axes: List[Var] = index_vars(2)
        intra_offsets: List[Expr] = [int32.zero for i in range(len(tile_offsets) - 2)] + cp_async_axes
        global_offsets: List[Expr] = index_add(tile_offsets, intra_offsets)
        global_offset: Expr = index_sum(index_multiply(global_offsets, self.op.strides))
        var2info: Dict[Var, TensorInfo] = {}
        var2bound: Dict[Var, BoundInfo] = {}
        for param, attr in self.codegen.graph.param2attrs.items():
            if attr.divisibility is not None:
                var2info[param] = TensorInfo.from_divisiblity(shape=self.shared_shape, divisibility=attr.divisibility)
            var2bound[param] = BoundInfo(min_value=attr.lower, max_value=attr.upper)
        for tile_axis, num_tiles in self.codegen.graph_tile.num_tiles_map.items():
            if isinstance(num_tiles, Constant):
                var2bound[tile_axis] = BoundInfo(min_value=0, max_value=int(num_tiles) - 1)

        global_offset = RuleBasedSimplifier(var2bound)(global_offset)
        global_info: TensorInfo = analyze_info(
            shape=self.shared_shape, axes=cp_async_axes, var2info=var2info, expr=global_offset
        )

        # get mask info
        mask: Expr = logical_and(global_offsets[0] < self.op.shape[0], global_offsets[1] < self.op.shape[1])
        mask = RuleBasedSimplifier(var2bound)(mask)
        mask_info: TensorInfo = analyze_info(shape=self.shared_shape, axes=cp_async_axes, var2info=var2info, expr=mask)

        nbytes = 4
        conditions = [
            shared_info.infos[1].continuity * self.dtype.nbytes % nbytes == 0,
            shared_info.infos[1].divisibility * self.dtype.nbytes % nbytes == 0,
            global_info.infos[1].continuity * self.dtype.nbytes % nbytes == 0,
            global_info.infos[1].divisibility * self.dtype.nbytes % nbytes == 0,
            mask_info.infos[1].constancy * self.dtype.nbytes % nbytes == 0,
        ]
        if not all(conditions):
            raise NotSupportedEmitter()

    def allocate_shared_memory(self, axis: Var) -> Union[Stmt, Instruction]:
        inst = AllocateSharedInst.create(
            dtype=self.dtype, shared_layout=self.shared_layout.prepend_dim(extent=self.num_stages)
        )
        self.shared_buf = inst.output
        return inst

    def preload(self, k_iter: Expr, stage: Expr) -> Union[Stmt, Instruction]:
        from hidet.ir.tools import rewrite

        tile: TensorTile = self.tensor2tile[self.op.output]
        tile_offsets: List[Expr] = tile.tile_offsets(stop_axis=self.k_axis, include_stop_axis=True)

        def f_offset(axes: List[Var]) -> Expr:
            intra_offsets: List[Expr] = [int32.zero for i in range(len(tile_offsets) - 2)] + axes
            global_offsets: List[Expr] = index_add(tile_offsets, intra_offsets)
            global_offset: Expr = index_sum(index_multiply(global_offsets, self.op.strides))
            global_offset = rewrite(global_offset, {self.k_axis: k_iter})
            return global_offset

        def f_mask(axes: List[Var]) -> Expr:
            intra_offsets: List[Expr] = [int32.zero for i in range(len(tile_offsets) - 2)] + axes
            global_offsets: List[Expr] = index_add(tile_offsets, intra_offsets)
            mask: Expr = logical_and(
                *[global_offset < extent for global_offset, extent in zip(global_offsets, self.op.shape)]
            )
            mask = rewrite(mask, {self.k_axis: k_iter})
            return mask

        view_inst = ViewSharedInst.create(x=self.shared_buf, indices=[stage], layout=self.shared_layout)
        copy_inst = CopyAsyncInst.create(dst=view_inst.output, ptr=self.op.ptr, f_offset=f_offset, f_mask=f_mask)
        return SeqStmt([view_inst, copy_inst])

    def emit(self) -> Union[Stmt, Instruction]:
        vb = VirtualMachineBuilder()

        x = vb.view_shared(x=self.shared_buf, indices=[self.current_stage], layout=self.shared_layout)
        offsets = self.tile.tile_offsets(start_axis=self.k_axis, include_start_axis=False)
        x = vb.load_matrix(src=x, register_layout=self.ldmatrix_layout, offsets=offsets[-2:])
        if len(offsets) > 2:
            x = vb.view(x=x, layout=self.tensor2layout[self.op.output])
        if self.op.cast_dtype and self.op.cast_dtype != self.op.dtype:
            x = vb.cast(x=x, dtype=self.op.cast_dtype)

        self.tensor2value[self.op.output] = x

        return vb.finish()

        # seq = []
        # view_shared_inst = ViewSharedInst.create(
        #     x=self.shared_buf, indices=[self.current_stage], layout=self.shared_layout
        # )
        # seq.append(view_shared_inst)
        #
        # load_inst = LoadMatrixInst.create(
        #     src=view_shared_inst.output,
        #     register_layout=self.ldmatrix_layout,
        #     offsets=offsets[-2:],  # ignore prefix zeros
        # )
        # seq.append(load_inst)
        # self.tensor2value[self.op.output] = load_inst.output
        #
        # if len(offsets) > 2:
        #     view_regs_inst = ViewInst.create(
        #         x=load_inst.output.as_register_value(), layout=self.tensor2layout[self.op.output]
        #     )
        #     seq.append(view_regs_inst)
        #     self.tensor2value[self.op.output] = view_regs_inst.output
        #
        # return SeqStmt(seq)

    def free_shared_memory(self) -> Union[Stmt, Instruction]:
        return FreeSharedInst(self.shared_buf)


@register_emitter(
    Load, priority=4, variant={'stages': 'gmem->smem->regs'}, target=nvgpu_sm80  # depends on cp.async in sm80 nv gpus
)
class PipelinedTransformedCopyAsyncLdsLoadEmitter(PipelinedCopyAsyncLoadEmitter, TransformLoadBaseEmitter):
    def __init__(self, codegen, op: Load, variant: Dict[str, Any]):
        super().__init__(codegen, op, variant)

        self.shared_original_shape: List[int] = self.tile.tiled_shape(stop_axis=self.k_axis, include_stop_axis=True)
        self.shared_layout = SharedLayout.repeat(
            prod(index_divide(self.shared_original_shape, self.original_layout.shape))
            * self.transformed_lhs_layout.shape[0]
            * self.transformed_rhs_layout.shape[0]
        )
        self.shared_buf: Optional[SharedValue] = None

        self.check()

    def check(self):
        # check whether we can use cp.async to load the data, there is only one condition that the cp.async instruction
        # can load at least 4 bytes by each thread (candidates: 4, 8, 16)
        pass

    def allocate_shared_memory(self, axis: Var) -> Union[Stmt, Instruction]:
        # we used transformed_dtype as the storage data type of shared buffer
        inst = AllocateSharedInst.create(
            dtype=self.transformed_dtype, shared_layout=self.shared_layout.prepend_dim(extent=self.num_stages)
        )
        self.shared_buf = inst.output
        return inst

    def preload(self, k_iter: Expr, stage: Expr) -> Union[Stmt, Instruction]:
        load = cast(Load, self.op)

        def f_outer_tile_indices_and_axes(axes: List[Var]) -> Tuple[List[Expr], List[Expr]]:
            shared_tile_indices = self.tensor2tile[load.output].tile_indices(
                stop_axis=self.k_axis, include_stop_axis=True
            )
            shared_tile_indices = rewrite(shared_tile_indices, {self.k_axis: k_iter})
            shared_tile_offsets = index_multiply(
                shared_tile_indices, index_divide(self.shared_original_shape, self.original_layout.shape)
            )
            intra_shared_tile_indices = index_deserialize(
                axes[0] // self.transformed_layout.shape[0],
                index_divide(self.shared_original_shape, self.original_layout.shape),
            )
            outer_tile_indices = index_add(shared_tile_offsets, intra_shared_tile_indices)
            axes = [axes[0] % self.transformed_layout.shape[0]]
            return outer_tile_indices, axes

        def f_ptr(axes: List[Var]) -> Expr:
            outer_tile_indices, axes = f_outer_tile_indices_and_axes(axes)
            return self.f_ptr(outer_tile_indices, axes)

        def f_mask(axes: List[Var]) -> Union[Expr, bool]:
            outer_tile_indices, axes = f_outer_tile_indices_and_axes(axes)
            return self.f_mask(outer_tile_indices, axes)

        vb = VirtualMachineBuilder()
        viewed = vb.view_shared(self.shared_buf, indices=[stage], layout=self.shared_layout)
        vb.copy_async(dst=viewed, ptr=load.ptr, f_offset=f_ptr, f_mask=f_mask)
        return vb.finish()

    def emit(self) -> Union[Stmt, Instruction]:
        vb = VirtualMachineBuilder()
        x = vb.view_shared(x=self.shared_buf, indices=[self.current_stage], layout=self.shared_layout)
        x = vb.load_shared(
            src=x,
            register_layout=self.transformed_layout,
            offsets=[
                index_serialize(
                    self.tile.tile_indices(start_axis=self.k_axis, include_start_axis=False),
                    index_divide(self.shared_original_shape, self.original_layout.shape),
                )
                * prod(self.transformed_layout.shape)
            ],
        )
        x = vb.view(x=x, dtype=self.dtype, layout=self.original_layout)
        if self.op.cast_dtype and self.op.cast_dtype != self.op.dtype:
            x = vb.cast(x=x, dtype=self.op.cast_dtype)
        self.tensor2value[self.op.output] = x
        return vb.finish()

    def free_shared_memory(self) -> Union[Stmt, Instruction]:
        return FreeSharedInst(self.shared_buf)
