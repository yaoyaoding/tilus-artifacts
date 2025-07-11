from typing import List, Union, Optional, Dict, Callable, Sequence, Tuple
from contextlib import contextmanager
from hidet.ir.type import BaseType, DataType, PointerType
from hidet.ir.expr import Expr, Var, convert
from hidet.ir.dtypes import int32, boolean
from hidet.ir.tools import infer_type
from mutis.ir.graph import ParamAttrs
from mutis.ir.layout import Layout
from mutis.ir.schedule import BlockMapping, GraphTile
from mutis.vm.ir.program import VirtualMachineProgram, BlockMapping, WeightTransform
from mutis.vm.ir.stmt import Stmt, ForStmt, SeqStmt, ForThreadGroupStmt, IfStmt, WhileStmt, BreakStmt
from mutis.vm.ir.value import RegisterValue, SharedValue, SharedLayout
from mutis.vm.ir.inst import Instruction
from mutis.vm.ir.inst import AllocateInst, PrintValueInst, SyncThreadsInst, AllocateSharedInst, ViewInst
from mutis.vm.ir.inst import StoreSharedInst, LoadSharedInst, StoreGlobalInst, FreeSharedInst, LoadGlobalInst
from mutis.vm.ir.inst import LoadScalarInst, StoreScalarInst, ExitInst, ElementwiseBinaryInst, AllocateScalarInst
from mutis.vm.ir.inst import SyncReduceThreadsInst, AssignScalarInst, AssignInst, ViewSharedInst, CopyAsyncInst
from mutis.vm.ir.inst import FormatPrintInst, CastInst, LoadMatrixInst, MmaDotInst, CopyAsyncWaitAllInst
from mutis.vm.ir.inst import CopyAsyncCommitGroupInst, CopyAsyncWaitGroupInst, AllocateGlobalInst, AtomicScalarInst


class ProgramContext:
    def __init__(self, vb, name: str, num_warps: int, params: Union[Dict[str, BaseType], List[Var]]):
        self.vb: VirtualMachineBuilder = vb
        self.vb._name = name
        self.vb._num_warps = num_warps
        self.vb._params = [Var(name, type) for name, type in params.items()] if isinstance(params, dict) else params
        self.vb._block_axes = []
        self.vb._num_blocks = []
        self.vb._stack = [[]]

    def __enter__(self):
        return self.vb._params

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return

        assert len(self.vb._stack) == 1, len(self.vb._stack)
        graph_tile = GraphTile()
        graph_tile.block_axes = self.vb._block_axes
        graph_tile.num_tiles_map = {a: b for a, b in zip(self.vb._block_axes, self.vb._num_blocks)}
        self.vb.built_program = VirtualMachineProgram(
            name=self.vb._name,
            params=self.vb._params,
            param2attrs=self.vb._param2attrs,
            num_warps=self.vb._num_warps,
            block_axes=self.vb._block_axes,
            num_blocks=self.vb._num_blocks,
            body=SeqStmt(self.vb._stack.pop()),
            block_mapping=BlockMapping.default_mapping(graph_tile),
            weight_transforms=self.vb._weight_transforms,
            var2divisibility=self.vb._var2divisibility,
            annotations={},
        )


class BaseContext:
    def __init__(self, vb):
        self.vb: VirtualMachineBuilder = vb

    def enter(self):
        self.vb._stack.append([])

    def pop(self) -> Stmt:
        return SeqStmt(self.vb._stack.pop())

    def append(self, stmt):
        self.vb._stack[-1].append(stmt)

    @property
    def innermost_stack(self):
        return self.vb._stack[-1]


class ForContext(BaseContext):
    def __init__(
        self, vb, iter_vars: List[Var], extents: List[Expr], unrolls: List[Optional[int]], unwrap: bool = False
    ):
        super().__init__(vb)
        self.iter_vars: List[Var] = iter_vars
        self.extents: List[Expr] = extents
        self.unrolls: List[Optional[int]] = unrolls
        self.unwrap: bool = unwrap

    def __enter__(self):
        self.enter()
        if self.unwrap:
            assert len(self.iter_vars) == 1
            return self.iter_vars[0]
        else:
            return self.iter_vars

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return

        body = self.pop()

        for iter_var, extent, unroll in reversed(list(zip(self.iter_vars, self.extents, self.unrolls))):
            body = ForStmt(iter_var, extent, body, unroll)

        self.append(body)


class ForThreadGroupContext(BaseContext):
    def __init__(self, vb, iter_var: Var, num_groups: int):
        super().__init__(vb)
        self.iter_var: Var = iter_var
        self.num_groups: int = num_groups

    def __enter__(self):
        self.enter()
        return self.iter_var

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.append(ForThreadGroupStmt(self.iter_var, self.num_groups, body=self.pop()))


class IfContext(BaseContext):
    def __init__(self, vb, cond: Expr):
        super().__init__(vb)
        self.cond: Expr = cond

    def __enter__(self):
        self.enter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.append(IfStmt(self.cond, then_body=self.pop(), else_body=None))


class ElseIfContext(BaseContext):
    def __init__(self, vb, cond: Expr):
        super().__init__(vb)
        self.cond: Expr = cond

    def __enter__(self):
        self.enter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        body = self.pop()
        if_stmt = self.innermost_stack[-1]
        assert isinstance(if_stmt, IfStmt), 'with vb.else_if() must be used after with vb.if_then() or vb.else_if()'
        while if_stmt.else_body is not None:
            assert isinstance(
                if_stmt.else_body, IfStmt
            ), 'with vb.else_if() must be used after with vb.if_then() or vb.else_if()'
            if_stmt = if_stmt.else_body
        if_stmt.else_body = IfStmt(self.cond, then_body=body, else_body=None)


class OtherwiseContext(BaseContext):
    def __init__(self, vb):
        super().__init__(vb)

    def __enter__(self):
        self.enter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        else_body = self.pop()
        if_stmt = self.innermost_stack[-1]
        assert isinstance(if_stmt, IfStmt), 'with vb.otherwise() must be used after with vb.if_then() or vb.else_if()'
        while if_stmt.else_body is not None:
            assert isinstance(
                if_stmt.else_body, IfStmt
            ), 'with vb.otherwise() must be used after with vb.if_then() or vb.else_if()'
            if_stmt = if_stmt.else_body
        if_stmt.else_body = else_body


class WhileContext(BaseContext):
    def __init__(self, vb, cond: Expr):
        super().__init__(vb)
        self.cond: Expr = cond

    def __enter__(self):
        self.enter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.append(WhileStmt(self.cond, body=self.pop()))


class VirtualMachineBuilderCore:
    def __init__(self):
        self._name: Optional[str] = None
        self._num_warps: Optional[int] = None
        self._params: List[Var] = []
        self._block_axes: List[Var] = []
        self._num_blocks: List[Expr] = []
        self._weight_transforms: Dict[Var, List[WeightTransform]] = {}
        self._param2attrs: Dict[Var, ParamAttrs] = {}
        self._var2divisibility: Dict[Var, int] = {}

        # context stack
        self._stack: List[List[Union[Stmt, Instruction]]] = [[]]

        # built program
        self.built_program: Optional[VirtualMachineProgram] = None

    def __call__(self, *args):
        return self.built_program(*args)

    def __str__(self):
        return str(self.built_program)

    def _reset(self):
        self._name = None
        self._num_warps = None
        self._params = []
        self._block_axes = []
        self._num_blocks = []
        self._stack = [[]]
        self.built_program = None

    def program(self, name: str, num_warps: int, params: Union[Dict[str, BaseType], List[Var]]) -> ProgramContext:
        return ProgramContext(self, name, num_warps, params)

    def virtual_blocks(self, num_blocks: List[Union[Expr, int]]) -> List[Var]:
        num_blocks: List[Expr] = [int32(num_block) for num_block in num_blocks]
        block_axes: List[Var] = [Var(f"b{i}", int32) for i in range(len(num_blocks))]
        self._block_axes.extend(block_axes)
        self._num_blocks.extend(num_blocks)
        return block_axes

    def annotate_divisibility(self, var2divisibility: Dict[Var, int]):
        self._var2divisibility.update(var2divisibility)

    def append_weight_transform(self, param: Var, weight_transform: WeightTransform):
        if param not in self._params:
            raise ValueError(f'Parameter {param} is not defined')
        if param not in self._weight_transforms:
            self._weight_transforms[param] = []
        self._weight_transforms[param].append(weight_transform)

    def set_weight_nbytes(self, param: Var, nbytes: int):
        if param not in self._params:
            raise ValueError(f'Parameter {param} is not defined')
        if param not in self._param2attrs:
            self._param2attrs[param] = ParamAttrs()
        self._param2attrs[param].is_weight = True
        self._param2attrs[param].weight_nbytes = nbytes

    def for_range(
        self, extent: Union[Expr, int], iter_name_hint: str = 'i', unroll_factor: Optional[int] = None
    ) -> ForContext:
        iter_var = Var(iter_name_hint, type=int32)
        return ForContext(self, [iter_var], [convert(extent)], [unroll_factor], unwrap=True)

    def for_grid(self, extents: List[Union[Expr, int]], iter_name_hints: Optional[List[str]] = None) -> ForContext:
        extents = [convert(extent) for extent in extents]
        if iter_name_hints is None:
            names = 'ijkpqrstuvw'
            assert len(extents) < len(names)
            iter_name_hints = names[: len(extents)]
        iter_vars = [Var(name, type=int32) for name in iter_name_hints]
        return ForContext(self, iter_vars, extents, unrolls=[None] * len(extents))

    def for_thread_group(self, num_groups: int) -> ForThreadGroupContext:
        iter_var = Var('tg', type=int32)
        return ForThreadGroupContext(self, iter_var, num_groups)

    def while_loop(self, cond: Union[Expr, bool]):
        return WhileContext(self, cond)

    def if_then(self, cond: Union[Expr, bool]) -> IfContext:
        return IfContext(self, cond)

    def else_if(self, cond: Union[Expr, bool]) -> ElseIfContext:
        return ElseIfContext(self, cond)

    def otherwise(self) -> OtherwiseContext:
        return OtherwiseContext(self)

    def brk(self):
        stmt = BreakStmt()
        self._stack[-1].append(stmt)

    def append(self, inst: Union[Instruction, Stmt]):
        self._stack[-1].append(inst)

    def finish(self) -> Union[VirtualMachineProgram, Stmt]:
        if self.built_program:
            ret = self.built_program
        else:
            if len(self._stack) != 1:
                raise ValueError('Unbalanced context stack')
            if len(self._stack[0]) != 1:
                ret = SeqStmt(self._stack.pop())
            else:
                ret = self._stack.pop()[0]
        self._reset()
        return ret


class VirtualMachineBuilder(VirtualMachineBuilderCore):
    # register value operations
    def allocate(
        self, dtype: DataType, layout: Layout, f_init: Optional[Callable[[List[Var]], Union[Expr, float, int]]] = None
    ) -> RegisterValue:
        inst = AllocateInst.create(dtype, layout, f_init)
        self.append(inst)
        return inst.output.as_register_value()

    def allocate_scalar(self, hint: str, scalar_type: Union[DataType, PointerType], init: Optional[Expr] = None) -> Var:
        inst = AllocateScalarInst.create(hint=hint, scalar_type=scalar_type, init=init)
        self.append(inst)
        return inst.var

    def allocate_global(
        self, hint: str, scalar_type: BaseType, nbytes: Union[Expr, int], require_clean: bool = False
    ) -> Var:
        inst = AllocateGlobalInst.create(hint=hint, scalar_type=scalar_type, nbytes=nbytes, require_clean=require_clean)
        self.append(inst)
        return inst.var

    def assign(self, output: RegisterValue, x: RegisterValue):
        inst = AssignInst.create(output, x)
        self.append(inst)

    def assign_scalar(self, var: Var, scalar_expr: Expr):
        inst = AssignScalarInst.create(var=var, scalar_expr=scalar_expr)
        self.append(inst)

    def view(
        self,
        x: RegisterValue,
        *,
        layout: Optional[Layout] = None,
        dtype: Optional[DataType] = None,
        local_offset: Union[Expr, int] = 0,
    ) -> RegisterValue:
        inst = ViewInst.create(x, layout=layout, dtype=dtype, local_offset=local_offset)
        self.append(inst)
        return inst.output.as_register_value()

    def cast(
        self,
        x: RegisterValue,
        *,
        dtype: DataType,
        interleave_width=None,
        interleave_stride=None,
        ignore_int4b_xor=False,
    ) -> RegisterValue:
        if x.dtype == dtype:
            return x
        inst = CastInst.create(
            dtype=dtype,
            x=x,
            interleave_width=interleave_width,
            interleave_stride=interleave_stride,
            ignore_int4b_xor=ignore_int4b_xor,
        )
        self.append(inst)
        return inst.output.as_register_value()

    def view_shared(
        self, x: SharedValue, *, indices: List[Expr], layout: SharedLayout, dtype: Optional[DataType] = None
    ) -> SharedValue:
        inst = ViewSharedInst.create(x, indices, layout, dtype)
        self.append(inst)
        return inst.output.as_shared_value()

    def copy_async(
        self,
        *,
        dst: SharedValue,
        ptr: Var,
        f_offset: Callable[[List[Var]], Expr],
        f_mask: Optional[Callable[[List[Var]], Expr]],
        evict: Optional[str] = None,
    ):
        inst = CopyAsyncInst.create(dst, ptr, f_offset, f_mask, evict=evict)
        self.append(inst)

    def copy_async_wait_all(self):
        inst = CopyAsyncWaitAllInst()
        self.append(inst)

    def copy_async_commit_group(self):
        inst = CopyAsyncCommitGroupInst()
        self.append(inst)

    def copy_async_wait_group(self, n: Union[Expr, int]):
        inst = CopyAsyncWaitGroupInst(n)
        self.append(inst)

    def elementwise_binary(
        self, x: RegisterValue, y: RegisterValue, op: str, *, out: Optional[RegisterValue] = None
    ) -> RegisterValue:
        inst = ElementwiseBinaryInst.create(x, y, op, output=out)
        self.append(inst)
        return inst.output

    def add(self, x: RegisterValue, y: RegisterValue, *, out: Optional[RegisterValue] = None) -> RegisterValue:
        return self.elementwise_binary(x, y, '+', out=out)

    def sub(self, x: RegisterValue, y: RegisterValue, *, out: Optional[RegisterValue] = None) -> RegisterValue:
        return self.elementwise_binary(x, y, '-', out=out)

    def mul(self, x: RegisterValue, y: RegisterValue, *, out: Optional[RegisterValue] = None) -> RegisterValue:
        return self.elementwise_binary(x, y, '*', out=out)

    def div(self, x: RegisterValue, y: RegisterValue, *, out: Optional[RegisterValue] = None) -> RegisterValue:
        return self.elementwise_binary(x, y, '/', out=out)

    def mod(self, x: RegisterValue, y: RegisterValue, *, out: Optional[RegisterValue] = None) -> RegisterValue:
        return self.elementwise_binary(x, y, '%', out=out)

    def print_value(
        self, msg: str, value: RegisterValue, fmt: Optional[str] = None, cond: Optional[Expr] = boolean.true
    ):
        inst = PrintValueInst(value, cond=cond, msg=msg, fmt=fmt)
        self.append(inst)

    def format_print(self, fstring: str, expressions: Sequence[Expr], cond: Optional[Expr] = None):
        if cond is None:
            cond = boolean.true
        inst = FormatPrintInst(cond=cond, fstring=fstring, expressions=expressions)
        self.append(inst)

    def printf(self, fstring: str, *expressions: Expr):
        self.format_print(fstring=fstring, expressions=expressions)

    def mma_dot(
        self,
        a: RegisterValue,
        b: RegisterValue,
        c: RegisterValue,
        mma_inst: str,
        warp_spatial: Tuple[int, int, int],
        warp_repeat: Tuple[int, int, int],
        output: Optional[RegisterValue] = None,
    ):
        inst = MmaDotInst.create(
            a=a, b=b, c=c, mma_inst=mma_inst, warp_spatial=warp_spatial, warp_repeat=warp_repeat, output=output
        )
        self.append(inst)
        return inst.output.as_register_value()

    # shared value operations

    def allocate_shared(
        self, dtype: DataType, shared_layout: SharedLayout, f_init: Optional[Callable[[List[Var]], Expr]] = None
    ) -> SharedValue:
        inst = AllocateSharedInst.create(dtype=dtype, shared_layout=shared_layout, f_init=f_init)
        self.append(inst)
        return inst.output.as_shared_value()

    def free_shared(self, shared_value: SharedValue):
        inst = FreeSharedInst.create(shared_value)
        self.append(inst)

    def store_shared(self, dst: SharedValue, src: RegisterValue, offsets: Optional[List[Expr]] = None):
        inst = StoreSharedInst.create(dst, src, offsets)
        self.append(inst)

    def load_shared(
        self,
        src: SharedValue,
        register_layout: Layout,
        offsets: Optional[List[Expr]] = None,
        output: Optional[RegisterValue] = None,
    ) -> RegisterValue:
        if offsets is None:
            offsets = [int32.zero for _ in range(len(src.shape))]
        inst = LoadSharedInst.create(src=src, register_layout=register_layout, offsets=offsets, output=output)
        self.append(inst)
        return inst.output.as_register_value()

    def load_matrix(
        self,
        src: SharedValue,
        *,
        register_layout: Layout,
        offsets: Optional[List[Expr]] = None,
        output: Optional[RegisterValue] = None,
    ):
        inst = LoadMatrixInst.create(src=src, register_layout=register_layout, offsets=offsets, output=output)
        self.append(inst)
        return inst.output.as_register_value()

    # global memory operations
    def store_global(
        self,
        x: RegisterValue,
        *,
        ptr: Var,
        f_offset: Callable[[List[Var]], Expr],
        f_mask: Optional[Callable[[List[Var]], Expr]] = None,
    ):
        inst = StoreGlobalInst.create(x=x, ptr=ptr, f_offset=f_offset, f_mask=f_mask)
        self.append(inst)

    def load_global(
        self,
        *,
        dtype: DataType,
        layout: Layout,
        ptr: Var,
        f_offset: Callable[[List[Var]], Expr],
        f_mask: Optional[Callable[[List[Var]], Expr]] = None,
        out: Optional[RegisterValue] = None,
    ) -> RegisterValue:
        inst = LoadGlobalInst.create(dtype=dtype, layout=layout, ptr=ptr, f_offset=f_offset, f_mask=f_mask, out=out)
        self.append(inst)
        return inst.output.as_register_value()

    def load_scalar(self, ptr: Expr, sync: str = 'weak') -> Var:
        inst = LoadScalarInst.create(ptr, sync)
        self.append(inst)
        return inst.var

    def store_scalar(self, ptr: Expr, value: Expr, sync: str = 'weak'):
        inst = StoreScalarInst.create(ptr, value, sync)
        self.append(inst)

    def atomic_scalar(self, ptr: Expr, op: str, value: Expr):
        inst = AtomicScalarInst.create(ptr=ptr, op=op, value=value)
        self.append(inst)

    # control operations

    def syncthreads(self):
        inst = SyncThreadsInst()
        self.append(inst)

    def syncthreads_and(self, cond: Union[Expr, bool]):
        inst = SyncReduceThreadsInst.create(reduce_op='and', var_hint='sync_and', reduce_value=cond)
        self.append(inst)
        return inst.var

    def syncthreads_or(self, cond: Union[Expr, bool]):
        inst = SyncReduceThreadsInst.create(reduce_op='or', var_hint='sync_and', reduce_value=cond)
        self.append(inst)
        return inst.var

    def exit(self):
        inst = ExitInst()
        self.append(inst)
