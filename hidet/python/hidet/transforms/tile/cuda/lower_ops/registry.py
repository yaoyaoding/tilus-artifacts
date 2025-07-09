import inspect
from typing import List, Dict, Type, Union, Callable, Tuple, Optional, Sequence

from hidet.ir.builders import StmtBuilder
from hidet.ir.expr import Expr, Var, convert, cast
from hidet.ir.primitives.cuda import syncthreads, threadIdx
from hidet.ir.stmt import Stmt
from hidet.ir.tile.expr import TileOp
from hidet.ir.tile.layout import TileLayout, BlockLayout, FlattenBlockLayout, repeat
from hidet.ir.tile.type import TileScope
from hidet.ir.type import DataType, PointerType, tensor_pointer_type
from hidet.transforms.tile import annotations
from hidet.transforms.tile.analyzers.value_analyzer import TensorInfo
from hidet.utils import same_list


class Buffer:
    def __init__(
        self,
        buf_var: Var,
        dtype: Union[PointerType, DataType],
        shape: List[int],
        scope: TileScope,
        local_shape: List[int],
        layout: TileLayout,
        info=None,
    ):
        self.var: Var = buf_var
        self.dtype: Union[PointerType, DataType] = dtype
        self.shape: List[int] = shape
        self.scope: TileScope = scope
        self.local_shape: List[int] = local_shape
        self.layout: TileLayout = layout
        self.info: TensorInfo = info

        if scope.is_shared():
            assert layout.num_workers() == 1
        assert same_list(layout.logical_shape(), shape)

    def __getitem__(self, item):
        raise RuntimeError('Please use "at_logical" or "at_local" to access the buffer.')

    def at_logical(self, indices: Sequence[Union[Expr, int]]) -> Expr:
        if len(indices) != len(self.shape):
            raise ValueError('Invalid {}-rank indices for shape {}: {}'.format(len(indices), len(self.shape), indices))
        indices = [convert(idx) for idx in indices]
        local_indices, _ = self.layout.logical2local(indices)
        return self.var[local_indices]

    def at_local(self, indices: Sequence[Union[Expr, int]]) -> Expr:
        if len(indices) != len(self.local_shape):
            raise ValueError(
                'Invalid {}-rank indices for shape {}: {}'.format(len(indices), len(self.local_shape), indices)
            )
        indices = [convert(idx) for idx in indices]
        return self.var[indices]

    @property
    def block_layout(self) -> BlockLayout:
        assert isinstance(self.layout, BlockLayout)
        return self.layout

    @property
    def flatten_block_layout(self) -> FlattenBlockLayout:
        assert isinstance(self.layout, FlattenBlockLayout)
        return self.layout

    def is_shared(self):
        return self.layout.num_workers() == 1

    def is_block(self):
        return isinstance(self.layout, BlockLayout)

    def is_flatten_block(self):
        return isinstance(self.layout, FlattenBlockLayout)


class TileOpImpl(StmtBuilder):
    def __init__(self, num_warps: int):
        super().__init__()
        self.num_warps: int = num_warps

    def make_shared_buffer(
        self,
        dtype: Union[DataType, PointerType],
        shape: List[int],
        hint: str,
        ptr: Expr,
        layout: Optional[TileLayout] = None,
    ) -> Buffer:
        if layout is None:
            layout = repeat(*shape)
        local_shape = layout.local_shape()

        assert same_list(shape, layout.logical_shape())

        buf_var = Var(hint=hint, type=tensor_pointer_type(dtype=dtype, shape=layout.local_shape()))
        self.declare(buf_var, init=cast(ptr, ~dtype))
        return Buffer(
            buf_var=buf_var, dtype=dtype, shape=shape, scope=TileScope.Shared, local_shape=local_shape, layout=layout
        )

    def buffer_store(self, buf: Union[Expr, Buffer], indices: Sequence[Union[Expr, int]], value: Expr):
        if isinstance(buf, Buffer):
            local_indices, _ = buf.layout.logical2local(list(indices))
            super().buffer_store(buf.var, indices=local_indices, value=value)
        else:
            super().buffer_store(buf, indices, value)

    def logical_store(self, buf: Buffer, indices: Sequence[Union[Expr, int]], value: Expr):
        local_indices, _ = buf.layout.logical2local(list(indices))
        super().buffer_store(buf.var, indices=local_indices, value=value)

    def local_store(self, buf: Buffer, indices: Sequence[Union[Expr, int]], value: Expr):
        super().buffer_store(buf.var, indices, value)

    def sync_threads(self):
        self.append(syncthreads())

    def iterate_dist_buffer_and_apply(self, buf: Buffer, f_apply: Callable[[List[Expr], List[Expr], Expr], None]):
        assert buf.scope == TileScope.Register

        layout: TileLayout = buf.layout
        local_shape: List[int] = layout.local_shape()

        with self.for_grid(local_shape) as local_indices:
            global_indices, not_duplicated = layout.local2logical(local_indices, worker=threadIdx.x)
            f_apply(local_indices, global_indices, not_duplicated)

    def iterate_dist_buffer_and_compute(self, buf: Buffer, f_compute: Callable[[List[Expr], List[Expr], Expr], Expr]):
        def f_apply(local_indices, global_indices, not_duplicated):
            value = f_compute(local_indices, global_indices, not_duplicated)
            self.buffer_store(buf.var, indices=local_indices, value=value)

        self.iterate_dist_buffer_and_apply(buf, f_apply)

    def get_smem_ptr(self, op: TileOp, nbytes: int) -> Expr:
        from hidet.ir.primitives.cuda.smem import dynamic_shared_memory

        requested: int = self.request_smem_nbytes(op)
        if requested == 0:
            raise RuntimeError(
                'Please implement the "request_smem_nbytes" method to return a positive integer before'
                ' accessing the requested shared memory.'
            )
        if annotations.global_offset not in op.annotations:
            raise RuntimeError('No shared memory offset found. Did you forget to run the PlanSharedMemory pass?')

        if nbytes > requested:
            raise RuntimeError(f"Requested {nbytes} bytes of shared memory, but only {requested} bytes are allocated.")

        return dynamic_shared_memory(op.annotations[annotations.global_offset])

    # --------------------------------------------------
    # virtual methods to be implemented for each tile op
    # --------------------------------------------------
    def request_smem_nbytes(self, op: TileOp) -> int:
        return 0

    def implement(self, op: TileOp, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        raise NotImplementedError()


_registered_implementations: Dict[Type[TileOp], Type[TileOpImpl]] = {}


def register_impl(op_cls: Type[TileOp]):
    def decorator(impl_cls: Type[TileOpImpl]):
        _registered_implementations[op_cls] = impl_cls
        return impl_cls

    return decorator


def get_tile_op_impl(op_or_op_cls: Union[TileOp, Type[TileOp]]) -> Type[TileOpImpl]:
    if isinstance(op_or_op_cls, TileOp):
        op_cls = type(op_or_op_cls)
    elif issubclass(op_or_op_cls, TileOp):
        op_cls = op_or_op_cls
    else:
        raise RuntimeError(f"Cannot get tile op impl for {op_or_op_cls}")

    if op_cls not in _registered_implementations:
        parent_classes: Tuple = inspect.getmro(op_cls)
        for cls in parent_classes:
            if cls in _registered_implementations:
                _registered_implementations[op_cls] = _registered_implementations[cls]
                break
        else:
            raise RuntimeError(f"Cannot implement tile op:\n {op_cls.op_name()}")
    impl_cls = _registered_implementations[op_cls]

    return impl_cls


def implement_tile_op(op: TileOp, args: List[Buffer], output: Buffer, num_warps: int) -> Stmt:
    impl_cls = get_tile_op_impl(op)
    impl = impl_cls(num_warps)
    impl.implement(op, args, output)
    return impl.finish()
