from typing import Dict, Optional, Union

import hidet.option
from hidet.ir.type import DataType
from hidet.ir.expr import Var, Expr
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.dtypes import float16
from hidet.ir.tile.layout import TileLayout, BlockLayout
from hidet.ir.tile.ops import ConvertLayout, convert_layout
from hidet.ir.tile.ops.memory import StoreBaseOp, Load, AtomicAdd
from hidet.ir.tile.type import TileType
from hidet.ir.type import PointerType, sizeof
from hidet.transforms.base import TileFunctionPass
from hidet.transforms.tile.analyzers.value_analyzer import analyze_value, TensorInfo, ValueInfo
from hidet.transforms.tile.generic.canonicalize_to_ssa import canonicalize_to_ssa


class CoalesceMemoryAccessRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.var2info: Dict[Var, ValueInfo] = {}

    def try_to_get_vectorized_layout(self, ptr: Expr, op: Union[Load, StoreBaseOp]) -> Optional[TileLayout]:
        ptr = self.visit(ptr)
        assert isinstance(ptr, Var)

        if ptr not in self.var2info:
            # does not know the constancy and divisibility information for ptr
            return None

        if not ptr.type.is_tile_type():
            # accessing a scalar value
            return None

        tv: TensorInfo = self.var2info[ptr].as_tensor_info()
        if len(tv.shape) == 0:
            # scalar tile (e.g., len(shape) == 0)
            return None

        # calculate the largest number of valid vectorized elements
        # in cuda, we can load at most 16 bytes per thread
        elem_type: Union[PointerType, DataType] = ptr.type.as_tile_type().type.base_type
        dtype_bytes: int = sizeof(elem_type)
        vector_elements: int = min(min(tv.divisibility[-1], tv.continuity[-1]) * dtype_bytes, 16) // dtype_bytes
        if vector_elements == 1:
            return None

        if isinstance(op, AtomicAdd):
            if hidet.option.cuda.get_arch_pair() < (9, 0) and elem_type == float16:
                # red instruction only supports f16x2 without vectorization for sm < 90
                vector_elements = 2

        ttype: TileType = ptr.type.as_tile_type()
        orig_layout = ttype.layout

        return BlockLayout.from_shape(
            shape=ttype.shape,
            num_warps=orig_layout.num_workers() // 32,
            size_per_thread=[1 if i != len(ttype.shape) - 1 else vector_elements for i in range(len(ttype.shape))],
        )

    def visit_Function(self, func: Function):
        self.var2info = analyze_value(func)
        return super().visit_Function(func)

    def visit_Load(self, e: Load):
        ptr = self.visit(e.ptr)
        new_layout: Optional[BlockLayout] = self.try_to_get_vectorized_layout(ptr, e)
        if new_layout is None:
            return super().visit_Load(e)
        else:
            orig_layout = ptr.type.as_tile_type().layout
            ptr = convert_layout(ptr, new_layout)
            mask: Optional[Expr] = convert_layout(self.visit(e.mask), new_layout) if e.mask is not None else None
            other: Optional[Expr] = convert_layout(self.visit(e.other), new_layout) if e.other is not None else None
            return ConvertLayout(Load(ptr=ptr, mask=mask, other=other).make_call(), orig_layout)

    def visit_StoreBaseOp(self, e: StoreBaseOp):
        ptr = self.visit(e.ptr)
        new_layout: Optional[BlockLayout] = self.try_to_get_vectorized_layout(ptr, e)
        if new_layout is None:
            return super().visit_StoreBaseOp(e)
        else:
            ptr = convert_layout(ptr, new_layout)
            value: Expr = convert_layout(self.visit(e.value), new_layout) if e.value is not None else None
            mask: Optional[Expr] = convert_layout(self.visit(e.mask), new_layout) if e.mask is not None else None
            return e.reforward(args=[ptr, value, mask])


class CoalesceMemoryAccessPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = CoalesceMemoryAccessRewriter()
        func = rewriter(func)
        func = canonicalize_to_ssa(func)
        return func


def coalesce_memory_access_pass() -> TileFunctionPass:
    return CoalesceMemoryAccessPass()
