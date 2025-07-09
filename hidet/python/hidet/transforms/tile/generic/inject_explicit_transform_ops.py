from typing import List

from hidet.ir.expr import Expr, BinaryExpr
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.tile.ops import Load, StoreBaseOp
from hidet.ir.tile.ops import broadcast, full, expand_dims
from hidet.ir.tile.type import TileType
from hidet.ir.tools import TypeInfer
from hidet.ir.type import PointerType, DataType
from hidet.ir.utils.broadcast_utils import broadcast_shape
from hidet.transforms.base import TileFunctionPass
from hidet.utils import same_list


class InjectExplicitTransformOpsRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()

    def transform_to(self, src: Expr, dst_shape: List[int]) -> Expr:
        src_type = self.type_infer(src)
        if isinstance(src_type, TileType):
            src_shape: List[int] = list(src_type.shape)
            while len(src_shape) < len(dst_shape):
                src_shape.insert(0, 1)
                src = expand_dims(src, 0)
            for a, b in zip(src_shape, dst_shape):
                if a not in [1, b]:
                    raise ValueError(
                        'Cannot transform from shape {} to shape {} with expand_dims and broadcast in expr: \n{}'.format(
                            src_shape, dst_shape, src
                        )
                    )
            if not same_list(src_shape, dst_shape):
                src = broadcast(src, dst_shape)
            return src
        else:
            return full(shape=dst_shape, value=src)

    def visit_Binary(self, e: BinaryExpr):
        a = self.visit(e.a)
        b = self.visit(e.b)
        a_type = self.type_infer.visit(a)
        b_type = self.type_infer.visit(b)
        if isinstance(a_type, TileType) and isinstance(b_type, TileType):
            if not same_list(a_type.shape, b_type.shape):
                shape: List[int] = broadcast_shape(a_type.shape, b_type.shape)
                a = self.transform_to(a, shape)
                b = self.transform_to(b, shape)
        elif isinstance(a_type, TileType) and isinstance(b_type, (PointerType, DataType)):
            b = full(shape=a_type.shape, value=b)
        elif isinstance(a_type, (PointerType, DataType)) and isinstance(b_type, TileType):
            a = full(shape=b_type.shape, value=a)
        return e.__class__(a, b)

    def visit_Load(self, e: Load):
        ptr = self.visit(e.ptr)
        mask = self.visit(e.mask)
        other = self.visit(e.other)

        ptr_type: TileType = self.type_infer(ptr)

        if mask is not None:
            mask = self.transform_to(mask, ptr_type.shape)

        if other is not None:
            other = self.transform_to(other, ptr_type.shape)

        if ptr is e.ptr and mask is e.mask and other is e.other:
            return e
        else:
            return e.reforward([ptr, mask, other])

    def visit_StoreBaseOp(self, e: StoreBaseOp):
        ptr = self.visit(e.ptr)
        mask = self.visit(e.mask)
        value = self.visit(e.value)

        ptr_type: TileType = self.type_infer(ptr)

        if mask is not None:
            mask = self.transform_to(mask, ptr_type.shape)

        value = self.transform_to(value, ptr_type.shape)

        if ptr is e.ptr and mask is e.mask and value is e.value:
            return e
        else:
            return e.reforward([ptr, value, mask])


class InjectExplicitTransformOpsPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = InjectExplicitTransformOpsRewriter()
        return rewriter.visit(func)


def inject_explicit_transform_ops_pass() -> TileFunctionPass:
    return InjectExplicitTransformOpsPass()
