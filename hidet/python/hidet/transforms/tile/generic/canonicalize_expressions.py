from typing import Type, Dict, Union

import hidet.ir.tile.ops.arthimatic as arith
from hidet.ir import expr
from hidet.ir.expr import Expr, BinaryExpr, UnaryExpr
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.tile.ops.arthimatic import UnaryTileOp, BinaryTileOp
from hidet.ir.tile.type import TileType
from hidet.ir.tools import TypeInfer
from hidet.transforms.base import TileFunctionPass

_convert_table: Dict[Type[Expr], Type[Union[UnaryTileOp, BinaryTileOp]]] = {
    # unary arithmetic
    expr.Neg: arith.Neg,
    expr.LogicalNot: arith.LogicalNot,
    expr.BitwiseNot: arith.BitwiseNot,
    # binary arithmetic
    expr.Add: arith.Add,
    expr.Sub: arith.Sub,
    expr.Multiply: arith.Multiply,
    expr.Div: arith.Div,
    expr.Mod: arith.Mod,
    expr.LessThan: arith.LessThan,
    expr.LessEqual: arith.LessEqual,
    expr.Equal: arith.Equal,
    expr.NotEqual: arith.NotEqual,
    expr.LogicalAnd: arith.LogicalAnd,
    expr.LogicalOr: arith.LogicalOr,
}


class CanonicalizeExpressionsRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()

    def visit_Unary(self, e: UnaryExpr):
        op_cls = type(e)
        a = self.visit(e.a)
        a_type = self.type_infer.visit(a)
        if op_cls in _convert_table and isinstance(a_type, TileType):
            tile_op_cls: Type[arith.UnaryTileOp] = _convert_table[op_cls]
            return tile_op_cls(a).make_call()
        else:
            return super().visit_Unary(e)

    def visit_Binary(self, e: BinaryExpr):
        op_cls = type(e)
        a = self.visit(e.a)
        b = self.visit(e.b)
        a_type = self.type_infer.visit(a)
        b_type = self.type_infer.visit(b)
        if op_cls in _convert_table and (isinstance(a_type, TileType) or isinstance(b_type, TileType)):
            tile_op_cls: Type[arith.BinaryTileOp] = _convert_table[op_cls]
            return tile_op_cls(a, b).make_call()
        else:
            return super().visit_Binary(e)


class CanonicalizeExpressionsPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = CanonicalizeExpressionsRewriter()
        return rewriter.visit(func)


def canonicalize_expressions_pass() -> TileFunctionPass:
    return CanonicalizeExpressionsPass()
