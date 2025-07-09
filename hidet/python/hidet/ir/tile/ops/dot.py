from typing import List, Type

from hidet.ir.expr import Expr
from hidet.ir.tile.expr import TileOp
from hidet.ir.tile.type import TileType
from hidet.ir.type import BaseType
from .creation import zeros


class Dot(TileOp):
    def __init__(self, a: Expr, b: Expr, c: Expr):
        super().__init__(args=[a, b, c], attrs={})
        self.a: Expr = a
        self.b: Expr = b
        self.c: Expr = c

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        return arg_types[2]


class SimtDot(Dot):
    pass


class MmaDot(Dot):
    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        from hidet.ir.tile.layout import MmaDotOperandLayout

        a_type = arg_types[0]
        b_type = arg_types[1]
        c_type = arg_types[2]
        assert isinstance(a_type, TileType) and isinstance(b_type, TileType) and isinstance(c_type, TileType)
        assert isinstance(a_type.layout, MmaDotOperandLayout) and isinstance(b_type.layout, MmaDotOperandLayout)
        assert a_type.layout.mma == b_type.layout.mma
        return TileType(elem_type=c_type.type, shape=c_type.shape, layout=a_type.layout.mma)


def _dot(a: Expr, b: Expr, cls: Type[Dot]):
    from hidet.ir.tools import infer_type

    a_type = infer_type(a)
    b_type = infer_type(b)
    assert isinstance(a_type, TileType) and isinstance(b_type, TileType), (a_type, b_type)
    assert a_type.type == b_type.type, '{} vs {}'.format(a_type, b_type)
    assert issubclass(cls, Dot)
    c = zeros([a_type.shape[0], b_type.shape[1]], a_type.type)
    return cls(a, b, c).make_call()


def dot(a: Expr, b: Expr):
    return _dot(a, b, Dot)


def simt_dot(a: Expr, b: Expr):
    return _dot(a, b, SimtDot)
