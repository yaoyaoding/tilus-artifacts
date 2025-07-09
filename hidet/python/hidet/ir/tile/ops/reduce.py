from enum import Enum
from typing import Optional, List

from hidet.ir.expr import Expr
from hidet.ir.tile.expr import TileOp
from hidet.ir.tile.layout import TileLayout
from hidet.ir.tile.type import TileType, tile_type
from hidet.ir.type import BaseType, DataType


class ReduceKind(Enum):
    min = 'min'
    max = 'max'
    sum = 'sum'

    def default_value(self, dtype: DataType):
        if self.name == 'min':
            return dtype.max_value
        elif self.name == 'max':
            return dtype.min_value
        elif self.name == 'sum':
            return dtype.zero
        else:
            raise RuntimeError(f"Unknown reduce kind {self.name}")

    def combine(self, lhs: Expr, rhs: Expr):
        from hidet.ir import primitives

        if self.name == 'min':
            return primitives.min(lhs, rhs)
        elif self.name == 'max':
            return primitives.max(lhs, rhs)
        elif self.name == 'sum':
            return lhs + rhs
        else:
            raise RuntimeError(f"Unknown reduce kind {self.name}")


class ReduceOp(TileOp):
    def __init__(self, x: Expr, axis: int, keepdims: bool, kind: ReduceKind, layout: Optional[TileLayout] = None):
        super().__init__(args=[x], attrs={"axis": axis, "keepdims": keepdims, "kind": kind, "layout": layout})
        self.x: Expr = x
        self.axis: int = axis
        self.keepdims: bool = keepdims
        self.kind: ReduceKind = kind
        self.layout: Optional[TileLayout] = layout

    @property
    def var_name_hint(self):
        return 'reduce_{}'.format(self.kind.name)

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        x_type = arg_types[0]
        assert isinstance(x_type, TileType)
        x_shape = x_type.shape
        axis = self.axis if self.axis >= 0 else len(x_shape) + self.axis + 1
        if self.keepdims:
            y_shape = x_shape[:axis] + [1] + x_shape[axis + 1 :]
        else:
            y_shape = x_shape[:axis] + x_shape[axis + 1 :]
        return tile_type(elem_type=x_type.type, shape=y_shape, layout=self.layout)


def sum(x: Expr, axis: int, keepdims: bool = False):
    return ReduceOp(x, axis, keepdims, ReduceKind.sum).make_call()


def min(x: Expr, axis: int, keepdims: bool = False):
    return ReduceOp(x, axis, keepdims, ReduceKind.min).make_call()


def max(x: Expr, axis: int, keepdims: bool = False):
    return ReduceOp(x, axis, keepdims, ReduceKind.max).make_call()
