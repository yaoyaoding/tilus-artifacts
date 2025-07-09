from typing import Optional, List, Union
from hidet.ir.type import BaseType, DataType, PointerType, data_type
from hidet.ir.expr import Expr
from hidet.ir.tile.type import TileType, TileScope, tile_type
from hidet.ir.tile.layout import TileLayout
from hidet.ir.tile.expr import TileOp
from hidet.utils import same_list


class Broadcast(TileOp):
    def __init__(self, x: Expr, shape: List[int], layout: Optional[TileLayout] = None):
        super().__init__(args=[x], attrs={"shape": shape, "layout": layout})
        self.x: Expr = x
        self.shape: List[int] = shape
        self.layout: Optional[TileLayout] = layout

        if layout is not None:
            assert same_list(layout.logical_shape(), shape)

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        x_type = arg_types[0]
        assert isinstance(x_type, TileType)
        return tile_type(elem_type=x_type.type, shape=self.shape, layout=self.layout, scope=TileScope.Register)


class Reshape(TileOp):
    def __init__(self, x: Expr, shape: List[int], layout: Optional[TileLayout] = None):
        super().__init__(args=[x], attrs={"shape": shape, "layout": layout})
        self.x: Expr = x
        self.shape: List[int] = shape
        self.layout: Optional[TileLayout] = layout

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        x_type = arg_types[0]
        assert isinstance(x_type, TileType)
        return tile_type(elem_type=x_type.type, shape=self.shape, layout=self.layout)


class ExpandDims(TileOp):
    def __init__(self, x: Expr, axis: int, layout: Optional[TileLayout] = None):
        super().__init__(args=[x], attrs={"axis": axis, "layout": layout})
        self.x: Expr = x
        self.axis: int = axis
        self.layout: Optional[TileLayout] = layout

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        x_type = arg_types[0]
        assert isinstance(x_type, TileType)
        x_shape = x_type.shape
        axis = self.axis if self.axis >= 0 else len(x_shape) + self.axis + 1
        y_shape = x_shape[:axis] + [1] + x_shape[axis:]
        return tile_type(elem_type=x_type.type, shape=y_shape, layout=self.layout)


class CastOp(TileOp):
    def __init__(self, x: Expr, dtype: Union[DataType, PointerType]):
        super().__init__(args=[x], attrs={"dtype": dtype})
        self.x: Expr = x
        self.dtype: Union[DataType, PointerType] = dtype

    @classmethod
    def op_name(cls):
        # we use CastOp as the class name to avoid conflict with hidet.ir.expr.Cast
        return 'cast'

    @property
    def var_name_hint(self):
        return 'cst'

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        x_type = arg_types[0]
        assert isinstance(x_type, TileType)
        return tile_type(elem_type=self.dtype, shape=x_type.shape, layout=x_type.layout)


class SliceOp(TileOp):
    def __init__(
        self,
        x: Expr,
        axis: int,
        start: int,
        extent: int,
        layout: Optional[TileLayout] = None,
        scope: Optional[TileScope] = None,
    ):
        super().__init__()
        self.x: Expr = x
        self.axis: axis = axis
        self.start: int = start
        self.extent: int = extent
        self.layout: Optional[TileLayout] = layout
        self.scope: Optional[TileScope] = scope


def broadcast(x: Expr, shape: List[int]):
    return Broadcast(x, shape).make_call()


def reshape(x: Expr, shape: List[int]):
    return Reshape(x, shape).make_call()


def expand_dims(x: Expr, axis: int):
    return ExpandDims(x, axis).make_call()


def cast(x: Expr, dtype: Union[DataType, PointerType, str]):
    if isinstance(dtype, str):
        dtype = data_type(dtype)
    return CastOp(x, dtype).make_call()


def slice(x: Expr, axis: int, start: int, extent: int):
    return SliceOp(x, axis, start, extent).make_call()
