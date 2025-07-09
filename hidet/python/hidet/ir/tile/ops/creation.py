from typing import Optional, Union, Sequence, Callable, List, Tuple, Any

from hidet.ir.expr import Var, convert, index_vars
from hidet.ir.tile.expr import TileOp, Expr
from hidet.ir.tile.type import tile_type, TileLayout
from hidet.ir.type import BaseType, DataType, data_type


class Create(TileOp):
    def __init__(self, value: Expr, shape: List[int], axes: List[Var], layout: Optional[TileLayout] = None):
        super().__init__(args=[value], attrs={"shape": shape, "axes": axes, "layout": layout})
        self.shape: List[int] = shape
        self.axes: List[Var] = axes
        self.value: Expr = value
        self.layout: Optional[TileLayout] = layout

    def __getitem__(self, actual_indices: List[Union[Expr, int]]) -> Expr:
        from hidet.ir.tools import rewrite

        remap = {axis: convert(actual_index) for axis, actual_index in zip(self.axes, actual_indices)}
        return rewrite(self.value, remap)

    @staticmethod
    def from_compute(shape: List[int], f_compute: Callable[[List[Var]], Expr], layout: Optional[TileLayout] = None):
        axes: List[Var] = index_vars(num_vars=len(shape))
        value: Expr = f_compute(axes)
        return Create(value, shape, axes, layout)

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        x_type = arg_types[0]
        return tile_type(elem_type=x_type, shape=self.shape, layout=self.layout)


def arange(begin: int, end: int):
    return Create.from_compute(shape=[end - begin], f_compute=lambda axes: axes[0] + convert(begin)).make_call()


def full(*, shape: Sequence[int], value: Union[Expr, int, bool, float], dtype: Optional[Union[DataType, str]] = None):
    if dtype is not None:
        dtype = data_type(dtype)
        value = dtype(value)
    else:
        value = convert(value)

    return Create.from_compute(shape=list(shape), f_compute=lambda axes: value).make_call()


def grid(shape: List[int], starts: List[Union[Expr, int]], strides: List[Union[Expr, int]], offset=0):
    offset = convert(offset)
    starts = [convert(start) for start in starts]
    strides = [convert(stride) for stride in strides]
    return Create.from_compute(
        shape=shape, f_compute=lambda axes: offset + sum((axes[i] + starts[i]) * strides[i] for i in range(len(shape)))
    ).make_call()


def compute(shape: List[int], f_compute: Callable):
    return Create.from_compute(shape=shape, f_compute=lambda axes: f_compute(*axes)).make_call()


def zeros(shape: List[int], dtype: Union[DataType, str] = 'float32'):
    dtype = data_type(dtype)
    return full(shape=shape, value=dtype.zero)


def ones(shape: List[int], dtype: Union[DataType, str] = 'float32'):
    dtype = data_type(dtype)
    return full(value=dtype.one, shape=shape)


def construct(shape: List[int], f_compute: Callable[[List[Var]], Expr]):
    return Create.from_compute(shape, f_compute).make_call()
