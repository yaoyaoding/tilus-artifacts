from typing import Optional, Union
from hidet.ir.type import DataType
from hidet.ir.expr import Expr
from mutis.ir.graph import Tensor
from mutis.ops.arithmatic import _unary_op


def relu(a: Tensor) -> Tensor:
    return _unary_op(a, op='relu')


def clip(a: Tensor, min_value: Optional[Union[float, int]], max_value: Optional[Union[float, int]]) -> Tensor:
    dtype = a.elem_type
    assert isinstance(dtype, DataType)
    return _unary_op(a, op='clip', attrs={'min_value': dtype(min_value), 'max_value': dtype(max_value)})
