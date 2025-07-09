from typing import Tuple, Union, List, Optional
from hidet.ir.expr import Expr, Var, convert
from hidet.ir.type import DataType, PointerType, type_equal, void_p
from hidet.ir.dtypes import int32
from mutis.ir import Tensor, Operator, GraphContext
from hidet.ir.tools import infer_type


class Load(Operator):
    def __init__(
        self, ptr: Var, dtype: DataType, shape: List[Expr], strides: List[Expr], cast_dtype: Optional[DataType]
    ):
        self.ptr: Var = ptr
        self.dtype: DataType = dtype
        self.shape: List[Expr] = shape
        self.strides: List[Expr] = strides
        self.cast_dtype: Optional[DataType] = cast_dtype
        super().__init__(
            inputs=[], attrs={'ptr': ptr, 'dtype': dtype, 'shape': shape, 'strides': strides, 'cast_dtype': cast_dtype}
        )

        assert all(isinstance(e, Expr) for e in shape)
        assert all(isinstance(e, Expr) for e in strides)
        assert len(shape) == len(strides)
        if not type_equal(ptr.type, void_p):
            assert type_equal(ptr.type, ~dtype)

    def infer_type(self) -> Tuple[Union[DataType, PointerType], List[Expr]]:
        dtype = self.cast_dtype if self.cast_dtype is not None else self.dtype
        return dtype, list(self.shape)

    def tile_propagation_sets(self) -> List[List[Tuple[int, int]]]:
        return []


class Store(Operator):
    NO_OUTPUT = True

    def __init__(self, tensor: Tensor, ptr: Var, strides: List[Expr], offset: Optional[Expr] = None):
        super().__init__(inputs=[tensor], attrs={'ptr': ptr, 'strides': strides, 'offset': offset})
        self.ptr: Var = ptr
        self.strides: List[Expr] = strides
        self.offset: Expr = convert(offset) if offset is not None else int32(0)

        assert all(isinstance(e, Expr) for e in strides)
        assert type_equal(ptr.type, void_p) or type_equal(ptr.type, ~tensor.elem_type)

    def tile_propagation_sets(self) -> List[List[Tuple[int, int]]]:
        return []


def _strides_from_order(shape: List[Expr], order: List[int]) -> List[Expr]:
    if len(shape) != len(order):
        raise ValueError('order must have the same number of dimensions as shape, got {} and {}'.format(shape, order))
    ndim = len(shape)
    if len(set(order)) != ndim or not all(0 <= v < ndim for v in order):
        raise ValueError('order must be a permutation of [0, 1, ..., len(shape) - 1], got {}'.format(order))
    order = list(order)
    strides = [-1 for _ in range(ndim)]
    stride = 1
    for _ in range(ndim):
        index = max(range(ndim), key=lambda i: order[i])
        strides[index] = stride
        stride *= shape[index]
        order[index] = -1
    return [convert(e) for e in strides]


def _resolve_strides(
    shape: List[Union[Expr, int]], strides: Optional[List[Union[Expr, int]]] = None, order: Optional[List[int]] = None
):
    if strides is not None and order is not None:
        raise ValueError('Can not provide strides and order at the same time.')
    if order is not None:
        strides = _strides_from_order(shape, order)
    elif strides is not None:
        pass  # do nothing
    else:
        strides = _strides_from_order(shape, order=list(range(len(shape))))
    return strides


def _check_dtype(ptr: Var, dtype: Optional[DataType]) -> DataType:
    import torch
    from hidet.graph.frontend.torch.utils import dtype_from_torch

    if isinstance(dtype, torch.dtype):
        dtype = dtype_from_torch(dtype)

    if dtype is None:
        if isinstance(ptr.type, PointerType) and isinstance(ptr.type.base_type, DataType):
            dtype = ptr.type.base_type
        elif type_equal(ptr.type, void_p):
            raise ValueError('To load from a void pointer, please specify the `dtype` parameter of `load` operator.')
        else:
            raise ValueError('Expect to load from a pointer of data type, got {}'.format(ptr.type))
    else:
        if type_equal(ptr.type, void_p):
            # okay
            pass
        elif isinstance(ptr.type, PointerType):
            if not type_equal(ptr.type, ~dtype):
                raise ValueError('Datatype mismatch: {} vs {}'.format(ptr.type, dtype))
        else:
            raise ValueError('Expect to load from a pointer of data type, got {}'.format(ptr.type))
    return dtype


def _check_cast_dtype(cast_dtype: Optional[Union[DataType, 'torch.dtype']]) -> Optional[DataType]:
    import torch
    from hidet.graph.frontend.torch.utils import dtype_from_torch

    if isinstance(cast_dtype, torch.dtype):
        cast_dtype = dtype_from_torch(cast_dtype)

    if cast_dtype is None:
        return None
    if not isinstance(cast_dtype, DataType):
        raise ValueError('Expect `cast_dtype` to be a DataType, got {}'.format(cast_dtype))
    return cast_dtype


def load(
    ptr: Var,
    *,
    dtype: Optional[Union[DataType, 'torch.dtype']] = None,
    cast_dtype: Optional[Union[DataType, 'torch.dtype']] = None,
    shape: List[Union[Expr, int]],
    strides: Optional[List[Union[Expr, int]]] = None,
    order: Optional[List[int]] = None,
) -> Tensor:
    assert isinstance(ptr, Var)
    dtype = _check_dtype(ptr, dtype)
    cast_dtype = _check_cast_dtype(cast_dtype)
    strides = _resolve_strides(shape, strides, order)
    shape = [convert(e) for e in shape]
    strides = [convert(e) for e in strides]
    op = Load(ptr, dtype=dtype, shape=shape, strides=strides, cast_dtype=cast_dtype)
    GraphContext.current().append(op)
    return op.output


def store(
    tensor: Tensor,
    ptr: Var,
    *,
    strides: Optional[List[Union[Expr, int]]] = None,
    order: Optional[List[int]] = None,
    offset: Optional[Union[Expr, int]] = None,
):
    _check_dtype(ptr, tensor.elem_type)
    strides = _resolve_strides(tensor.shape, strides, order)
    strides = [convert(e) for e in strides]
    op = Store(tensor, ptr, strides=strides, offset=offset)
    GraphContext.current().append(op)
