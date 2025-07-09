from typing import List, Optional, Union
import torch

from hidet.ir.type import DataType
from hidet.ir.dtypes import float32, int32
from hidet.graph.frontend.torch.utils import dtype_from_torch, dtype_to_torch
from hidet.utils import prod
from mutis.jit import jit
from mutis.jit.types import void, constant
from mutis import ops
from mutis.utils import same_list


@jit(tuning_trigger_params=[])  # do not tune for different n
def _cast_kernel(n: int, src_ptr: ~void, dst_ptr: ~void, src_dtype: constant[DataType], dst_dtype: constant[DataType]):
    src = ops.load(src_ptr, dtype=src_dtype, shape=[n])
    dst = ops.cast(src, dst_dtype)
    ops.store(dst, dst_ptr)


@jit(tuning_trigger_params=[])  # do not tune for different n
def _set_slice_2d_kernel(
    m: int,
    n: int,
    offset_m: int,
    offset_n: int,
    value_m: int,
    value_n: int,
    tensor_ptr: ~void,
    value_ptr: ~void,
    dtype: constant[DataType],
):
    value = ops.load(value_ptr, dtype=dtype, shape=[value_m, value_n])
    ops.store(value, ptr=tensor_ptr, strides=[n, 1], offset=offset_m * n + offset_n)


class Tensor:
    def __init__(self, dtype: DataType, shape: List[int], storage: torch.Tensor):
        self.dtype: DataType = dtype
        self.shape: List[int] = shape
        self.storage: torch.Tensor = storage

    def __str__(self):
        if self.dtype.is_integer():
            storage = cast(self, int32).storage
            torch_tensor = storage.view(torch.int32).to(torch.int64).reshape(self.shape)
        elif self.dtype.is_float():
            storage = cast(self, float32).storage
            torch_tensor = storage.view(torch.float32).reshape(self.shape)
        else:
            raise ValueError()
        return '{}{}:\n{}'.format(self.dtype.name, self.shape, torch_tensor)

    def __setitem__(self, key, value):
        # analyze the key
        offsets = []
        slice_shape = []
        if isinstance(key, int):
            key = (key,)
        if len(key) > len(self.shape):
            raise ValueError('Too many indices.')
        if len(key) < len(self.shape):
            key = key + (slice(None),) * (len(self.shape) - len(key))
        for i, item in enumerate(key):
            if isinstance(item, int):
                offsets.append(item)
                slice_shape.append(1)
            elif isinstance(item, slice):
                if item.step is not None:
                    raise ValueError('Slice step is not supported.')
                start = 0 if item.start is None else int(item.start)
                stop = self.shape[i] if item.stop is None else int(item.stop)
                offsets.append(start)
                slice_shape.append(stop - start)
            else:
                raise ValueError('Unsupported index: {}.'.format(item))

        # check the value
        if isinstance(value, (float, int)):
            value = full(slice_shape, value, self.dtype)
        if not same_list(slice_shape, value.shape):
            raise ValueError('Shape mismatch: {} vs {}.'.format(slice_shape, value.shape))
        if self.dtype != value.dtype:
            value = cast(value, self.dtype)
        set_slice(self, offsets=offsets, value=value)

    def view(self, *, dtype: Optional[DataType] = None, shape: Optional[List[int]] = None):
        return view(self, dtype, shape)

    def clone(self):
        return Tensor(self.dtype, list(self.shape), self.storage.clone())

    def torch(self) -> torch.Tensor:
        torch_dtype = dtype_to_torch(self.dtype)
        if torch_dtype is None:
            raise ValueError('PyTorch does not support dtype {} for now.'.format(self.dtype.name))
        return self.storage.view(torch_dtype).reshape(self.shape)

    def to(self, dtype: DataType):
        return cast(self, dtype)

    def data_ptr(self) -> int:
        return self.storage.data_ptr()


def from_torch(torch_tensor: torch.Tensor) -> Tensor:
    dtype = dtype_from_torch(torch_tensor.dtype)
    return Tensor(dtype, list(torch_tensor.shape), torch_tensor)


def view_torch(torch_tensor: torch.Tensor, *, dtype: DataType, shape: List[int]) -> Tensor:
    assert (dtype.nbits * prod(shape) + 7) // 8 == torch_tensor.nbytes
    return Tensor(dtype, shape, torch_tensor)


def cast(tensor: Tensor, dtype: DataType) -> Tensor:
    if tensor.dtype == dtype:
        return tensor
    new_tensor = empty(shape=tensor.shape, dtype=dtype)
    n = prod(tensor.shape)
    _cast_kernel(n, tensor.storage, new_tensor.storage, tensor.dtype, dtype)
    return new_tensor


def empty(shape: List[int], dtype: DataType) -> Tensor:
    nbytes = (dtype.nbits * prod(shape) + 7) // 8
    storage = torch.empty([nbytes], dtype=torch.uint8, device='cuda')
    return Tensor(dtype, shape, storage)


def randn(shape: List[int], dtype: DataType) -> Tensor:
    tensor = from_torch(torch.randn(shape, dtype=torch.float32, device='cuda'))
    tensor = cast(tensor, dtype)
    return tensor


def randint(low: int, high: int, shape: List[int], dtype: DataType) -> Tensor:
    tensor = from_torch(torch.randint(low=low, high=high, size=shape, dtype=torch.int32, device='cuda'))
    tensor = cast(tensor, dtype)
    return tensor


def ones(shape: List[int], dtype: DataType) -> Tensor:
    tensor = from_torch(torch.ones(shape, dtype=torch.float32, device='cuda'))
    tensor = cast(tensor, dtype)
    return tensor


def zeros(shape: List[int], dtype: DataType) -> Tensor:
    tensor = from_torch(torch.zeros(shape, dtype=torch.float32, device='cuda'))
    tensor = cast(tensor, dtype)
    return tensor


def full(shape: List[int], fill_value: Union[float, int], dtype: DataType) -> Tensor:
    tensor = from_torch(torch.full(shape, fill_value, dtype=torch.float32, device='cuda'))
    tensor = cast(tensor, dtype)
    return tensor


def view(tensor: Tensor, dtype: Optional[DataType] = None, shape: Optional[List[int]] = None) -> Tensor:
    if dtype is None:
        dtype = tensor.dtype
    if shape is None:
        shape = list(tensor.shape)
        if dtype.nbits == tensor.dtype.nbits:
            pass  # no change
        elif (
            dtype.nbits % tensor.dtype.nbits == 0 and shape[-1] % (dtype.nbits // tensor.dtype.nbits) == 0
        ) or tensor.dtype.nbits % dtype.nbits == 0:
            shape[-1] = shape[-1] * tensor.dtype.nbits // dtype.nbits
        else:
            raise ValueError('Cannot infer shape.')

    actual_nbits = dtype.nbits * prod(shape)
    expect_nbits = tensor.dtype.nbits * prod(tensor.shape)
    assert actual_nbits == expect_nbits, f'{actual_nbits} != {expect_nbits}'
    return Tensor(dtype, shape, tensor.storage)


def set_slice(tensor: Tensor, offsets: List[int], value: Tensor):
    assert tensor.dtype == value.dtype
    assert len(tensor.shape) == len(value.shape) == len(offsets) == 2
    assert all(offset + value.shape[i] <= tensor.shape[i] for i, offset in enumerate(offsets))
    m, n = tensor.shape
    offset_m, offset_n = offsets
    value_m, value_n = value.shape
    _set_slice_2d_kernel(m, n, offset_m, offset_n, value_m, value_n, tensor, value, tensor.dtype)


def arange(start: int, end: Optional[int] = None, step: Optional[int] = None, *, dtype: DataType):
    if end is None:
        end = start
        start = 0
    if step is None:
        step = 1
    return from_torch(torch.arange(int(start), int(end), int(step))).to(dtype)
