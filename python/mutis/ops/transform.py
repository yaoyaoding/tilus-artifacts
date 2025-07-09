from typing import Tuple, Union, List
import torch
import hidet
from hidet.ir.expr import Expr
from hidet.ir.type import DataType, PointerType, data_type
from mutis.ir import Tensor, Operator, GraphContext
from mutis import utils


class Cast(Operator):
    def __init__(self, a: Tensor, casted_type: Union[DataType, PointerType]):
        self.casted_type = casted_type
        super().__init__(inputs=[a], attrs={'casted_type': casted_type})

    def infer_type(self) -> Tuple[Union[DataType, PointerType], List[Expr]]:
        a: Tensor = self.get_input(0)
        return self.casted_type, list(a.shape)

    def tile_propagation_sets(self) -> List[List[Tuple[int, int]]]:
        sets = []
        a = self.get_input(0)
        for i in range(len(a.shape)):
            sets.append([(0, i), (1, i)])
        return sets


class Repeat(Operator):
    def __init__(self, a: Tensor, repeats: int, dim: int):
        self.repeats: int = repeats
        self.dim: int = dim
        super().__init__(inputs=[a], attrs={'repeats': repeats, 'dim': dim})

    def infer_type(self) -> Tuple[Union[DataType, PointerType], List[Expr]]:
        a = self.get_input(0)
        input_shape = list(a.shape)
        input_shape[self.dim] = input_shape[self.dim] * self.repeats
        return a.elem_type, input_shape

    def tile_propagation_sets(self) -> List[List[Tuple[int, int]]]:
        raise NotImplementedError()


def cast(a: Tensor, casted_type: Union[str, DataType, torch.dtype]) -> Tensor:
    if isinstance(casted_type, str):
        casted_type = data_type(casted_type)
    elif isinstance(casted_type, torch.dtype):
        casted_type = hidet.torch.utils.dtype_from_torch(casted_type)
    assert isinstance(casted_type, (PointerType, DataType))
    op = Cast(a, casted_type)
    GraphContext.current().append(op)
    return op.output


def repeat(a: Tensor, *, repeats: int, dim: int) -> Tensor:
    if dim < 0:
        dim += len(a.shape)
    if dim < 0 or dim >= len(a.shape):
        raise ValueError('Invalid dim: {} for shape {}'.format(dim, a.shape))
    if repeats < 1:
        raise ValueError('Invalid repeats: {}'.format(repeats))
    op = Repeat(a, repeats, dim)
    GraphContext.current().append(op)
    return op.output
