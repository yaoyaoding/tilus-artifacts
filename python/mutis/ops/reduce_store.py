from typing import Tuple, Union, List, Optional
from hidet.ir.expr import Expr, Var, convert
from hidet.ir.type import DataType, PointerType
from mutis.ir import Tensor, Operator, GraphContext
from mutis.ops.ldst import _resolve_strides


class ReduceStore(Operator):
    """
    Perform atomic operation:

        dst = reduce(src)

    where dst is the tensor determined via `ptr`, `tensor.shape` and `strides` on global memory.
    """

    NO_OUTPUT = True

    def __init__(self, tensor: Tensor, reduce_op: str, ptr: Var, strides: List[Expr]):
        super().__init__(inputs=[tensor], attrs={'ptr': ptr, 'strides': strides})
        self.ptr: Var = ptr
        self.strides: List[Expr] = strides

    def tile_propagation_sets(self) -> List[List[Tuple[int, int]]]:
        return []


def _reduce_store(
    tensor: Tensor,
    reduce_op: str,
    ptr: Var,
    strides: Optional[List[Union[Expr, int]]] = None,
    order: Optional[List[int]] = None,
):
    strides = _resolve_strides(tensor.shape, strides, order)
    strides = [convert(e) for e in strides]
    op = ReduceStore(tensor, reduce_op=reduce_op, ptr=ptr, strides=strides)
    GraphContext.current().append(op)


def reduce_max_store(
    tensor: Tensor, ptr: Var, strides: Optional[List[Union[Expr, int]]] = None, order: Optional[List[int]] = None
):
    _reduce_store(tensor=tensor, reduce_op='reduce_max', ptr=ptr, strides=strides, order=order)
