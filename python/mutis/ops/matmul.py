from typing import Tuple, Union, List, Optional
from hidet.ir.expr import Expr
from hidet.ir.type import DataType, PointerType, data_type
from mutis.ir import Tensor, Operator, GraphContext
from mutis import utils


class Matmul(Operator):
    def __init__(self, a: Tensor, b: Tensor, acc_dtype: DataType, out_dtype: DataType):
        super().__init__(inputs=[a, b], attrs={'acc_dtype': acc_dtype, 'out_dtype': out_dtype})
        self.acc_dtype: DataType = acc_dtype
        self.out_dtype: DataType = out_dtype

    def infer_type(self) -> Tuple[Union[DataType, PointerType], List[Expr]]:
        a = self.get_input(0)
        b = self.get_input(1)
        utils.check_same_elem_type(a, b, msg='matmul require a and b have the same type')
        shape = utils.broadcast_shape(a.shape[:-2], b.shape[:-2])
        return self.attrs['out_dtype'], shape + [a.shape[-2], b.shape[-1]]

    def tile_propagation_sets(self) -> List[List[Tuple[int, int]]]:
        a = self.get_input(0)
        b = self.get_input(1)
        c = self.get_output()
        sets = [
            [(0, len(a.shape) - 2), (2, len(c.shape) - 2)],
            [(0, len(a.shape) - 1), (1, len(b.shape) - 2)],
            [(1, len(b.shape) - 1), (2, len(c.shape) - 1)],
        ]
        for i in range(len(c.shape) - 2):
            st = [(2, i)]
            if len(c.shape) - i <= len(a.shape):
                st.append((0, len(a.shape) - len(c.shape) + i))
            if len(c.shape) - i <= len(b.shape):
                st.append((1, len(b.shape) - len(c.shape) + i))
            sets.append(st)

        return sets


def matmul(
    a: Tensor, b: Tensor, acc_dtype: Union[str, DataType] = 'float32', out_dtype: Optional[Union[str, DataType]] = None
) -> Tensor:
    if isinstance(acc_dtype, str):
        acc_dtype = data_type(acc_dtype)
    if out_dtype is None:
        out_dtype = a.elem_type
    elif isinstance(out_dtype, str):
        out_dtype = data_type(out_dtype)
    op = Matmul(a, b, acc_dtype, out_dtype)
    GraphContext.current().append(op)
    return op.output
