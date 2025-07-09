from typing import Tuple, Union, List, Iterable, Dict, Optional

from hidet.ir import PointerType
from hidet.ir.expr import Expr
from hidet.ir.type import DataType
from mutis.ir.graph import Tensor, Operator, GraphContext
from mutis.ir.layout import Layout
from mutis import utils


class ElementwiseUnary(Operator):
    VALID_OPS = ['relu', 'clip']

    def __init__(self, a: Tensor, op: str, other_attrs: Dict[str, Union[Expr, int, float]]):
        self.op = op
        super().__init__(inputs=[a], attrs={'op': op, **other_attrs})

        assert op in ElementwiseUnary.VALID_OPS

    def infer_type(self) -> Tuple[Union[DataType, PointerType], List[Expr]]:
        a = self.get_input(0)
        return a.elem_type, list(a.shape)

    def tile_propagation_sets(self) -> List[List[Tuple[int, int]]]:
        sets = []
        a = self.get_input(0)
        b = self.get_output()
        for i in range(len(b.shape)):
            st = [(1, i)]
            if len(b.shape) - i <= len(a.shape):
                st.append((0, len(a.shape) - len(b.shape) + i))
                sets.append(st)
        return sets


class ElementwiseBinary(Operator):
    VALID_OPS = ['+', '-', '*', '/', '%', '<', '<=', '>', '>=', '==', '!=', '&&', '||', '&', '|']

    def __init__(self, a: Tensor, b: Tensor, op: str):
        self.op = op
        super().__init__(inputs=[a, b], attrs={'op': op})
        assert op in ElementwiseBinary.VALID_OPS

    def infer_type(self) -> Tuple[Union[DataType], List[Expr]]:
        a = self.get_input(0)
        b = self.get_input(1)
        shape = utils.broadcast_shape(a.shape, b.shape)
        utils.check_same_elem_type(a, b, msg='elementwise operator requires both operands have the same type')
        if self.op in ['+', '-', '*', '/', '%']:
            if self.op == '%' and not (a.elem_type.is_data_type() and a.elem_type.as_data_type().is_integer()):
                raise ValueError('% only supports integer, got {}'.format(a.elem_type))
            return a.elem_type, shape
        else:
            raise NotImplementedError()

    def tile_propagation_sets(self) -> List[List[Tuple[int, int]]]:
        sets = []
        a = self.get_input(0)
        b = self.get_input(1)
        c = self.get_output()
        for i in range(len(c.shape)):
            st = [(2, i)]
            if len(c.shape) - i <= len(a.shape):
                st.append((0, len(a.shape) - len(c.shape) + i))
            if len(c.shape) - i <= len(b.shape):
                st.append((1, len(b.shape) - len(c.shape) + i))
            sets.append(st)

        return sets


class BroadcastElementwiseBinary(Operator):
    VALID_OPS = ['+', '-', '*', '/', '%', '<', '<=', '>', '>=', '==', '!=', '&&', '||', '&', '|']

    def __init__(self, t: Tensor, s: Expr, op: str, tensor_left: bool):
        self.op = op
        self.s = s
        self.tensor_left = tensor_left

        super().__init__(inputs=[t], attrs={'op': op, 'scalar': s, 'tensor_left': tensor_left})
        assert op in BroadcastElementwiseBinary.VALID_OPS

    def infer_type(self) -> Tuple[Union[DataType, PointerType], List[Expr]]:
        t = self.get_input(0)
        shape = list(t.shape)
        if self.op in ['+', '-', '*', '/', '%']:
            return t.elem_type, shape
        else:
            raise NotImplementedError()

    def tile_propagation_sets(self) -> List[List[Tuple[int, int]]]:
        sets = []
        a = self.get_input(0)
        b = self.get_output()
        for i in range(len(b.shape)):
            st = [(1, i)]
            if len(b.shape) - i <= len(a.shape):
                st.append((0, len(a.shape) - len(b.shape) + i))
                sets.append(st)
        return sets


def _unary_op(a: Tensor, op: str, attrs: Optional[Dict[str, Union[Expr, int, float]]] = None) -> Tensor:
    op = ElementwiseUnary(a, op, attrs or {})
    GraphContext.current().append(op)
    return op.output


def _binary_op(a: Union[Tensor, Expr], b: Union[Tensor, Expr], op: str) -> Tensor:
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        op = ElementwiseBinary(a, b, op)
    elif isinstance(a, Tensor) and isinstance(b, Expr):
        op = BroadcastElementwiseBinary(a, b, op, tensor_left=True)
    elif isinstance(a, Expr) and isinstance(b, Tensor):
        op = BroadcastElementwiseBinary(b, a, op, tensor_left=False)
    else:
        raise ValueError('unsupported type: {} and {}'.format(type(a), type(b)))

    GraphContext.current().append(op)
    return op.output


def add(a: Union[Tensor, Expr], b: Union[Tensor, Expr]) -> Tensor:
    return _binary_op(a, b, '+')


def sub(a: Union[Tensor, Expr], b: Union[Tensor, Expr]) -> Tensor:
    return _binary_op(a, b, '-')


def multiply(a: Union[Tensor, Expr], b: Union[Tensor, Expr]) -> Tensor:
    return _binary_op(a, b, '*')


def divide(a: Union[Tensor, Expr], b: Union[Tensor, Expr]) -> Tensor:
    return _binary_op(a, b, '/')


def maximum(a: Union[Tensor, Expr], b: Union[Tensor, Expr]) -> Tensor:
    return _binary_op(a, b, 'max')


def minimum(a: Union[Tensor, Expr], b: Union[Tensor, Expr]) -> Tensor:
    return _binary_op(a, b, 'min')
