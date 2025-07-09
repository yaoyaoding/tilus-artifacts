from hidet.ir.type import DataType
from hidet.ir.expr import Expr
from .arthimatic import UnaryTileOp


class ActivationOp(UnaryTileOp):
    def apply_scalar(self, x: Expr) -> Expr:
        raise NotImplementedError()


class Exp(ActivationOp):
    def apply_scalar(self, x: Expr) -> Expr:
        from hidet.ir.primitives import math

        return math.exp(x)


class Silu(ActivationOp):
    def apply_scalar(self, x: Expr) -> Expr:
        # x * sigmoid(x) = x * (1 / (1 + exp(-x)))
        from hidet.ir.primitives import math
        from hidet.ir.tools import infer_type

        dtype = infer_type(x)
        assert isinstance(dtype, DataType)
        return x / (dtype.one + math.exp(-x))


def exp(x):
    return Exp(x).make_call()


def silu(x):
    return Silu(x).make_call()
