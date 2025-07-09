from __future__ import annotations
from typing import Optional, List, Dict, Type, Callable
import operator

import hidet.ir.tools
from hidet.ir.type import DataType, PointerType, sizeof
from hidet.ir.expr import Var, Constant, Add, Multiply, Sub, Expr, BinaryExpr, LessThan, LogicalAnd
from hidet.ir.stmt import LetStmt
from hidet.ir.func import Function
from hidet.ir.tile.type import TileType
from hidet.ir.tile.expr import CallTileOp, TileOp
from hidet.ir.tile.stmt import PureForStmt, YieldStmt
from hidet.ir.tile.ops import Create, BinaryTileOp
from hidet.ir.functors import IRVisitor
from hidet.utils import gcd, same_list
from hidet.transforms.base import TileFunctionPass


class Constancy:
    def __init__(self, is_constant: bool):
        self.is_constant: bool = is_constant


class Divisibility:
    def __init__(self, v: int = 1):
        self.v: int = v


class Continuity:
    def __init__(self, v: int = 1):
        self.v: int = v


class ValueInfo:
    def as_tensor_info(self):
        assert isinstance(self, TensorInfo)
        return self

    def as_scalar_info(self):
        assert isinstance(self, ScalarInfo)
        return self

    def merge(self, other: ValueInfo) -> ValueInfo:
        raise NotImplementedError()


class OptionalInt:
    def __init__(self, v: Optional[int] = None):
        self.v: Optional[int] = v

    def _binary(self, other, op):
        assert isinstance(other, OptionalInt)
        if self.v is None or other.v is None:
            return OptionalInt(None)
        return OptionalInt(op(self.v, other.v))

    def __str__(self):
        if self.v is None:
            return 'none'
        return str(self.v)

    def __eq__(self, other):
        assert isinstance(other, OptionalInt)
        return self.v == other.v

    def __add__(self, other):
        return self._binary(other, operator.add)

    def __sub__(self, other):
        return self._binary(other, operator.sub)

    def __mul__(self, other):
        return self._binary(other, operator.mul)

    def __lt__(self, other):
        return self._binary(other, lambda a, b: int(a < b))

    def __and__(self, other):
        return self._binary(other, lambda a, b: int(a & b))

    def merge(self, other: OptionalInt) -> OptionalInt:
        if self.v is None or other.v is None or self.v != other.v:
            return OptionalInt(None)
        return self


class ScalarInfo(ValueInfo):
    def __init__(self, divisibility: int = None, const_value=None):
        self.divisibility: int = abs(divisibility) if divisibility is not None else 1
        self.const_value: OptionalInt = const_value if const_value is not None else OptionalInt(None)

    def __str__(self):
        return 'scalar(divisibility={}, const={})'.format(self.divisibility, self.const_value)

    def __eq__(self, other):
        assert isinstance(other, ScalarInfo)
        return self.divisibility == other.divisibility and self.const_value == other.const_value

    def __add__(self, other):
        if isinstance(other, TensorInfo):
            return NotImplemented
        assert isinstance(other, ScalarInfo)
        return ScalarInfo(
            divisibility=gcd(self.divisibility, other.divisibility), const_value=self.const_value + other.const_value
        )

    def __sub__(self, other):
        if isinstance(other, TensorInfo):
            return NotImplemented
        assert isinstance(other, ScalarInfo)
        return ScalarInfo(
            divisibility=gcd(self.divisibility, other.divisibility), const_value=self.const_value - other.const_value
        )

    def __mul__(self, other):
        if isinstance(other, TensorInfo):
            return NotImplemented
        assert isinstance(other, ScalarInfo)
        return ScalarInfo(
            divisibility=self.divisibility * other.divisibility, const_value=self.const_value * other.const_value
        )

    def __lt__(self, other):
        if isinstance(other, TensorInfo):
            return NotImplemented
        assert isinstance(other, ScalarInfo)
        return ScalarInfo(divisibility=1, const_value=self.const_value < other.const_value)

    def merge(self, info: ValueInfo):
        if isinstance(info, ScalarInfo):
            return ScalarInfo(
                divisibility=gcd(self.divisibility, info.divisibility),
                const_value=self.const_value.merge(info.const_value),
            )
        elif isinstance(info, TensorInfo):
            return info.merge(TensorInfo.from_scalar(info.shape, self))
        else:
            raise NotImplementedError()


class TensorInfo(ValueInfo):
    def __init__(
        self, shape: List[int], divisibility: List[int], continuity: List[int], constancy: List[int], const_value=None
    ):
        self.shape: List[int] = shape
        self.continuity: List[int] = continuity
        self.divisibility: List[int] = divisibility
        self.constancy: List[int] = constancy
        self.const_value: OptionalInt = const_value if const_value else OptionalInt()

        assert all(v >= 0 for v in shape + continuity + divisibility)

    def _convert_other(self, other) -> TensorInfo:
        if isinstance(other, ScalarInfo):
            other = TensorInfo.from_scalar(self.shape, other)
        assert isinstance(other, TensorInfo)
        assert same_list(self.shape, other.shape)
        return other

    def __str__(self):
        return 'tensor(shape={}, continuity={}, divisibility={}, constancy={}, value={})'.format(
            self.shape, self.continuity, self.divisibility, self.constancy, self.const_value
        )

    def __eq__(self, other):
        if not isinstance(other, TensorInfo):
            return False
        return (
            same_list(self.shape, other.shape)
            and same_list(self.continuity, other.continuity)
            and same_list(self.divisibility, other.divisibility)
            and same_list(self.constancy, other.constancy)
            and self.const_value == other.const_value
        )

    def __add__(self, other):
        other = self._convert_other(other)
        continuity, constancy, divisibility = [], [], []
        for i in range(len(self.shape)):
            constancy.append(gcd(self.constancy[i], other.constancy[i]))
            if self.constancy[i] > 1 and other.constancy[i] > 1:
                continuity.append(1)
                divisibility.append(gcd(self.divisibility[i], other.divisibility[i]))
            elif self.constancy[i] > 1:
                continuity.append(gcd(self.constancy[i], other.continuity[i]))
                divisibility.append(gcd(other.divisibility[i], self.divisibility[i], continuity[-1]))
            elif other.constancy[i] > 1:
                continuity.append(gcd(self.continuity[i], other.constancy[i]))
                divisibility.append(gcd(self.divisibility[i], other.divisibility[i], continuity[-1]))
            else:
                continuity.append(1)
                divisibility.append(1)
        return TensorInfo(self.shape, divisibility, continuity, constancy, self.const_value + other.const_value)

    def __sub__(self, other):
        other = self._convert_other(other)
        add_result = self + other
        add_result.const_value = add_result.const_value - other.const_value
        return add_result

    def __mul__(self, other):
        other = self._convert_other(other)
        continuity, constancy, divisibility = [], [], []
        for i in range(len(self.shape)):
            constancy.append(gcd(self.constancy[i], other.constancy[i]))
            if self.constancy[i] > 1 and other.constancy[i] > 1:
                continuity.append(1)
                divisibility.append(self.divisibility[i] * other.divisibility[i])
            else:
                continuity.append(1)
                divisibility.append(1)
        return TensorInfo(self.shape, divisibility, continuity, constancy, self.const_value * other.const_value)

    def __lt__(self, other):
        other = self._convert_other(other)
        continuity, constancy, divisibility = [], [], []
        for i in range(len(self.shape)):
            continuity.append(1)
            divisibility.append(1)
            if self.constancy[i] > 1 and other.constancy[i] > 1:
                constancy.append(gcd(self.constancy[i], other.constancy[i]))
            elif self.constancy[i] > 1:
                constancy.append(
                    gcd(self.constancy[i], self.divisibility[i], other.continuity[i], other.divisibility[i])
                )
            elif other.constancy[i] > 1:
                constancy.append(
                    gcd(other.constancy[i], other.divisibility[i], self.continuity[i], self.divisibility[i])
                )
            else:
                constancy.append(gcd(self.constancy[i], other.constancy[i]))
        return TensorInfo(self.shape, divisibility, continuity, constancy, self.const_value < other.const_value)

    def __and__(self, other):
        other = self._convert_other(other)
        continuity, constancy, divisibility = [], [], []
        for i in range(len(self.shape)):
            constancy.append(gcd(self.constancy[i], other.constancy[i]))
            continuity.append(1)
            divisibility.append(1)
        return TensorInfo(self.shape, divisibility, continuity, constancy, self.const_value & other.const_value)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        other = self._convert_other(other)
        return other - self

    def __rmul__(self, other):
        return self.__mul__(other)

    @staticmethod
    def from_axis(shape: List[int], dim: int):
        continuity, constancy, divisibility = [], [], []
        for i, extent in enumerate(shape):
            if i == dim:
                continuity.append(extent)
                divisibility.append(0)
                constancy.append(1)
            else:
                continuity.append(1)
                divisibility.append(1)
                constancy.append(extent)
        return TensorInfo(shape, divisibility, continuity, constancy)

    @staticmethod
    def from_scalar(shape: List[int], scalar_info: ScalarInfo):
        continuity, constancy, divisibility = [], [], []
        for extent in shape:
            continuity.append(1)
            divisibility.append(scalar_info.divisibility)
            constancy.append(extent)
        return TensorInfo(shape, divisibility, continuity, constancy, const_value=scalar_info.const_value)

    @staticmethod
    def from_shape(shape: List[int]):
        continuity, constancy, divisibility = [], [], []
        for _ in range(len(shape)):
            continuity.append(1)
            divisibility.append(1)
            constancy.append(1)
        return TensorInfo(shape, divisibility, continuity, constancy)

    def merge(self, info: ValueInfo):
        if isinstance(info, ScalarInfo):
            return self.merge(TensorInfo.from_scalar(self.shape, info))
        elif isinstance(info, TensorInfo):
            continuity, constancy, divisibility = [], [], []
            for i in range(len(self.shape)):
                continuity.append(gcd(self.continuity[i], info.continuity[i]))
                constancy.append(gcd(self.constancy[i], info.constancy[i]))
                divisibility.append(gcd(self.divisibility[i], info.divisibility[i]))
            return TensorInfo(self.shape, divisibility, continuity, constancy, self.const_value.merge(info.const_value))
        else:
            raise NotImplementedError()


class ValueAnalyzer(IRVisitor):
    """
    Given a variable, it may have multiple potential values
    a := 8
    b := 16
       | a + 4
    c := b
       | a + b
    Let value(x) be the Value (with constancy and divisibility) of x.
    Then we have the following equations:
    value(a) = value(8)
    value(b) = merge(value(16), value(a) + value(4))
    value(c) = merge(value(b), value(a) + value(b))

    This class implements an iterative algorithm to solve the above equations.
    """

    def __init__(self):
        super().__init__()
        self.var2value: Dict[Var, ValueInfo] = {}
        self.updated: bool = False

    def analyze(self, func: Function):
        # repeat the update until it converges to a fixed point, which is the solution of the coalesce analysis
        while True:
            self.updated = False
            self.visit(func)
            if not self.updated:
                break

    def merge(self, v: Var, value: Optional[ValueInfo]):
        if value is None:
            return
        if v not in self.var2value:
            self.var2value[v] = value
            self.updated = True
        else:
            new_value = self.var2value[v].merge(value)
            if new_value != self.var2value[v]:
                self.var2value[v] = new_value
                self.updated = True

    def visit_Function(self, func: Function):
        for arg in func.params:
            if arg.type.is_pointer():
                # we assume that the pointer has 128-bytes alignment
                elem_type = arg.type.as_pointer_type().base_type
                self.merge(arg, ScalarInfo(divisibility=128 // sizeof(elem_type)))
            elif arg.type.is_data_type() and arg.type.as_data_type().is_integer():
                self.merge(arg, ScalarInfo(divisibility=1))

        self.visit(func.body)

    def visit_LetStmt(self, stmt: LetStmt):
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            assert isinstance(bind_var, Var)
            var_type = bind_var.type
            if isinstance(var_type, TileType):
                self.merge(bind_var, self.visit(bind_value))
            elif isinstance(var_type, DataType) and var_type.is_integer() or isinstance(var_type, PointerType):
                self.merge(bind_var, self.visit(bind_value))
            else:
                # do nothing for other types
                pass
        self.visit(stmt.body)

    def visit_PureForStmt(self, stmt: PureForStmt):
        for arg, value in zip(stmt.args, stmt.values):
            if isinstance(arg.type, TileType):
                self.merge(arg, self.visit(value))
        self.pure_for_stmts.append(stmt)
        self.visit(stmt.body)
        self.pure_for_stmts.pop()
        for arg, let_var in zip(stmt.args, stmt.let_vars):
            if isinstance(arg.type, TileType):
                self.merge(let_var, self.var2value.get(arg, None))
        self.visit(stmt.let_body)

    def visit_YieldStmt(self, stmt: YieldStmt):
        for_stmt = self.pure_for_stmts[-1]
        for arg, yield_value in zip(for_stmt.args, stmt.values):
            if isinstance(arg.type, TileType):
                self.merge(arg, self.visit(yield_value))

    def visit_CallTileOp(self, call: CallTileOp):
        return self.visit(call.op)

    def visit_Var(self, e: Var):
        if e in self.var2value:
            return self.var2value[e]
        else:
            return None

    # scalar expressions

    def visit_Constant(self, e: Constant):
        if e.type.is_data_type() and e.type.is_integer() or e.type.is_pointer():
            return ScalarInfo(divisibility=int(e.value), const_value=OptionalInt(int(e.value)))
        else:
            return None

    def visit_binary(self, e: BinaryExpr):
        a = self.visit(e.a)
        b = self.visit(e.b)
        if a is None:
            a = ScalarInfo()
        if b is None:
            b = ScalarInfo()
        op_dict: Dict[Type[Expr], Callable] = {
            Add: operator.add,
            Sub: operator.sub,
            Multiply: operator.mul,
            LessThan: operator.lt,
            LogicalAnd: operator.and_,
        }
        if type(e) not in op_dict:
            raise NotImplementedError()
        c = op_dict[type(e)](a, b)

        # for debugging
        # from hidet.utils.py import color
        # print('{} {} {} = {}'.format(
        #     color(a, fg='magenta'),
        #     color(type(e).__name__, fg='yellow'),
        #     color(b, fg='green'),
        #     color(c, fg='cyan')
        # ))
        return c

    def visit_Add(self, e: Add):
        return self.visit_binary(e)

    def visit_Sub(self, e: Sub):
        return self.visit_binary(e)

    def visit_Multiply(self, e: Multiply):
        return self.visit_binary(e)

    def visit_LessThan(self, e: LessThan):
        return self.visit_binary(e)

    def visit_And(self, e: LogicalAnd):
        return self.visit_binary(e)

    # tile operators

    def visit_Create(self, e: Create):
        for dim, axis in enumerate(e.axes):
            self.memo[axis] = TensorInfo.from_axis(shape=e.shape, dim=dim)
        info: Optional[ValueInfo] = self.visit(e.value)
        if info is None:
            return None
        if isinstance(info, TensorInfo):
            return info
        elif isinstance(info, ScalarInfo):
            return TensorInfo.from_scalar(e.shape, info)
        else:
            assert False

    def visit_BinaryTileOp(self, e: BinaryTileOp):
        import hidet.ir.tile.ops.arthimatic as ops

        op_dict: Dict[Type[TileOp], Callable] = {
            ops.Add: operator.add,
            ops.Sub: operator.sub,
            ops.Multiply: operator.mul,
            ops.Mod: operator.mod,
            ops.LogicalAnd: operator.and_,  # use bitwise and as a proxy for logical and
        }
        if type(e) in op_dict:
            op = op_dict[type(e)]
            x = self.visit(e.x)
            y = self.visit(e.y)
            if x is None or y is None:
                return None
            else:
                return op(x, y)
        else:
            return None


class ValueAnalyzePass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        analyzer = ValueAnalyzer()
        analyzer.analyze(func)

        # print the result
        from tabulate import tabulate

        printer = hidet.ir.tools.IRPrinter()
        print('Function: ')
        print(printer.astext(func))
        lines = []
        for arg, info in analyzer.var2value.items():
            lines.append([str(printer(arg)), str(info)])
        print(tabulate(lines, headers=['Variable', 'Value Information']))

        # an analysis pass should not change the function
        return func


def analyze_value(func: Function):
    analyzer = ValueAnalyzer()
    analyzer.analyze(func)
    return analyzer.var2value


def value_analyze_pass() -> TileFunctionPass:
    return ValueAnalyzePass()
