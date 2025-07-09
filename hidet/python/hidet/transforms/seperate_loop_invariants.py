# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import operator
from typing import Dict, Optional, Union, Tuple, List, Callable, Set
from itertools import product
import functools

from hidet.ir import ForMappingStmt, AssignStmt, WhileStmt, Let, DeclareStmt
from hidet.ir.dialects.pattern import PlaceholderExpr, match
from hidet.ir.dtypes import boolean, int32
from hidet.ir.expr import (
    Add,
    convert,
    Sub,
    Multiply,
    Mod,
    LessThan,
    LessEqual,
    Equal,
    NotEqual,
    BinaryExpr,
    LogicalAnd,
    SymbolVar,
)
from hidet.ir.expr import BitwiseXor, BitwiseAnd, BitwiseOr, BitwiseNot, Var, LogicalOr
from hidet.ir.expr import Div, Constant, Expr, logical_and, constant, IfThenElse
from hidet.ir.stmt import LetStmt, ForStmt
from hidet.ir.primitives.cuda.vars import blockDim, gridDim, blockIdx, threadIdx
from hidet.ir.tile.ops import Assign
from hidet.ir.tools import TypeInfer, IRPrinter, collect
from hidet.ir.functors import IRRewriter, IRVisitor
from hidet.ir.tools import rewrite, simplify
from hidet.transforms.base import FunctionPass
from hidet.utils import prod, repeat_until_converge, same_list
from hidet.ir.func import Function
from hidet.ir.analyzers import BoundAnalyzer, BoundInfo


class DecomposeRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()

    def visit_Multiply(self, e: Multiply):
        if isinstance(e.a, Constant) and isinstance(e.b, Constant):
            # c1 * c2 => c1*c2
            return Constant(value=e.a.value * e.b.value, const_type=self.type_infer(e))
        elif isinstance(e.a, Add) and isinstance(e.b, Constant):
            # (e1 + e2) * c => e1 * c + e2 * c
            e1, e2, c = e.a.a, e.a.b, e.b
            if self.type_infer(e1 + e2) == self.type_infer(e1 * c) == self.type_infer(e2 * c):
                return self.visit(e1 * c + e2 * c)
            else:
                return super().visit_Multiply(e)
        elif isinstance(e.a, Multiply) and isinstance(e.a.b, Constant) and isinstance(e.b, Constant):
            # (e1 * c1) * c2 => e1 * (c1 * c2)
            e1, c1, c2 = e.a.a, e.a.b, e.b
            if self.type_infer(e1 * c1) == self.type_infer(c1 * c2):
                # in case: (uint64 * uint32) * uint32 and c1 * c2 overflow
                return self.visit(e1 * (c1 * c2))
            else:
                return super().visit_Multiply(e)
        else:
            return super().visit_Multiply(e)


class DepthAnalyzer(IRVisitor):
    """
    Collect the information of a function, and use the collected information to generate the key used for
    sort the items in an addition chain. First need use this analyzer to visit the function, and then
    it can be used to generate the keys.
    """
    FIRST_LOOP_DEPTH = 2

    def __init__(self, func, printer):
        super().__init__()
        self.var2depth: Dict[Var, int] = {}
        self.func = func
        self.type_infer = TypeInfer()
        self.printer = printer
        self.current_depth = 1

        self.expr2depth: Dict[Expr, int] = {}

        self.printer(func)
        self.visit(func)

    def get_depth(self, expr: Expr):
        if expr in self.expr2depth:
            return self.expr2depth[expr]

        used_vars: List[Var] = collect(expr, node_types=(Var,))
        depth = 0
        for var in used_vars:
            if var.type.is_func_type():
                continue
            if var in self.var2depth:
                var_depth = self.var2depth[var]
            else:
                msg = 'The depth for variable "{}" has not been determined in function:\n{}'.format(
                    self.printer(var), self.printer(self.func)
                )
                raise ValueError(msg)
            depth = max(depth, var_depth)

        self.expr2depth[expr] = depth
        return depth

    def visit_Function(self, func: Function):
        global_invariants = [
            gridDim.x,
            gridDim.y,
            gridDim.z,
            blockDim.x,
            blockDim.y,
            blockDim.z,
            blockIdx.x,
            blockIdx.y,
            blockIdx.z,
            threadIdx.x,
            threadIdx.y,
            threadIdx.z,
        ]
        for invariant in global_invariants:
            self.var2depth[invariant] = self.current_depth
        for param in func.params:
            self.var2depth[param] = self.current_depth
        self.current_depth += 1
        super().visit_Function(func)
        self.current_depth -= 1

    def visit_ForStmt(self, stmt: ForStmt):
        self.current_depth += 1
        self.var2depth[stmt.loop_var] = self.current_depth
        super().visit_ForStmt(stmt)
        self.current_depth -= 1

    def visit_WhileStmt(self, stmt: WhileStmt):
        self.current_depth += 1
        super().visit_WhileStmt(stmt)
        self.current_depth -= 1

    def visit_LetStmt(self, stmt: LetStmt):
        for var in stmt.bind_vars:
            self.var2depth[var] = self.current_depth
        super().visit_LetStmt(stmt)

    def visit_Var(self, e: Var):
        if isinstance(e, SymbolVar):
            self.var2depth[e] = 1
        super().visit_Var(e)

    def visit_AssignStmt(self, stmt: AssignStmt):
        self.var2depth[stmt.var] = max(self.var2depth[stmt.var], self.current_depth)
        super().visit_AssignStmt(stmt)

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        self.var2depth[stmt.var] = self.current_depth
        super().visit_DeclareStmt(stmt)

    def visit_Let(self, e: Let):
        raise ValueError('Please first lower the Let expression.')

    def visit_ForTaskStmt(self, stmt: ForMappingStmt):
        raise ValueError('Please first lower the ForMappingStmt statement.')

class AdditionChainTransform:
    def __call__(self, chain: List[Expr]) -> List[Expr]:
        raise NotImplementedError()

class ReorderChain(AdditionChainTransform):
    """
    Reorder the items of the addition chain according to key(expr): (max-depth(all vars in expr), str(expr))

    The max-depth is the maximum loop depth that modified/defined the variable.
    """
    def __init__(self, printer: IRPrinter, depth_analyzer: DepthAnalyzer):
        self.analyzer: DepthAnalyzer = depth_analyzer
        self.printer: IRPrinter = printer

    def __call__(self, chain: List[Expr]) -> List[Expr]:
        def key_func(e):
            depth = self.analyzer.get_depth(e)
            text = str(self.printer(e))
            # we order the items like
            # items with depth = 1, items with depth = 0, items with depths >= 2
            # if depth in [0, 1]:
            #     depth = 1 - depth
            if depth == 0:
                depth = 1000
            return depth, text
        return list(sorted(chain, key=key_func))


class DivModCancellation(AdditionChainTransform):
    """
    e / c * c + e % c => e where e / c * c and e % c are two expressions in the addition chain

    expr1: e / c * c
    expr2: e % c
    """
    def __init__(self, printer):
        self.printer: IRPrinter = printer

    def __call__(self, chain: List[Expr]) -> List[Expr]:
        while True:
            replaced = self.search_and_replace(chain)
            if not replaced:
                break
        return chain

    def search_and_replace(self, chain: List[Expr]) -> bool:
        for e1 in chain:
            for e2 in chain:
                if e1 is e2:
                    continue
                ec = self.check(e1, e2)
                if ec is None:
                    continue
                e, c = ec
                chain.remove(e1)
                chain.remove(e2)
                chain.append(e)
                return True
        return False

    def check(self, e1: Expr, e2: Expr) -> Optional[Tuple[Expr, Constant]]:
        if isinstance(e2, Mod) and isinstance(e2.b, Constant):
            e, c = e2.a, e2.b
            if (
                    isinstance(e1, Multiply)
                    and isinstance(e1.a, Div)
                    and isinstance(e1.b, Constant)
                    and isinstance(e1.a.b, Constant)
                    and e1.b == e1.a.b == c
                    and str(self.printer(e1.a.a)) == str(self.printer(e))
            ):
                return e, c
        return None


class AdditionChainRewriter(IRRewriter):
    def __init__(self, transform: Optional[Callable[[List[Expr]], List[Expr]]] = None):
        super().__init__()
        self.transform: Callable[[List[Expr]], List[Expr]] = transform if transform is not None else lambda x: x
        self.add2chain: Dict[Add, List[Expr]] = {}

    def visit_Add(self, e: Add):
        # chain decomposition
        chain: List[Expr] = []
        for operand in [e.a, e.b]:
            self.visit(operand)
            if isinstance(operand, Add):
                chain.extend(self.add2chain[operand])
            else:
                chain.append(self.visit(operand))

        # chain transformation
        chain = self.transform(chain)

        # add expression reconstruction
        ret = chain[0]
        for i in range(1, len(chain)):
            ret = Add(ret, chain[i])

        self.add2chain[e] = chain
        return ret

class RegroupComparisonOperands(AdditionChainRewriter):
    def __init__(self, depth_analyzer: DepthAnalyzer):
        super().__init__()
        self.analyzer: DepthAnalyzer = depth_analyzer

    def visit_compare(self, e: Union[LessThan, LessEqual, Equal, NotEqual]):
        a = self.visit(e.a)
        b = self.visit(e.b)

        lhs_exprs = self.add2chain[e.a] if isinstance(e.a, Add) else [a]
        rhs_exprs = self.add2chain[e.b] if isinstance(e.b, Add) else [b]

        lhs_depths = [self.analyzer.get_depth(expr) for expr in lhs_exprs]
        rhs_depths = [self.analyzer.get_depth(expr) for expr in rhs_exprs]

        # print(self.analyzer.printer(e))
        # print('lhs depths: ', lhs_depths)
        # print('rhs depths: ', rhs_depths)

        FIRST_LOOP_DEPTH = DepthAnalyzer.FIRST_LOOP_DEPTH
        if max(lhs_depths) <= FIRST_LOOP_DEPTH and max(rhs_depths) <= FIRST_LOOP_DEPTH:
            return e.__class__(a, b) if a is not e.a or b is not e.b else e
        elif max(rhs_depths) <= FIRST_LOOP_DEPTH:
            rhs = b
            lhs_remaining = []
            for lhs_expr in lhs_exprs:
                if self.analyzer.get_depth(lhs_expr) <= FIRST_LOOP_DEPTH:
                    rhs = Sub(rhs, lhs_expr)
                else:
                    lhs_remaining.append(lhs_expr)
            lhs = functools.reduce(Add, lhs_remaining)
            ret = e.__class__(lhs, rhs)
            return ret
        elif max(lhs_depths) <= FIRST_LOOP_DEPTH:
            lhs = a
            rhs_remaining = []
            for rhs_expr in rhs_exprs:
                if self.analyzer.get_depth(rhs_expr) <= FIRST_LOOP_DEPTH:
                    lhs = Sub(lhs, rhs_expr)
                else:
                    rhs_remaining.append(rhs_expr)
            rhs = functools.reduce(Add, rhs_remaining)
            return e.__class__(lhs, rhs)
        else:
            return e.__class__(a, b) if a is not e.a or b is not e.b else e


    def visit_LessThan(self, e: LessThan):
        return self.visit_compare(e)

    def visit_LessEqual(self, e: LessEqual):
        return self.visit_compare(e)

    def visit_Equal(self, e: Equal):
        return self.visit_compare(e)

    def visit_NotEqual(self, e: NotEqual):
        return self.visit_compare(e)


class SeparateLoopInvariantsPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        printer = IRPrinter()
        printer(func)

        # decompose expressions like (a + b)*c => a*c + b*c
        rewriter = DecomposeRewriter()
        func = rewriter(func)

        depth_analyzer = DepthAnalyzer(func, printer)
        depth_analyzer.visit(func)

        # reorder order of addition
        rewriter = AdditionChainRewriter(transform=ReorderChain(printer=printer, depth_analyzer=depth_analyzer))
        func = rewriter(func)

        # cancel div-mod pair: e / c * c + e % c => e
        rewriter = AdditionChainRewriter(transform=DivModCancellation(printer=printer))
        func = rewriter(func)

        # # regroups comparison operands
        # rewriter = RegroupComparisonOperands(depth_analyzer=depth_analyzer)
        # func = rewriter(func)

        return func


def separate_loop_invariants_pass() -> FunctionPass:
    return SeparateLoopInvariantsPass()
