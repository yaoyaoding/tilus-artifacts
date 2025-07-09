"""
Dead code elimination pass eliminates the code that does not affect the final result.

We solve following equation:

live[u] = u used in an operator that writes to memory (e.g., store)
        | any(live[v] for v that depends u)

where live[u] is a boolean value indicating whether u is live or not.

This pass assumes that the input function is in SSA form.
"""
from typing import Dict, Union, Set, List

from hidet.ir.expr import Var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.module import IRModule
from hidet.ir.stmt import LetStmt, Stmt
from hidet.ir.tile.expr import CallTileOp, TileOp
from hidet.ir.tile.ops import DebugPrint
from hidet.ir.tile.stmt import PureForStmt, YieldStmt
from hidet.ir.tools import collect
from hidet.transforms.base import TileFunctionPass
from hidet.transforms.tile.analyzers import DependencyAnalyzer
from hidet.transforms.tile.utils import collect_yield_stmts


class DeadCodeEliminationRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.live: Set[Var] = set()

    def visit_Function(self, func: Function):
        self.memo.clear()  # in case calling this rewriter multiple times

        # get the dependency relation-ship
        dependency_analyzer = DependencyAnalyzer()
        dependency_analyzer.visit_Function(func)
        depends: Dict[Var, List[Var]] = dependency_analyzer.depends

        # add the dependency relation-ship from pure for arg to its yield stmt
        for2yields: Dict[PureForStmt, List[YieldStmt]] = collect_yield_stmts(func)
        roots: List[Var] = []  # the roots of the dependency graph
        for for_stmt, yield_stmts in for2yields.items():
            for yield_stmt in yield_stmts:
                for arg, value in zip(for_stmt.args, yield_stmt.values):
                    depends[arg].append(value)
            assert isinstance(for_stmt.extent, Var), 'Need SSA form to perform DCE'
            roots.append(for_stmt.extent)

        # find all the CallTileOp and mark the args of the memory-writing ops as live
        for call_tile_op in collect(func, CallTileOp):
            op: TileOp = call_tile_op.op
            if op.write_memory_op() or isinstance(op, DebugPrint):
                for arg in op.args:
                    assert isinstance(arg, Var), 'DeadCodeEliminationRewriter only works on SSA form'
                    roots.append(arg)

        # mark the roots and all its dependencies as live
        stack: List[Var] = roots
        self.live: Set[Var] = set(roots)
        while len(stack) > 0:
            u = stack.pop()
            if u not in depends:
                continue
            for v in depends[u]:
                if v not in self.live:
                    self.live.add(v)
                    stack.append(v)

        return super().visit_Function(func)

    def visit_LetStmt(self, stmt):
        bind_vars = []
        bind_values = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            if bind_var in self.live:
                bind_vars.append(bind_var)
                bind_values.append(bind_value)
        body = self.visit(stmt.body)
        if len(bind_vars) == len(stmt.bind_vars) and body is stmt.body:
            return stmt
        else:
            if len(bind_vars) == 0:
                return body
            else:
                return LetStmt(bind_vars, bind_values, body)

    def visit_PureForStmt(self, stmt: PureForStmt):
        args = []
        values = []
        let_vars = []
        for arg, value, let_var in zip(stmt.args, stmt.values, stmt.let_vars):
            if arg in self.live:
                args.append(arg)
                values.append(value)
                let_vars.append(let_var)
        self.pure_for_stmts.append(stmt)
        body = self.visit(stmt.body)
        self.pure_for_stmts.pop()
        let_body = self.visit(stmt.let_body)
        if len(args) == len(stmt.args) and body is stmt.body and let_body is stmt.let_body:
            return stmt
        else:
            return PureForStmt(
                args=args,
                values=values,
                loop_var=stmt.loop_var,
                extent=stmt.extent,
                body=body,
                let_vars=let_vars,
                let_body=let_body,
            )

    def visit_YieldStmt(self, stmt: YieldStmt):
        for_stmt = self.pure_for_stmts[-1]
        values = []
        for arg, value in zip(for_stmt.args, stmt.values):
            if arg in self.live:
                values.append(value)
        if len(values) == len(stmt.values):
            return stmt
        else:
            return YieldStmt(values)


class DeadCodeEliminationPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = DeadCodeEliminationRewriter()
        return rewriter.visit(func)


def eliminate_dead_code(node: Union[IRModule, Function, Stmt]):
    rewriter = DeadCodeEliminationRewriter()
    return rewriter.visit(node)


def dead_code_elimination_pass() -> TileFunctionPass:
    return DeadCodeEliminationPass()
