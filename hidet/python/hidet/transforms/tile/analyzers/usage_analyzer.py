from typing import List, Dict, Set
from collections import defaultdict
from hidet.ir.expr import Var, Expr
from hidet.ir.stmt import LetStmt, EvaluateStmt
from hidet.ir.tile.stmt import PureForStmt, YieldStmt
from hidet.ir.functors import IRVisitor
from hidet.ir.tile.expr import TileOp, CallTileOp


class LetUsage:
    """let ... = convert_layout(x)"""

    def __init__(self, let_stmt, idx):
        self.let_stmt: LetStmt = let_stmt
        self.idx: int = idx
        self.bind_var: Var = let_stmt.bind_vars[idx]
        self.bind_value: Expr = let_stmt.bind_values[idx]

    @property
    def op(self) -> TileOp:
        assert isinstance(self.bind_value, CallTileOp)
        return self.bind_value.op


class StmtUsage:
    """
    for ... in ... with arg=x, ...
    yield x
    store(...)
    """

    def __init__(self, stmt):
        self.stmt = stmt


class VarUsage:
    def __init__(self):
        self.let_usages: List[LetUsage] = []
        self.stmt_usages: List[StmtUsage] = []

    def count(self):
        return len(self.let_usages) + len(self.stmt_usages)

    def call_op_let_usages(self):
        return [usage for usage in self.let_usages if isinstance(usage.bind_value, CallTileOp)]


class UsageAnalyzer(IRVisitor):
    def __init__(self):
        super().__init__(use_memo=False)
        self.usages: Dict[Var, VarUsage] = defaultdict(VarUsage)
        self.used_vars: Set[Var] = set()

    def visit_Var(self, var: Var):
        self.used_vars.add(var)

    def collect_used_vars(self, expr) -> Set[Var]:
        self.used_vars.clear()
        self.visit(expr)
        return self.used_vars

    def visit_LetStmt(self, stmt: LetStmt):
        for idx, (_, bind_value) in enumerate(zip(stmt.bind_vars, stmt.bind_values)):
            for used_var in self.collect_used_vars(bind_value):
                self.usages[used_var].let_usages.append(LetUsage(stmt, idx))
        self.visit(stmt.body)

    def visit_PureForStmt(self, stmt: PureForStmt):
        for value in stmt.values:
            for used_var in self.collect_used_vars(value):
                self.usages[used_var].stmt_usages.append(StmtUsage(stmt))
        self.visit(stmt.body)
        self.visit(stmt.let_body)

    def visit_YieldStmt(self, stmt: YieldStmt):
        for value in stmt.values:
            for used_var in self.collect_used_vars(value):
                self.usages[used_var].stmt_usages.append(StmtUsage(stmt))

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        for used_var in self.collect_used_vars(stmt.expr):
            self.usages[used_var].stmt_usages.append(StmtUsage(stmt))


def analyze_usage(node) -> Dict[Var, VarUsage]:
    analyzer = UsageAnalyzer()
    analyzer.visit(node)
    return analyzer.usages
