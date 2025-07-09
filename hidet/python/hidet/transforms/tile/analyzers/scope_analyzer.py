from typing import Dict, Tuple

from hidet.ir.expr import Var, Expr
from hidet.ir.functors import IRVisitor
from hidet.ir.stmt import Stmt, LetStmt


class ScopeEntity:
    pass


class LetBindingEntity(ScopeEntity):
    def __init__(self, stmt: LetStmt, idx: int):
        self.stmt: LetStmt = stmt
        self.idx: int = idx
        self.bind_var: Var = stmt.bind_vars[idx]
        self.bind_value: Expr = stmt.bind_values[idx]

    def __eq__(self, other):
        assert isinstance(other, ScopeEntity)
        return isinstance(other, LetBindingEntity) and self.stmt is other.stmt and self.idx == other.idx

    def __hash__(self):
        return hash((self.stmt, self.idx))


class StmtEntity(ScopeEntity):
    def __init__(self, stmt: Stmt):
        self.stmt: Stmt = stmt

    def __eq__(self, other):
        assert isinstance(other, ScopeEntity)
        return isinstance(other, StmtEntity) and self.stmt is other.stmt

    def __hash__(self):
        return hash(self.stmt)


class ScopeAnalyzer(IRVisitor):
    def __init__(self):
        super().__init__(use_memo=False)
        self.numbering: Dict[ScopeEntity, Tuple[int, int]] = {}
        self.clock: int = 0

    def visit(self, node):
        scope = []
        if isinstance(node, Stmt):
            scope.append(self.clock)
            self.clock += 1
        super().visit(node)
        if isinstance(node, Stmt):
            scope.append(self.clock)
            self.clock += 1
            self.numbering[StmtEntity(node)] = (scope[0], scope[1])

    def visit_LetStmt(self, stmt: LetStmt):
        for idx in range(len(stmt.bind_vars)):
            self.numbering[LetBindingEntity(stmt, idx)] = (self.clock, self.clock + 1)
            self.clock += 2
        self.visit(stmt.body)
