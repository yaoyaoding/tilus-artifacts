from typing import Dict, List
from hidet.ir.stmt import Stmt, LetStmt, SeqStmt
from hidet.ir.tile.stmt import PureForStmt, YieldStmt
from hidet.ir.functors import IRVisitor


class YieldStmtCollector(IRVisitor):
    def __init__(self):
        super().__init__()
        self.for2yield: Dict[PureForStmt, List[YieldStmt]] = {}

    def visit_YieldStmt(self, stmt: YieldStmt):
        for_stmt = self.pure_for_stmts[-1]
        if for_stmt not in self.for2yield:
            self.for2yield[for_stmt] = []
        self.for2yield[for_stmt].append(stmt)


def collect_yield_stmts(node):
    collector = YieldStmtCollector()
    collector.visit(node)
    return collector.for2yield


def glue_let_chain(seq: List[Stmt]) -> Stmt:
    assert len(seq) > 0
    body = seq[-1]
    for stmt in reversed(seq[:-1]):
        if isinstance(stmt, LetStmt):
            if isinstance(body, LetStmt):
                body = LetStmt(stmt.bind_vars + body.bind_vars, stmt.bind_values + body.bind_values, body.body)
            else:
                body = LetStmt(stmt.bind_vars, stmt.bind_values, body)
        else:
            body = SeqStmt([stmt, body])
    return body
