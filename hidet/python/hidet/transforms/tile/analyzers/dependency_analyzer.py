from typing import List, Dict

from hidet.ir.expr import Expr
from hidet.ir.expr import Var
from hidet.ir.functors import IRVisitor
from hidet.ir.stmt import LetStmt
from hidet.ir.tile.stmt import PureForStmt


class DependencyAnalyzer(IRVisitor):
    def __init__(self):
        super().__init__()
        self.depends: Dict[Var, List[Var]] = {}

    def get_direct_depends(self, e: Expr):
        self.memo.clear()
        self.visit(e)
        return [v for v in self.memo if isinstance(v, Var)]

    def add_depends(self, user: Var, depends: List[Var]):
        if user not in self.depends:
            self.depends[user] = []
        for v in depends:
            if v not in self.depends[user]:
                self.depends[user].append(v)

    def visit_LetStmt(self, stmt: LetStmt):
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            self.add_depends(bind_var, self.get_direct_depends(bind_value))
        self.visit(stmt.body)

    def visit_PureForStmt(self, stmt: PureForStmt):
        for arg, value in zip(stmt.args, stmt.values):
            self.add_depends(arg, self.get_direct_depends(value))
        for let_var, arg in zip(stmt.let_vars, stmt.args):
            self.add_depends(let_var, self.get_direct_depends(arg))
        self.pure_for_stmts.append(stmt)
        self.visit(stmt.body)
        self.pure_for_stmts.pop()
        self.visit(stmt.let_body)
