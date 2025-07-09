from typing import Dict

from hidet.ir import Function
from hidet.ir.expr import Var
from hidet.ir.functors import IRVisitor
from hidet.ir.stmt import LetStmt
from hidet.ir.tile.stmt import PureForStmt


class LevelAnalyzer(IRVisitor):
    def __init__(self):
        super().__init__()
        self.var2level: Dict[Var, int] = {}
        self.current_level: int = 0

    def define(self, v):
        assert v not in self.var2level
        self.var2level[v] = self.current_level

    def visit_Function(self, func: Function):
        self.current_level = 0
        for param in func.params:
            self.define(param)
        super().visit_Function(func)

    def visit_PureForStmt(self, stmt: PureForStmt):
        self.current_level += 1
        self.define(stmt.loop_var)
        for arg in stmt.args:
            self.define(arg)
        self.visit(stmt.body)
        self.current_level -= 1
        for let_arg in stmt.let_vars:
            self.define(let_arg)
        self.visit(stmt.let_body)

    def visit_LetStmt(self, stmt: LetStmt):
        for bind_var in stmt.bind_vars:
            self.define(bind_var)
        self.visit(stmt.body)
