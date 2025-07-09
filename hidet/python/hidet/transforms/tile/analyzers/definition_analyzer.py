from typing import Dict, Optional

from hidet.ir.expr import Var, Expr
from hidet.ir.func import Function
from hidet.ir.functors import IRVisitor
from hidet.ir.stmt import LetStmt
from hidet.ir.tile.stmt import PureForStmt


class VarDefinition:
    pass


class LetDefinition(VarDefinition):
    def __init__(self, let_stmt: LetStmt, idx: int):
        self.let_stmt: LetStmt = let_stmt
        self.idx: int = idx
        self.bind_var: Var = let_stmt.bind_vars[idx]
        self.bind_value: Expr = let_stmt.bind_values[idx]


class ForArgDefinition(VarDefinition):
    def __init__(self, for_stmt: PureForStmt, idx: int):
        self.for_stmt: PureForStmt = for_stmt
        self.idx: int = idx


class ForLetDefinition(VarDefinition):
    def __init__(self, for_stmt: PureForStmt, idx: int):
        self.for_stmt: PureForStmt = for_stmt
        self.idx: int = idx


class FuncParamDefinition(VarDefinition):
    def __init__(self, func: Function, idx: int):
        self.func: Function = func
        self.idx: int = idx


class DefinitionAnalyzer(IRVisitor):
    def __init__(self):
        super().__init__()
        self.definitions: Dict[Var, VarDefinition] = {}

    def analyze(self, node):
        self.definitions.clear()
        self.visit(node)
        return self.definitions

    def visit_Function(self, func: Function):
        for idx, param in enumerate(func.params):
            self.definitions[param] = FuncParamDefinition(func, idx)
        self.visit(func.body)

    def visit_LetStmt(self, stmt: LetStmt):
        for idx, bind_var in enumerate(stmt.bind_vars):
            self.definitions[bind_var] = LetDefinition(stmt, idx)
        self.visit(stmt.body)

    def visit_PureForStmt(self, stmt: PureForStmt):
        for idx, arg in enumerate(stmt.args):
            self.definitions[arg] = ForArgDefinition(stmt, idx)
        for idx, bind_var in enumerate(stmt.let_vars):
            self.definitions[bind_var] = ForLetDefinition(stmt, idx)
        self.visit(stmt.body)
        self.visit(stmt.let_body)


def analyze_definitions(node) -> Dict[Var, VarDefinition]:
    analyzer = DefinitionAnalyzer()
    analyzer.visit(node)
    return analyzer.definitions
