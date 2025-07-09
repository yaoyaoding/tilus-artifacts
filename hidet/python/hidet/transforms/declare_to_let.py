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
"""
Convert DeclareStmt with initialized value to LetStmt if the declared variable satisfy the following conditions:
    1. has never been modified with AssignStmt statement, and
    2. has never been addressed with Address expression, and
    3. has never been referenced with Reference expression, and
    4. has never appeared in outputs of AsmStmt statement

"""
from typing import List, Dict
from collections import defaultdict

from hidet.ir import SeqStmt
from hidet.ir.expr import Expr, Var, Address, Reference
from hidet.ir.stmt import AssignStmt, DeclareStmt, LetStmt, Stmt, AsmStmt
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.tools import collect
from hidet.transforms.base import Pass, FunctionPass


class DeclareToLetRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.assigns: Dict[Var, int] = defaultdict(int)

    def update_assigns(self, node):
        for potential_usage in collect(node, (DeclareStmt, AssignStmt, AsmStmt, Address, Reference)):
            if isinstance(potential_usage, Stmt):
                stmt = potential_usage
                if isinstance(stmt, DeclareStmt):
                    if stmt.init is not None:
                        self.assigns[stmt.var] += 1
                elif isinstance(stmt, AssignStmt):
                    self.assigns[stmt.var] += 1
                elif isinstance(stmt, AsmStmt):
                    for output_expr in stmt.output_exprs:
                        if isinstance(output_expr, Var):
                            self.assigns[output_expr] += 1
                else:
                    assert False
            elif isinstance(potential_usage, Expr):
                expr = potential_usage
                if isinstance(expr, Address):
                    if isinstance(expr.expr, Var):
                        self.assigns[expr.expr] += 1
                elif isinstance(expr, Reference):
                    if isinstance(expr.expr, Var):
                        self.assigns[expr.expr] += 1
                else:
                    assert False
            else:
                assert False

    def __call__(self, node):
        self.update_assigns(node)
        return self.visit(node)

    def visit_SeqStmt(self, seq_stmt: SeqStmt):
        seq = [self.visit(stmt) for stmt in seq_stmt.seq]
        for i in range(len(seq) - 1, -1, -1):
            stmt = seq[i]
            if isinstance(stmt, DeclareStmt):
                if self.assigns[stmt.var] == 1 and stmt.init is not None:
                    let_stmt = LetStmt(bind_vars=[stmt.var], bind_values=[stmt.init], body=self.concat(seq[i + 1 :]))
                    seq = seq[:i] + [let_stmt]
        return self.concat(seq)

    def concat(self, seq: List[Stmt]):
        if len(seq) == 1:
            return seq[0]
        else:
            return SeqStmt(seq)


class UpliftLetBodyRewriter(IRRewriter):
    """
    Uplift the tail part of the stmt body of LetStmt that has not used the bind_vars

    let a = foo()
      use(a)
      let b = goo()
        use(b)

    =>

    let a = foo()
      use(a)
    let b = goo()
      use(b)
    """

    def flatten_seq(self, seq_stmt: SeqStmt) -> List[Stmt]:
        seq = []
        for stmt in seq_stmt.seq:
            if isinstance(stmt, SeqStmt):
                seq.extend(self.flatten_seq(stmt))
            else:
                seq.append(stmt)
        return seq

    def visit_LetStmt(self, stmt: LetStmt):
        body = stmt.body

        if isinstance(body, SeqStmt):
            seq: List[Stmt] = self.flatten_seq(body)

            uplifted_stmts = []
            while seq:
                s = seq.pop()
                s = self.visit(s)
                if all(bind_var not in self.memo for bind_var in stmt.bind_vars):
                    uplifted_stmts.append(s)
                else:
                    seq.append(s)
                    break
            if len(uplifted_stmts) > 0:
                stmts: List[Stmt] = [LetStmt(stmt.bind_vars, stmt.bind_values, SeqStmt(seq))]
                stmts.extend(list(reversed(uplifted_stmts)))
                return SeqStmt(stmts)
            else:
                return LetStmt(stmt.bind_vars, stmt.bind_values, SeqStmt(seq))
        else:
            return super().visit_LetStmt(stmt)


class DeclareToLetPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        return self.apply_transforms(func, [DeclareToLetRewriter(), UpliftLetBodyRewriter()])


def declare_to_let_pass() -> Pass:
    return DeclareToLetPass()
