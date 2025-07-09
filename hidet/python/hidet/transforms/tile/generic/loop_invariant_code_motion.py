"""
Move the loop invariant code out of the loop body.

More specifically, for each let statement in the loop body, if the used variables in the bind value are all defined
outside the loop, then move the let statement out of the loop body (right before the loop statement).

For example, the following code:
```
  x = zeors(10)
  for i in range(10):
    y = x + 1
    z = x * y
    w = z + i
    print(w)
```

will be transformed to:
```
  x = zeors(10)
  y = x + 1
  z = x * y
  for i in range(10):
    w = z + i
```

The steps to apply this pass are:
1. Collect the mapping from variables to the loops that define them.
2. For each variable, determine whether we need to move the variable out of the loop,
   if so, append the variable definition to the queue of the loop that stores the variables that should defined
   before the loop.
3. Move the variables according the queue in step 2.
"""
from typing import Dict, Optional, List, Tuple, Set, Type
from collections import defaultdict
from hidet.ir.expr import Var, Expr
from hidet.ir.stmt import LetStmt, ForStmt, DeclareStmt
from hidet.ir.func import Function
from hidet.ir.functors import IRVisitor, IRRewriter
from hidet.ir.tools import collect
from hidet.ir.tile.stmt import PureForStmt
from hidet.ir.tile.expr import CallTileOp, TileOp
from hidet.ir.tile.ops import Create
from hidet.transforms.base import TileFunctionPass


class VarToLoopAnalyzer(IRVisitor):
    def __init__(self):
        super().__init__()
        self.var2loop: Dict[Var, PureForStmt] = {}
        self.var2value: Dict[Var, Expr] = {}
        self.loop_parent: Dict[PureForStmt, Optional[PureForStmt]] = {}
        self.loop2level: Dict[PureForStmt, int] = {}

        self.sentinel: PureForStmt = object.__new__(PureForStmt)
        self.loop_stack: List[PureForStmt] = [self.sentinel]
        self.loop2level[self.sentinel] = 0

    def visit_Function(self, func: Function):
        for param in func.params:
            self.var2loop[param] = self.loop_stack[-1]
        self.visit(func.body)

    def visit_PureForStmt(self, stmt: PureForStmt):
        self.loop_parent[stmt] = self.loop_stack[-1]
        self.loop2level[stmt] = self.loop2level[self.loop_stack[-1]] + 1

        self.var2loop[stmt.loop_var] = stmt
        for arg in stmt.args:
            self.var2loop[arg] = stmt

        self.loop_stack.append(stmt)
        self.visit(stmt.body)
        self.loop_stack.pop()

        for let_var in stmt.let_vars:
            self.var2loop[let_var] = self.loop_stack[-1]
        self.visit(stmt.let_body)

    def visit_LetStmt(self, stmt: LetStmt):
        for idx, bind_var in enumerate(stmt.bind_vars):
            self.var2loop[bind_var] = self.loop_stack[-1]
            self.var2value[bind_var] = stmt.bind_values[idx]
        self.visit(stmt.body)

    def visit_ForStmt(self, stmt: ForStmt):
        raise RuntimeError('ForStmt should not appear in the tile-dialect.')

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        raise RuntimeError('ForStmt should not appear in the tile-dialect.')


class ApplyMovementRewriter(IRRewriter):
    def __init__(self, moving_vars: Set[Var], loop2queue: Dict[PureForStmt, List[Tuple[Var, Expr]]]):
        super().__init__()
        self.moving_vars: Set[Var] = moving_vars
        self.loop2queue: Dict[PureForStmt, List[Tuple[Var, Expr]]] = loop2queue

    def visit_PureForStmt(self, stmt: PureForStmt):
        if len(self.loop2queue[stmt]) > 0:
            bind_vars = []
            bind_values = []
            for var, expr in self.loop2queue[stmt]:
                bind_vars.append(var)
                bind_values.append(expr)
            return LetStmt(bind_vars, bind_values, super().visit_PureForStmt(stmt))
        else:
            return super().visit_PureForStmt(stmt)

    def visit_LetStmt(self, stmt: LetStmt):
        bind_vars = []
        bind_values = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            if bind_var in self.moving_vars:
                continue
            bind_vars.append(bind_var)
            bind_values.append(bind_value)
        if len(bind_vars) == 0:
            return self.visit(stmt.body)
        else:
            return LetStmt(bind_vars, bind_values, self.visit(stmt.body))


class LoopInvariantCodeMotionRewriter(IRRewriter):
    def __init__(self, allow: Optional[List[Type[TileOp]]] = None, disallow: Optional[List[Type[TileOp]]] = None):
        super().__init__()
        self.allow: Optional[List[Type[TileOp]]] = allow
        self.disallow: Optional[List[Type[TileOp]]] = disallow

    def check_move(self, value: Expr) -> bool:
        # sometimes, it is better to keep the variable in the loop body because it is cheap to compute.
        # moving the variable out of the loop body may increase the lifetime of the variable and
        # increase the register pressure. When self.allow is not None, we only move the variable out of the loop body
        # if the op is in the allow list. When self.disallow is not None, we only move the variable out of the loop body
        # if the op is not in the disallow list.

        if isinstance(value, CallTileOp):
            op: TileOp = value.op
            if self.allow is not None and not any(isinstance(op, op_type) for op_type in self.allow):
                return False
            if self.disallow is not None and any(isinstance(op, op_type) for op_type in self.disallow):
                return False
            return True
        else:
            # scalar value will always be moved out of the loop body
            return True

    def visit_Function(self, func: Function):
        if func.kind != 'cuda_tile':
            return func

        # step 1
        analyzer = VarToLoopAnalyzer()
        analyzer.visit(func)
        loop2level: Dict[PureForStmt, int] = analyzer.loop2level
        var2loop: Dict[Var, PureForStmt] = analyzer.var2loop
        var2value: Dict[Var, Expr] = analyzer.var2value
        loop_parent: Dict[PureForStmt, PureForStmt] = analyzer.loop_parent

        # step 2
        moving_vars: Set[Var] = set()
        loop2queue: Dict[PureForStmt, List[Tuple[Var, Expr]]] = defaultdict(list)

        for var, value in var2value.items():
            loop = var2loop[var]
            if loop is analyzer.sentinel:
                # it is defined outside any loop
                continue

            if not self.check_move(value):
                # do not move the variable.
                continue

            # find the maximum level of all used variables that are in the definition of the current variable
            used_vars: List[Var] = collect(value, node_types=[Var])
            if isinstance(value, CallTileOp) and isinstance(value.op, Create):
                for axis in value.op.axes:
                    if axis in used_vars:
                        used_vars.remove(axis)

            level = 0
            for used_var in used_vars:
                assert used_var in var2loop, used_var
                level = max(level, loop2level[var2loop[used_var]])

            # find the outermost loop that can be used to defines the current variable, such that the used variables
            # are defined before the current variable
            assert level <= loop2level[loop]
            if level == loop2level[loop]:
                # the current variable can not move out of the current loop
                continue

            # find the loop to define the current variable before
            while loop2level[loop_parent[loop]] > level:
                loop = loop_parent[loop]
            loop2queue[loop].append((var, value))
            moving_vars.add(var)

        # step 3
        if all(len(queue) == 0 for queue in loop2queue.values()):
            # no variable can be moved out of any loop
            return func
        else:
            rewriter = ApplyMovementRewriter(moving_vars, loop2queue)
        return rewriter.visit(func)


class LoopInvariantCodeMotionPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        return self.apply_transforms(
            node=func,
            transforms=[
                # do not move the "Create" op out of the loop body to reduce the register pressure
                LoopInvariantCodeMotionRewriter(disallow=[Create])
            ],
            repeat_limit=-1,
        )


def loop_invariant_code_motion_pass() -> TileFunctionPass:
    return LoopInvariantCodeMotionPass()
