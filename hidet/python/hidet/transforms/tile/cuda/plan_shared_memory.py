"""
Plan the shared memory allocation.

Problem:
The shared memory is a precious resource in GPU, we need to allocate shared memory for tensors in shared scope to
make the tensors with non-overlapping lifetime to share the same shared memory space.
In tile dialect, we have the tile operators that allocate shared memory. We will annotate the attribute
"global_offset" to specify the offset of the shared memory region in the whole shared memory space window.
This pass tries to find the best global offset for such operators to minimize the total shared memory usage
while making sure that the tensors with overlapping lifetime do not share the same shared memory space.

Solution:
1. Analyze the list of tile operators that require shared memory, and associate the tile variables in shared scope with
   the tile operators that allocate its shared memory.
   For example:
   ```
       1: let a = alloc_tensor([3, 8])
       2: let b = alloc_tensor([3, 16])
       3: let c = create([16, 16], scope=shared)
       4: let d = extract_slice(a, axis=0, index=0)
       5: let e = extract_slice(b, axis=0, index=0)
       6: let f = convert_layout(c, layout=block)  # will use shared memory as the temporary buffer for conversion
   ```
   We will have the following operators requiring shared memory:
       1: alloc_tensor([3, 8])
       2: alloc_tensor([3, 16])
       5: create([16, 16], scope=shared)
       6: convert_layout(e, layout=block)
   We will have the following association:
       {a, d}: alloc_tensor([3, 8])
       {b, e}: alloc_tensor([3, 16])
       {e}: create([16, 16], scope=shared)

2. Analyze the lifespan of each tile variable in shared scope as well as the tile operators that allocating shared
   memory.
   For the following program:
   ```
       1: let a = alloc_tensor([3, 8])
       2: let b = alloc_tensor([3, 16])
       3: let c = create([16, 16], scope=shared)
       4: let d = extract_slice(a, axis=0, index=0)
       5: let e = extract_slice(b, axis=0, index=0)
       6: let f = convert_layout(c, layout=block)  # will use shared memory as the temporary buffer for conversion
   ```
   we will record the clock for each tile operator calling:
   ```
       t 0:
          let a = alloc_tensor([3, 8])
       t 1:
          let b = alloc_tensor([3, 16])
       t 2:
          let c = create([16, 16], scope=shared)
       t 3:
          let d = extract_slice(b, axis=0, index=0)
       t 4:
          let e = extract_slice(a, axis=0, index=0)
       t 5:
          let f = convert_layout(c, layout=block)  # will use shared memory as the temporary buffer for conversion
       t 6
   ```
   and the lifespan of each tile variable in shared scope and the tile operators that allocating shared memory:
       a: [0, 3)
       b: [1, 4)
       c: [2, 6)
       d: [3, 4)
       e: [4, 5)
       f: [5, 6)
       alloc_tensor([3, 8]): [0, 1)
       alloc_tensor([3, 16]): [1, 2)
       create([16, 16], scope=shared): [2, 3)
       convert_layout(e, layout=block): [5, 6)
3. For each pair of tile operators that allocating shared memory, we check if any variables associated with them and
   have overlapping lifespan.
   Take the same example in step 2, we will have the following merged lifespan for each tile operator:
       op1: alloc_tensor([3, 8]): [0, 4) (merged from: [0, 1), [0, 3), [3, 4))
       op2: alloc_tensor([3, 16]): [1, 5) (merged from: [1, 2), [1, 4), [4, 5))
       op3: create([16, 16], scope=shared): [2, 6) (merged from: [2, 3), [2, 6))
       op4: convert_layout(e, layout=block): [5, 6) (merged from: [5, 6))
    We can see that the following pairs of tile operators have overlapping lifespan:
       op1, op2
       op1, op3
       op2, op3
       op3, op4

4. With the overlapping information, use a greedy algorithm to find the best global offset for each tile operator that
   allocating shared memory. The algorithm is as follows:
   4.1 Sort the tile operators in descending order of the shared memory size they requested.
   4.2 Iterate the tile operators in the sorted order, for each tile operator, we find the best window in shared memory
       that does not overlap with any tile operator that has been allocated shared memory.
       If we cannot find such a window, we raise an error.

   Taking the 4 operators in step 3 as an example. If they require the following shared memory size:
       op1: 200 bytes
       op2: 200 bytes
       op3: 300 bytes
       op4: 400 bytes
   We sort them in descending order of the shared memory size:
       op4, op3, op1, op2
   We will have the following allocation plan generation process:
       iter 0:
           op4: [0, 400)
       iter 1:
           op4: [0, 400)
           op3: [400, 700)   (we need to make sure op3 does not overlap with op4)
       iter 2:
           op4: [0, 400)
           op3: [400, 700)
           op1: [0, 200)    (op1 can overlap with op4, but not op3)
       iter 3:
           op4: [0, 400)
           op3: [400, 700)
           op1: [0, 200)
           op2: [200, 400)  (op2 can overlap with op4, but not op3 and op1)

    Finally, we will have a plan with 700 bytes allocated shared memory.
"""
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import logging
from hidet.ir.type import sizeof
from hidet.ir.expr import Var
from hidet.ir.stmt import LetStmt, Stmt, DeclareStmt
from hidet.ir.func import Function
from hidet.ir.functors import IRVisitor, IRRewriter
from hidet.ir.tools import TypeInfer
from hidet.ir.tile.type import TileType
from hidet.ir.tile.expr import CallTileOp, TileOp
from hidet.ir.tile.stmt import PureForStmt, YieldStmt
from hidet.ir.tile.ops.convert_layout import ConvertLayout
from hidet.ir.tile.ops.smem import AllocTensor, InsertSliceAsync, ExtractSlice
from hidet.ir.tile.ops.creation import Create
from hidet.transforms.base import TileFunctionPass
from hidet.transforms.tile import annotations
from hidet.transforms.tile.cuda.lower_ops.registry import get_tile_op_impl
from hidet.transforms.tile.exceptions import SharedMemoryPlanningError


logger = logging.getLogger(__name__)

# work around for passing this parameter to alloc_nbytes every time, which will make the code more verbose
_num_warps = 0


def alloc_nbytes(alloc: TileOp) -> int:
    impl_cls = get_tile_op_impl(alloc)
    impl = impl_cls(_num_warps)
    return impl.request_smem_nbytes(alloc)


class AssociateVisitor(IRVisitor):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()

        # all tile operators that require shared memory will be recorded in allocation list
        # include the AllocTensor op that allocates the shared memory, or operator like ConvertLayout op that
        # need shared memory as the temporary buffer
        self.allocations: List[TileOp] = []

        # if an operator will return the allocated shared memory, we also should track all the variables that
        # will hold the shared memory. With this information, we can know whether it is possible to allocate
        # the same space for two operators that require shared memory. Currently, we only consider the case
        # that each operator only holds the reference for at most a single operator's requested shared memory.
        self.var2alloc: Dict[Var, TileOp] = {}
        self.alloc2var: Dict[TileOp, List[Var]] = defaultdict(list)

    def associate(self, var: Var, op: TileOp):
        self.var2alloc[var] = op
        self.alloc2var[op].append(var)

    def visit_LetStmt(self, stmt: LetStmt):
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            assert isinstance(bind_var, Var)

            if not isinstance(bind_value, CallTileOp):
                continue

            op: TileOp = bind_value.op
            if alloc_nbytes(op) > 0:
                # this tile operator requires shared memory allocation
                self.allocations.append(op)

            if not (isinstance(bind_var.type, TileType) and bind_var.type.scope.is_shared()):
                continue

            # for all operators that manipulate the shared memory, we need to associate the variables with the
            # original tile operator that allocates the memory
            if isinstance(op, AllocTensor):
                self.associate(bind_var, op)
            elif isinstance(op, InsertSliceAsync):
                assert isinstance(op.dst, Var)
                self.associate(bind_var, self.var2alloc[op.dst])
            elif isinstance(op, ExtractSlice):
                assert isinstance(op.src, Var)
                self.associate(bind_var, self.var2alloc[op.src])
            elif isinstance(op, ConvertLayout):
                assert isinstance(op.x, Var)
                # op.x must be a shared tensor, otherwise this op will be resolved in ResolveConvertLayoutPass
                self.associate(bind_var, op)
            elif isinstance(op, Create):
                self.associate(bind_var, op)
            else:
                raise NotImplementedError(op.__class__.__name__)
        self.visit(stmt.body)

    def visit_PureForStmt(self, stmt: PureForStmt):
        for arg, let_var, value in zip(stmt.args, stmt.let_vars, stmt.values):
            if not (isinstance(arg.type, TileType) and arg.type.scope.is_shared()):
                continue
            assert isinstance(value, Var) and value in self.var2alloc
            self.var2alloc[arg] = self.var2alloc[value]
            self.var2alloc[let_var] = self.var2alloc[value]

        self.pure_for_stmts.append(stmt)
        self.visit(stmt.body)
        self.pure_for_stmts.pop()
        self.visit(stmt.let_body)

    def visit_YieldStmt(self, stmt: YieldStmt):
        for_stmt = self.pure_for_stmts[-1]
        for arg, value in zip(for_stmt.args, stmt.values):
            if not (isinstance(arg.type, TileType) and arg.type.scope.is_shared()):
                continue
            assert arg in self.var2alloc
            assert value in self.var2alloc


class LifeSpan:
    def __init__(self, left: int = int(1e9), right: int = int(-1e9)):
        self.left: int = left
        self.right: int = right

    def __str__(self):
        return [self.left, self.right].__str__()

    def expand(self, clock: int):
        self.left = min(self.left, clock)
        self.right = max(self.right, clock)

    def merge(self, other):
        self.left = min(self.left, other.left)
        self.right = max(self.right, other.right)

    def intersect_with(self, other) -> bool:
        return self.left <= other.right and other.left <= self.right


class LifeSpanAnalyzer(IRVisitor):
    def __init__(self):
        super().__init__(use_memo=False)
        self.var2lifespan: Dict[Var, LifeSpan] = {}
        self.op2lifespan: Dict[TileOp, LifeSpan] = {}
        self.clock: int = 0

        # if a variable is defined outside a loop while used inside it, we will make sure the lifespan of the variable
        # is at least the lifespan of the whole loop
        self.loop_stack: List[PureForStmt] = []
        self.loop2lifespan: Dict[PureForStmt, LifeSpan] = {}
        self.var2defined: Dict[Var, Optional[PureForStmt]] = {}
        self.var2loops: Dict[Var, Set[PureForStmt]] = {}

    def analyze(self, node):
        self.visit(node)

        # post processing
        for var, loops in self.var2loops.items():
            for loop in loops:
                self.var2lifespan[var].merge(self.loop2lifespan[loop])

    def visit(self, node):
        super().visit(node)
        if isinstance(node, Stmt):
            # after every statement, we enter the next clock
            self.clock += 1

    def define(self, v: Var):
        if not (isinstance(v.type, TileType) and v.type.scope.is_shared()):
            # only analyze the lifespan of shared variables
            return

        self.var2lifespan[v] = LifeSpan(self.clock, self.clock)
        self.var2loops[v] = set()
        self.var2defined[v] = self.loop_stack[-1] if len(self.loop_stack) > 0 else None

    def visit_Var(self, v: Var):
        if not (isinstance(v.type, TileType) and v.type.scope.is_shared()):
            # only analyze the lifespan of shared variables
            return

        self.var2lifespan[v].expand(self.clock)
        for loop in reversed(self.loop_stack):
            if loop is self.var2defined[v]:
                break
            self.var2loops[v].add(loop)

    def visit_CallTileOp(self, call: CallTileOp):
        self.op2lifespan[call.op] = LifeSpan(self.clock, self.clock)
        super().visit_CallTileOp(call)

    def visit_LetStmt(self, stmt: LetStmt):
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            self.define(bind_var)
            self.visit(bind_value)
            self.clock += 1

        self.visit(stmt.body)

    def visit_PureForStmt(self, stmt: PureForStmt):
        for arg, value in zip(stmt.args, stmt.values):
            self.define(arg)
            self.visit(value)
            self.clock += 1

        start_clock = self.clock
        self.loop_stack.append(stmt)
        self.visit(stmt.body)
        self.loop_stack.pop()
        self.loop2lifespan[stmt] = LifeSpan(start_clock, self.clock)

        for let_var in stmt.let_vars:
            self.define(let_var)
        self.clock += 1

        self.visit(stmt.let_body)

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        raise RuntimeError('SSA form is required')


class ApplyPlanRewriter(IRRewriter):
    def __init__(self, alloc2offset: Dict[TileOp, int], dynamic_smem_nbytes: int):
        super().__init__()
        self.alloc2offset: Dict[TileOp, int] = alloc2offset
        self.dynamic_smem_nbytes: int = dynamic_smem_nbytes

    def visit_Function(self, func: Function):
        func = super().visit_Function(func)
        if func.kind == 'cuda_tile':
            func.attrs['cuda.dynamic_smem_bytes'] = self.dynamic_smem_nbytes
        return func

    def visit_CallTileOp(self, call: CallTileOp):
        op = call.op

        if alloc_nbytes(op) > 0:
            if op not in self.alloc2offset:
                raise RuntimeError('Tile operator {} has requested shared memory but not allocated.'.format(op))
            op = op.reforward(args=op.args, annotations_update={annotations.global_offset: self.alloc2offset[op]})
        return CallTileOp(op)


Edges = Dict[TileOp, List[TileOp]]
Plan = Tuple[Dict[TileOp, int], int]  # {op: offset}, allocated


class PlanSharedMemoryPass(TileFunctionPass):
    @staticmethod
    def get_max_smem_size():
        from hidet.cuda.capability import capability

        return capability().sharedMemPerBlock

    def analyze_alloc_edges(
        self,
        allocations: List[TileOp],
        alloc2var: Dict[TileOp, List[Var]],
        var2lifespan: Dict[Var, LifeSpan],
        op2lifespan: Dict[TileOp, LifeSpan],
    ) -> Dict[TileOp, List[TileOp]]:
        # if (u, v) in edges, then u and v have overlap
        edges: Edges = defaultdict(list)
        for u in allocations:
            for v in allocations:
                if u is v:
                    continue
                u_span = op2lifespan[u]
                v_span = op2lifespan[v]
                for u_var in alloc2var[u]:
                    u_span.merge(var2lifespan[u_var])
                for v_var in alloc2var[v]:
                    v_span.merge(var2lifespan[v_var])
                if u_span.intersect_with(v_span) and v not in edges[u]:
                    edges[u].append(v)
                    edges[v].append(u)
        return edges

    def plan(self, allocations: List[TileOp], edges: Edges, max_nbytes: int, alignment=16) -> Plan:
        allocations: List[TileOp] = list(sorted(allocations, key=alloc_nbytes, reverse=True))
        plan: Dict[TileOp, int] = {}
        allocated: int = 0

        for u in allocations:
            # event: (offset, delta)
            events: List[Tuple[int, int]] = []
            for v in edges[u]:
                if v in plan:
                    aligned_nbytes = (alloc_nbytes(v) + alignment - 1) // alignment * alignment
                    events.append((plan[v], 1))
                    events.append((plan[v] + aligned_nbytes, -1))
            events.append((0, 0))
            events.append((max_nbytes, 0))
            events = sorted(events, key=lambda event: event[0])
            cnt = 0
            for i in range(len(events)):
                cnt += events[i][1]
                if cnt == 0 and i < len(events) - 1:
                    space = events[i + 1][0] - events[i][0]
                    if space >= alloc_nbytes(u):
                        plan[u] = events[i][0]
                        allocated = max(allocated, plan[u] + alloc_nbytes(u))
                        break
            else:
                lines = ['Cannot find a valid shared memory allocation plan.']
                for idx, v in enumerate(allocations):
                    lines.append(f'[{idx}] {v}: {alloc_nbytes(v)} bytes')
                for idx, v in enumerate(allocations):
                    conflicts = [allocations.index(v) for v in edges[v]]
                    if len(conflicts) > 0:
                        lines.append(f'[{idx}]: {conflicts}')
                raise SharedMemoryPlanningError('\n'.join(lines))
        return plan, allocated

    def process_tile_func(self, func: Function) -> Function:
        global _num_warps
        _num_warps = self.num_warps

        # step 1
        associate_visitor = AssociateVisitor()
        associate_visitor.visit(func)

        # step 2
        lifespan_analyzer = LifeSpanAnalyzer()
        lifespan_analyzer.analyze(func)

        # step 3
        alloc2var: Dict[TileOp, List[Var]] = associate_visitor.alloc2var
        var2lifespan: Dict[Var, LifeSpan] = lifespan_analyzer.var2lifespan
        op2lifetime: Dict[TileOp, LifeSpan] = lifespan_analyzer.op2lifespan
        allocations: List[TileOp] = associate_visitor.allocations
        edges: Edges = self.analyze_alloc_edges(allocations, alloc2var, var2lifespan, op2lifetime)

        # step 4
        plan: Dict[TileOp, int]
        allocated: int
        plan, allocated = self.plan(allocations, edges, max_nbytes=self.get_max_smem_size())

        # print the plan
        if logger.getEffectiveLevel() <= logging.DEBUG:
            from hidet.ir.tools.printer import IRPrinter

            # print the function
            printer = IRPrinter(inline_attrs=True)
            logger.debug(printer(func))

            # print the lifespan of operators that require allocation
            logger.debug('Operators that requested shared memory:')
            op2idx: Dict[TileOp, int] = {}
            for idx, op in enumerate(allocations):
                logger.debug(f'  [{idx}] {printer(op)}')
                logger.debug(f'      request {alloc_nbytes(op)} bytes')
                logger.debug(f'      lifespan {lifespan_analyzer.op2lifespan[op]}')
                op2idx[op] = idx

            # print the edges
            logger.debug('Edges:')
            for u in allocations:
                edge_list = ' '.join([str(op2idx[v]) for v in edges[u]])
                logger.debug(f'  {op2idx[u]}: {edge_list}')

            # print the allocations
            logger.debug(f'Allocated {allocated} nbytes, plan (offset):')
            for u in allocations:
                logger.debug(f'  {op2idx[u]}: {plan[u]}')

        rewriter = ApplyPlanRewriter(plan, allocated)
        func = rewriter(func)

        return func


def plan_shared_memory_pass() -> TileFunctionPass:
    return PlanSharedMemoryPass()
