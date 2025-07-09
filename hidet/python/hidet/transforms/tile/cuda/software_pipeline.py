import logging
from collections import defaultdict
from typing import List, Dict, Set, Optional, Tuple, Union

from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, less_than
from hidet.ir.expr import Var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter, IRVisitor
from hidet.ir.stmt import LetStmt, Stmt, EvaluateStmt, SeqStmt
from hidet.ir.tile.expr import TileOp, CallTileOp
from hidet.ir.tile.layout import TileLayout, MmaDotOperandLayout
from hidet.ir.tile.layout import repeat
from hidet.ir.tile.ops import AllocTensor, Load, InsertSliceAsync, AsyncCommitGroup, ExtractSlice, AsyncWait
from hidet.ir.tile.ops import ConvertLayout, Create
from hidet.ir.tile.ops.arthimatic import LogicalAnd
from hidet.ir.tile.stmt import PureForStmt, YieldStmt
from hidet.ir.tile.type import TileType, TileScope
from hidet.ir.tools import TypeInfer, rewrite
from hidet.transforms.base import TileFunctionPass
from hidet.transforms.tile.analyzers import DefinitionAnalyzer, VarDefinition, LetDefinition, DependencyAnalyzer
from hidet.transforms.tile.analyzers import UsageAnalyzer, VarUsage
from hidet.transforms.tile.cuda.remove_layout_convert import FoldConvertLayoutTransform, DeadCodeEliminationRewriter
from hidet.transforms.tile.generic.canonicalize_to_ssa import canonicalize_to_ssa

logger = logging.getLogger(__name__)

"""
Apply software pipeline to the load operators in a loop:

  for i in range(n) with other_args=..., ptr_args=inits:
      ...
      ptr = ptr_expr(ptr_args)
      x = load(ptr)
      ...
      yield iter_expr(ptr_args), ...

convert to:

  stages = 3

  buf = alloc_tensor([stages, ...])   # ... is the shape of x

  ptr_args0 = ptr_inits
  ptr0 = ptr_expr(ptr_args0)
  insert_slice_async(ptr0, buf, mask, other, axis=0, index=0)

  ptr_args1 = iter_expr(ptr_args0)
  ptr1 = ptr_expr(ptr_args1)
  insert_slice_async(ptr1, buf, mask, other, axis=0, index=1)

  async_wait(stages - 2)
  s0 = extract_slice(buf, axis=0, index=0)
  ptr_args2 = iter_expr(ptr_args1)

  for i in range(n) with (
     other_args=...,
     ptr_args = ptr_args2,
     s = s0,
     insert_index = 2,
     extract_index = 1,
  ):
     ...
     x = convert_layout(s, x's layout)
     ...
     ptr = ptr_expr(ptr_args)
     insert_slice_async(ptr, buf, mask=mask && i + stages - 1 < n, other, axis=0, index=insert_index)
     ptr_args' = iter_expr(ptr_args)
     async_wait(staged - 2)
     s' = extract_slice(buf, axis=0, index=extract_index)
     insert_index' = (insert_index + 1) % stages
     extract_index' = (extract_index + 1) % stages
     yield ..., iter_expr(ptr_args), s', insert_index', extract_index'
  async_wait(0)

 Let depends(loop, x) be the loop's arguments that are depended by the computation of x

 To follow several steps to achieve above transformation:
 1. Find the pairs of (loop, (load_1, load_2, ...)) where the loads are directly in the loop body without nested loops.
 2. For each (loop, loads) pair, do the following:
    2.1. Let ptr_i be the pointer argument of the load_i. Find ptr_args = union{depends(loop, ptr_i) for all load_i}.
    2.2. Repeat ptr_args = depends(loop, ptr_args) until ptr_args is stable.
    2.3. Right before the loop, creates the shared memory and rematerialize the computation of ptr_args
            buf = alloc_tensor([stages, ...])   # ... is the shape of x
            ... (see above example)
    2.4. Update loop arguments to include ptr_args, s, insert_index, extract_index
    2.5. Replace the load(...) with convert_layout(s)
    2.6. At the end of the loop body, insert the code like:
            ptr = ptr_expr(ptr_args)
            insert_slice_async(ptr, buf, mask=mask && i < n, other, axis=0, index=insert_index)
            ptr_args' = iter_expr(ptr_args)
            async_wait(staged - 2)
            s' = extract_slice(buf, axis=0, index=extract_index)
            insert_index' = (insert_index + 1) % stages
            extract_index' = (extract_index + 1) % stages
    2.7. Update yield statement to
            yield ..., iter_expr(ptr_args), s', insert_index', extract_index'
    2.8. Add async_wait(0) after the loop body
"""


class DetectLoadInLoopVisitor(IRVisitor):
    def __init__(self):
        super().__init__()
        self.loop2loads: Dict[PureForStmt, List[Load]] = defaultdict(list)
        self.loop2yields: Dict[PureForStmt, List[YieldStmt]] = defaultdict(list)
        self.load2usage: Dict[Load, VarUsage] = {}

        self._load2var: Dict[Load, Var] = {}

    def visit_Function(self, func: Function):
        self.loop2loads.clear()
        self.loop2yields.clear()
        super().visit_Function(func)

        self.load2usage.clear()
        usage_analyzer = UsageAnalyzer()
        usage_analyzer.visit(func)
        for loads in self.loop2loads.values():
            for load in loads:
                self.load2usage[load] = usage_analyzer.usages[self._load2var[load]]

    def visit_LetStmt(self, stmt: LetStmt):
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            if isinstance(bind_value, CallTileOp) and isinstance(bind_value.op, Load):
                self._load2var[bind_value.op] = bind_var
            self.visit(bind_value)
        self.visit(stmt.body)

    def visit_Load(self, e: Load):
        if len(self.pure_for_stmts) > 0:
            loop = self.pure_for_stmts[-1]
            self.loop2loads[loop].append(e)
        super().visit_Load(e)

    def visit_YieldStmt(self, stmt: YieldStmt):
        if len(self.pure_for_stmts) > 0:
            loop = self.pure_for_stmts[-1]
            self.loop2yields[loop].append(stmt)
        super().visit_YieldStmt(stmt)


class Rematerializer:
    def __init__(self, args: List[Var], bind_vars: List[Var], bind_values: List[Expr], results: List[Var]):
        super().__init__()
        self.args: List[Var] = args
        self.bind_vars: List[Var] = bind_vars
        self.bind_values: List[Expr] = bind_values
        self.results: List[Var] = results

    def __str__(self):
        from hidet.ir.tools.printer import IRPrinter, Doc, NewLine, doc_join

        printer = IRPrinter()
        args_doc = Doc()
        for arg in self.args:
            args_doc += NewLine() + printer(arg) + ': ' + printer(arg.type)
        body = Doc()
        for bind_var, bind_value in zip(self.bind_vars, self.bind_values):
            body += NewLine() + printer(bind_var) + ': ' + printer(bind_var.type) + ' = ' + printer(bind_value)
        doc = Doc()
        doc += NewLine() + 'args:'
        doc += args_doc.indent()
        if len(self.bind_vars) > 0:
            doc += NewLine() + 'body:'
            doc += body.indent()
        doc += NewLine() + 'return ' + printer(self.results)
        attrs = [b + ' = ' + a for a, b in printer.attributes.items()]
        if len(attrs) > 0:
            doc = NewLine() + doc_join(attrs, NewLine()) + doc
        return str(doc)

    @staticmethod
    def create(loop: PureForStmt, depends: Dict[Var, List[Var]], args: List[Var], values: List[Expr]):
        given_values = values
        values: List[Var] = [v for v in given_values if isinstance(v, Var)]
        assert len(values) == len(given_values), "Expected all values to be Var"

        definitions: Dict[Var, VarDefinition] = DefinitionAnalyzer().analyze(loop.body)
        bind_vars: List[Var] = []
        bind_values: List[Expr] = []

        # bfs starting from the given values to its dependencies to get the computations from for_args and external
        # variables (outside the loop) to the values
        visited: Set[Var] = set(values)
        queue: List[Var] = values.copy()

        while queue:
            u = queue.pop(0)

            if u not in definitions:
                # external variable defined outside the for loop, keep it as is
                continue

            # for variable defined inside the for loop
            definition = definitions[u]
            if not isinstance(definition, LetDefinition):
                raise RuntimeError(f"Expected LetDefinition, got {definition}")
            bind_vars.append(definition.bind_var)
            bind_values.append(definition.bind_value)

            for v in depends[u]:
                if v in visited:
                    # already in the computation (bind_vars, bind_values), skip
                    continue
                visited.add(v)
                queue.append(v)

        # inplace reverse the order of bind_vars and bind_values
        bind_vars.reverse()
        bind_values.reverse()
        return Rematerializer(args, bind_vars, bind_values, values)

    def rematerialize(
        self, updated_args: List[Var], extra_remap: Dict[Var, Expr] = None
    ) -> Tuple[List[Var], List[Expr], List[Var]]:
        bind_vars = []
        bind_values = []
        assert len(self.args) == len(updated_args)
        remap: Dict[Var, Expr] = {a: b for a, b in zip(self.args, updated_args)}
        if extra_remap is not None:
            remap.update(extra_remap)
        for bind_var, bind_value in zip(self.bind_vars, self.bind_values):
            updated_bind_value = rewrite(bind_value, remap)
            updated_bind_var = Var(bind_var.hint, bind_var.type)
            remap[bind_var] = updated_bind_var
            bind_vars.append(updated_bind_var)
            bind_values.append(updated_bind_value)
        results = [remap[r] if r in remap else r for r in self.results]
        return bind_vars, bind_values, results


class LoopArgs:
    def __init__(self, original, extra_indices, loaded_tiles, buffers, load_dependent_args, arg2value):
        self.original: List[Var] = original
        self.extra_indices: List[Var] = extra_indices
        self.loaded_tiles: List[Var] = loaded_tiles
        self.buffers: List[Var] = buffers
        self.load_dependent_args: List[Var] = load_dependent_args

        self.arg2value: Dict[Var, Expr] = arg2value

    def concatenated(self) -> List[Var]:
        return self.original + self.extra_indices + self.loaded_tiles + self.buffers + self.load_dependent_args

    def insert_index(self) -> Var:
        return self.extra_indices[0]

    def extract_index(self) -> Var:
        return self.extra_indices[1]


class RematerializePrefetchResult:
    def __init__(self, current_for_args, tile_vars, buffer_vars):
        self.current_for_args: List[Var] = current_for_args  # the arguments of the current loop
        self.tile_vars: List[Var] = tile_vars  # the first tile of each load
        self.buffer_vars: List[Var] = buffer_vars  # the shared memory buffers for each load


class SoftwarePipelineRewriter(IRRewriter):
    def __init__(self, loop, yield_stmt, loads, load2usage, dependency_graph, num_stages=3):
        super().__init__()
        self.loop: PureForStmt = loop
        self.yield_stmt: YieldStmt = yield_stmt
        self.loads: List[Load] = loads
        self.load2usage: Dict[Load, VarUsage] = load2usage
        self.dependency_graph: Dict[Var, List[Var]] = dependency_graph
        self.num_stages: int = num_stages

        self.type_infer = TypeInfer()

        # the information that will be used by Load and YieldStmt visitors, filled by PureForStmt visitor
        self.load_args: List[Var] = self.get_load_args()  # step 2.1
        self.load_dependent_args: List[Var] = self.get_load_dependent_args()  # step 2.2
        self.yield_load_dependent_values: List[Expr] = [
            self.yield_stmt.values[self.loop.args.index(arg)] for arg in self.load_dependent_args
        ]
        dep = self.dependency_graph
        self.load_args_remat: Rematerializer = Rematerializer.create(
            loop=self.loop, depends=dep, args=self.load_dependent_args, values=self.load_args
        )
        self.load_dependent_args_remat: Rematerializer = Rematerializer.create(
            loop=self.loop, depends=dep, args=self.load_dependent_args, values=self.yield_load_dependent_values
        )
        self.updated_args: Optional[LoopArgs] = None
        self.remat_result: Optional[RematerializePrefetchResult] = None

        logger.debug('load_args_remat: %s', self.load_args_remat)
        logger.debug('load_dependent_args_remat: %s', self.load_dependent_args_remat)

    def depends(self, users: List[Var]):
        # find all the variables that are the dependencies of users
        stack: List[Var] = list(users)
        visited: Set[Var] = set(users)
        while len(stack) > 0:
            u = stack.pop()
            if u not in self.dependency_graph:
                # variables without any dependency like loop variable
                continue
            for v in self.dependency_graph[u]:
                if v not in visited:
                    stack.append(v)
                    visited.add(v)
        return list(visited)

    def get_load_args(self) -> List[Var]:
        """step 2.1: find the arguments of the loads"""
        load_args: Set[Var] = set()
        for load in self.loads:
            assert isinstance(load.ptr, Var)
            load_args.add(load.ptr)
            if load.mask is not None:
                assert isinstance(load.mask, Var)
                load_args.add(load.mask)
            if load.other is not None:
                assert isinstance(load.other, Var)
                load_args.add(load.other)
        return list(load_args)

    def get_load_dependent_args(self) -> List[Var]:
        """step 2.2: find the self-contained set of arguments to compute load args as well as themselves"""
        load_args = self.load_args
        load_dependent_args = load_args
        while True:
            orig_num = len(load_dependent_args)
            load_dependent_args = [v for v in self.depends(users=load_dependent_args) if v in self.loop.args]
            yielded_values: List[Expr] = [self.yield_stmt.values[self.loop.args.index(v)] for v in load_dependent_args]
            yielded_vars: List[Var] = [v for v in yielded_values if isinstance(v, Var)]
            assert len(yielded_vars) == len(yielded_values)
            for v in self.depends(users=yielded_vars):
                if v in self.loop.args and v not in load_dependent_args:
                    assert isinstance(v, Var)
                    load_dependent_args.append(v)
            if len(load_dependent_args) == orig_num:
                # converged to a self-contained set of arguments to compute themselves during loop iteration
                break
        return load_dependent_args

    def generate_slice_layout(self, load: Load, slice_shape: List[int]) -> TileLayout:
        usage: VarUsage = self.load2usage[load]

        # if the result of load operator is used in the following way,
        #  a = load(...)
        #  b = convert_layout(a, layout=MmaDotOperandLayout(...))
        # then we can use MmaDotOperandSharedLayout instead of the default row major layout
        if len(usage.let_usages) > 0 and any(isinstance(u.op, ConvertLayout) for u in usage.let_usages):
            cvt_layout_op = next(u.op for u in usage.let_usages if isinstance(u.op, ConvertLayout))
            assert isinstance(cvt_layout_op, ConvertLayout)
            cvt_layout: TileLayout = cvt_layout_op.layout
            if isinstance(cvt_layout, MmaDotOperandLayout):
                return cvt_layout.shared_layout()

        # default row major layout
        return repeat(*slice_shape)

    def rematerialized_prefetch(self) -> Tuple[List[Stmt], RematerializePrefetchResult]:
        stmts = []

        # allocate shared memory buffers
        load_types: List[TileType] = [self.type_infer(load.make_call()) for load in self.loads]
        buffer_vars: List[Var] = []
        slice_layouts: List[TileLayout] = []
        for load, load_type in zip(self.loads, load_types):
            # buffer shape
            shape = [self.num_stages] + load_type.shape

            # buffer layout
            slice_layout = self.generate_slice_layout(load, slice_shape=load_type.shape)
            slice_layouts.append(slice_layout)
            layout = repeat(*([self.num_stages] + [1 for _ in load_type.shape])) * slice_layout.expand(dim=0)

            # declare the buffer variable
            buffer_type = TileType(load_type.type, shape, layout=layout, scope=TileScope.Shared)
            buffer_var = Var('smem', type=buffer_type)
            buffer_vars.append(buffer_var)
            stmts.append(LetStmt(buffer_var, AllocTensor(dtype=load_type.type, shape=shape, layout=layout).make_call()))

        # construct the rematerializer for for_args calculation
        load_args = self.get_load_args()

        # rematerialize the load args and loop args
        arg2init: Dict[Var, Var] = {arg: value for arg, value in zip(self.loop.args, self.loop.values)}
        current_for_args: List[Var] = [arg2init[arg] for arg in self.load_dependent_args]
        for i in range(self.num_stages - 1):
            # rematerialize the load arguments computations
            bind_vars, bind_values, remat_load_args = self.load_args_remat.rematerialize(
                current_for_args, extra_remap={self.loop.loop_var: int32(i)}
            )
            if len(bind_vars) > 0:
                stmts.append(LetStmt(bind_vars, bind_values))

            # rematerialize the load operations
            load_arg_map: Dict[Expr, Var] = {a: b for a, b in zip(load_args, remat_load_args)}
            for idx, load in enumerate(self.loads):
                # calculate the ptr, mask, and other arguments
                ptr = load_arg_map[load.ptr]
                mask = load_arg_map[load.mask] if load.mask is not None else None
                other = load_arg_map[load.other] if load.other is not None else None
                buf_var = buffer_vars[idx]
                op = InsertSliceAsync(ptr=ptr, dst=buf_var, index=int32(i), mask=mask, other=other, axis=0)
                new_buf_var = Var(buf_var.hint, type=buf_var.type)
                buffer_vars[idx] = new_buf_var
                stmts.append(LetStmt(new_buf_var, op.make_call()))
            stmts.append(AsyncCommitGroup())

            # rematerialize the loop arguments computations
            bind_vars, bind_values, current_for_args = self.load_dependent_args_remat.rematerialize(
                current_for_args, extra_remap={self.loop.loop_var: int32(i)}
            )
            if len(bind_vars) > 0:
                stmts.append(LetStmt(bind_vars, bind_values))

        # extract the first stage
        stmts.append(AsyncWait(self.num_stages - 2))
        tile_vars: List[Var] = []
        for idx, (buf_var, slice_layout) in enumerate(zip(buffer_vars, slice_layouts)):
            op = ExtractSlice(buf_var, start=int32(0), axis=0, extent=1, layout=slice_layout)
            tile_var = Var('ext_slice', type=self.type_infer(op.make_call()))
            stmts.append(LetStmt(tile_var, op.make_call()))
            tile_vars.append(tile_var)

        result = RematerializePrefetchResult(
            current_for_args=current_for_args, tile_vars=tile_vars, buffer_vars=buffer_vars
        )
        return stmts, result

    def update_loop_args_values(self, stmt: PureForStmt, remat_result: RematerializePrefetchResult):
        tile_vars: List[Var] = remat_result.tile_vars
        buffers_vars: List[Var] = remat_result.buffer_vars
        load_dependent_args_vals: List[Var] = remat_result.current_for_args
        arg2value: Dict[Var, Expr] = {}

        # prepare the new loop args and init values
        original: List[Var] = stmt.args.copy()
        arg2value.update({arg: value for arg, value in zip(original, stmt.values)})

        # insert_index and extract_index
        extra_indices: List[Var] = [Var('insert_index', type=int32), Var('extract_index', type=int32)]
        values = [int32(self.num_stages - 1), 1]
        arg2value.update({arg: value for arg, value in zip(extra_indices, values)})

        # extracted slices
        loaded_tiles: List[Var] = [Var(tile_var.hint, type=tile_var.type) for tile_var in tile_vars]
        arg2value.update({arg: value for arg, value in zip(loaded_tiles, tile_vars)})

        # shared memory buffers
        buffers: List[Var] = [Var(buffer_var.hint, type=buffer_var.type) for buffer_var in buffers_vars]
        arg2value.update({arg: value for arg, value in zip(buffers, buffers_vars)})

        # extra loop args used to compute load args
        load_dependent_args: List[Var] = [Var(arg.hint, type=arg.type) for arg in load_dependent_args_vals]
        arg2value.update({arg: value for arg, value in zip(load_dependent_args, load_dependent_args_vals)})

        self.updated_args = LoopArgs(
            original=original,
            extra_indices=extra_indices,
            loaded_tiles=loaded_tiles,
            buffers=buffers,
            load_dependent_args=load_dependent_args,
            arg2value=arg2value,
        )

    def concat_stmts(self, stmts: List[Union[Stmt, Expr, TileOp]]) -> Stmt:
        # concatenate the stmts
        body = stmts.pop()
        for s in reversed(stmts):
            if isinstance(s, TileOp):
                s = s.make_call()
            if isinstance(s, Expr):
                s = EvaluateStmt(s)
            if isinstance(s, LetStmt) and s.body is None:
                if isinstance(body, LetStmt):
                    body = LetStmt(s.bind_vars + body.bind_vars, s.bind_values + body.bind_values, body.body)
                else:
                    body = LetStmt(s.bind_vars, s.bind_values, body)
            elif isinstance(s, PureForStmt) and s.let_body is None:
                s.let_body = body
                body = s
            else:
                body = SeqStmt([s, body])
        return body

    def visit_PureForStmt(self, stmt: PureForStmt):
        if stmt is not self.loop:
            return super().visit_PureForStmt(stmt)

        stmts: List[Union[Stmt, Expr, TileOp]] = []

        # step 2.3: hoist the loading logic out of the loop body
        new_stmts, remat_result = self.rematerialized_prefetch()
        stmts.extend(new_stmts)
        self.remat_result = remat_result

        self.update_loop_args_values(stmt, remat_result)
        loop_args: List[Var] = self.updated_args.concatenated()
        loop_values: List[Expr] = [self.updated_args.arg2value[arg] for arg in loop_args]

        # step 2.4 to 2.7 in separate visit methods
        self.pure_for_stmts.append(stmt)
        body = self.visit(stmt.body)
        self.pure_for_stmts.pop()

        # update let vars
        let_vars = stmt.let_vars.copy()
        for idx in range(len(stmt.args), len(loop_args)):
            let_vars.append(Var(loop_args[idx].hint, type=loop_args[idx].type))

        # step 2.8
        for_stmt = PureForStmt(
            args=loop_args,
            values=loop_values,
            loop_var=stmt.loop_var,
            extent=stmt.extent,
            body=body,
            let_vars=let_vars,
            let_body=None,
        )
        stmts.append(for_stmt)
        stmts.append(AsyncWait(int32(0)))
        let_body = self.visit(stmt.let_body)
        stmts.append(let_body)

        # so far, we have finished the software pipelining
        return self.concat_stmts(stmts)

    def visit_Load(self, e: Load):
        if e not in self.loads:
            return super().visit_Load(e)
        # step 2.5: replace the load(...) with convert_layout(s, load's layout)
        idx = self.loads.index(e)
        ptr_type: TileType = self.type_infer(e.ptr)
        cvt = ConvertLayout(x=self.updated_args.loaded_tiles[idx], layout=ptr_type.layout)
        return cvt

    def visit_YieldStmt(self, stmt: YieldStmt):
        if self.pure_for_stmts[-1] is not self.loop:
            return super().visit_YieldStmt(stmt)

        stmts: List[Union[Stmt, Expr, TileOp]] = []
        arg2yield: Dict[Var, Expr] = {}

        # 2.6: update the end of loop body
        for orig_arg, orig_value in zip(self.loop.args, stmt.values):
            arg2yield[orig_arg] = orig_value

        # rematerialize the load arguments
        bind_vars, bind_values, remat_load_args = self.load_args_remat.rematerialize(
            self.updated_args.load_dependent_args,
            extra_remap={self.loop.loop_var: self.loop.loop_var + self.num_stages - 1},
        )
        load_arg_map: Dict[Expr, Var] = {a: b for a, b in zip(self.load_args, remat_load_args)}
        if len(bind_vars) > 0:
            stmts.append(LetStmt(bind_vars, bind_values))

        # load the next tile asynchronously for each load
        for idx, load in enumerate(self.loads):
            # calculate the ptr, mask, and other arguments
            ptr = load_arg_map[load.ptr]
            mask = load_arg_map[load.mask] if load.mask is not None else None
            other = load_arg_map[load.other] if load.other is not None else None
            assert isinstance(ptr.type, TileType)
            extra_mask = Create.from_compute(
                shape=ptr.type.shape,
                f_compute=lambda *indices: less_than(self.loop.loop_var + self.num_stages - 1, self.loop.extent),
                layout=ptr.type.layout,
            ).make_call()
            if mask is None:
                mask = extra_mask
            else:
                mask = LogicalAnd(mask, extra_mask).make_call()
            buf_var = self.updated_args.buffers[idx]
            op = InsertSliceAsync(
                ptr=ptr, dst=buf_var, index=self.updated_args.insert_index(), mask=mask, other=other, axis=0
            )
            updated_buf_var = Var(buf_var.hint, type=buf_var.type)
            stmts.append(LetStmt([updated_buf_var], [op.make_call()]))
            arg2yield[buf_var] = updated_buf_var
        stmts.append(AsyncCommitGroup())

        # rematerialize the load dependent for-loop arguments
        bind_vars, bind_values, remat_load_dependent_args = self.load_dependent_args_remat.rematerialize(
            self.updated_args.load_dependent_args,
            extra_remap={self.loop.loop_var: self.loop.loop_var + self.num_stages - 1},
        )
        if len(bind_vars) > 0:
            stmts.append(LetStmt(bind_vars, bind_values))
        arg2yield.update({a: b for a, b in zip(self.updated_args.load_dependent_args, remat_load_dependent_args)})

        # sync and extract the next tile for each load
        stmts.append(AsyncWait(self.num_stages - 2).make_call())
        for idx, tile in enumerate(self.updated_args.loaded_tiles):
            buf_var = arg2yield[self.updated_args.buffers[idx]]
            new_tile = Var(tile.hint, type=tile.type)
            assert isinstance(new_tile.type, TileType)
            op = ExtractSlice(buf_var, self.updated_args.extract_index(), axis=0, extent=1, layout=new_tile.type.layout)
            stmts.append(LetStmt([new_tile], [op.make_call()]))
            arg2yield[tile] = new_tile

        # update indices
        for extra_index in self.updated_args.extra_indices:
            updated_var = Var(extra_index.hint, type=extra_index.type)
            updated_value = (extra_index + 1) % self.num_stages
            stmts.append(LetStmt([updated_var], [updated_value]))
            arg2yield[extra_index] = updated_var

        yield_values = [arg2yield[arg] for arg in self.updated_args.concatenated()]
        stmts.append(YieldStmt(yield_values))

        return self.concat_stmts(stmts)


class SoftwarePipelinePass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        pipe_stages: int = func.attrs.pop('cuda.opt.pipe_stages', 3)

        # step 1: find all the (loop, loads) pairs
        loop_loads_detector = DetectLoadInLoopVisitor()
        loop_loads_detector.visit(func)
        loop2loads = loop_loads_detector.loop2loads
        loop2yields = loop_loads_detector.loop2yields
        load2usage = loop_loads_detector.load2usage

        # step 2: for each pair (loop, loads), rewrite the loop
        for loop, loads in loop2loads.items():
            if len(loop2yields[loop]) != 1:
                # we expect there is only one yield statement in the loop to apply software pipelining
                continue
            yield_stmt = loop2yields[loop][0]

            # analyze dependency graph: depends[u] = [v | u depends on v directly]
            dependency_analyzer = DependencyAnalyzer()
            dependency_analyzer.visit(func)
            dependency_graph = dependency_analyzer.depends

            # step 2.1 to 2.8: rewrite the loop
            rewriter = SoftwarePipelineRewriter(loop, yield_stmt, loads, load2usage, dependency_graph, pipe_stages)
            func = rewriter(func)

        # finally, remove nested convert_layout
        func = canonicalize_to_ssa(func)
        func = FoldConvertLayoutTransform()(func)
        func = DeadCodeEliminationRewriter()(func)
        return func


def software_pipeline_pass() -> TileFunctionPass:
    return SoftwarePipelinePass()
