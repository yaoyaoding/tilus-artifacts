from typing import Dict, List, Union
from hidet.ir.expr import Var, Expr
from hidet.ir.stmt import LetStmt
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.tile.expr import CallTileOp
from hidet.ir.tile.ops import Create, Broadcast, ExpandDims, BinaryTileOp
from hidet.ir.tile.type import TileType
from hidet.ir.tools import TypeInfer
from hidet.utils import same_list
from hidet.transforms.base import TileFunctionPass
from .dead_code_elimination import DeadCodeEliminationRewriter


class SimplifyTileCreationRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()
        self.var2construct: Dict[Var, Create] = {}

    def visit_LetStmt(self, stmt):
        bind_values: List[Expr] = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            bind_value = self.visit(bind_value)
            if isinstance(bind_value, CallTileOp) and isinstance(bind_value.op, Create):
                self.var2construct[bind_var] = bind_value.op
            elif isinstance(bind_value, Var) and bind_value in self.var2construct:
                self.var2construct[bind_var] = self.var2construct[bind_value]
            bind_values.append(bind_value)
        body = self.visit(stmt.body)
        if same_list(bind_values, stmt.bind_values) and body is stmt.body:
            return stmt
        else:
            return LetStmt(stmt.bind_vars, bind_values, body)

    def visit_Broadcast(self, e: Broadcast):
        if e.x in self.var2construct:
            assert isinstance(e.x, Var)
            x: Create = self.var2construct[e.x]

            def f_compute(indices: List[Var]) -> Expr:
                assert len(indices) == len(x.shape)
                x_indices: List[Union[Expr, int]] = []
                for idx, (x_extent, _) in enumerate(zip(x.shape, e.shape)):
                    if x_extent == 1:
                        x_indices.append(0)
                    else:
                        x_indices.append(indices[idx])
                return x[x_indices]

            return Create.from_compute(shape=e.shape, f_compute=f_compute)
        else:
            return super().visit_Broadcast(e)

    def visit_ExpandDims(self, e: ExpandDims):
        if e.x in self.var2construct:
            assert isinstance(e.x, Var)
            x: Create = self.var2construct[e.x]
            y_type: TileType = self.type_infer(e.make_call())

            def f_compute(indices: List[Var]) -> Expr:
                assert len(indices) == len(x.shape) + 1
                x_indices = indices[: e.axis] + indices[e.axis + 1 :]
                return x[x_indices]

            return Create.from_compute(shape=y_type.shape, f_compute=f_compute)
        else:
            return super().visit_ExpandDims(e)

    def visit_BinaryTileOp(self, e: BinaryTileOp):
        if e.x in self.var2construct and e.y in self.var2construct:
            assert isinstance(e.x, Var) and isinstance(e.y, Var)
            x: Create = self.var2construct[e.x]
            y: Create = self.var2construct[e.y]
            shape = x.shape
            return Create.from_compute(shape=shape, f_compute=lambda indices: e.apply_scalar(x[indices], y[indices]))
        else:
            return super().visit_BinaryTileOp(e)


class FoldConstantPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewrites = [SimplifyTileCreationRewriter(), DeadCodeEliminationRewriter()]
        for rewriter in rewrites:
            func = rewriter.visit(func)
        return func


def fold_constant_pass() -> TileFunctionPass:
    return FoldConstantPass()
