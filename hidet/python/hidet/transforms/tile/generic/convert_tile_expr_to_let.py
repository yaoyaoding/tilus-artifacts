from typing import Union, List

from hidet.ir.expr import Let, var
from hidet.ir.stmt import LetStmt
from hidet.ir.module import IRModule
from hidet.ir.functors import IRRewriter
from hidet.ir.func import Function
from hidet.ir.tools import TypeInfer, collect
from hidet.ir.tile.type import TileType
from hidet.ir.tile.expr import TileOp, CallTileOp
from hidet.utils import same_list
from hidet.transforms.expand_let_expr import LetExprExpander
from hidet.transforms.declare_to_let import DeclareToLetRewriter
from hidet.transforms.base import TileFunctionPass


class TileDeclareToLetRewriter(DeclareToLetRewriter):
    def update_assigns(self, node):
        from hidet.ir.tile.ops import Assign

        super().update_assigns(node)

        # mark the dst of all Assign ops
        # so that their definition will not be converted to LetStmt from DeclareStmt
        calls: List[CallTileOp] = collect(node, node_types=[CallTileOp])
        for call in calls:
            if isinstance(call.op, Assign):
                self.assigns[call.op.dst] += 1


class ConvertTileExprToLetRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()

    def visit_LetStmt(self, stmt: LetStmt):
        stmt = super().visit_LetStmt(stmt)
        bind_values = []
        for _, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            if isinstance(bind_value, Let) and bind_value.body is bind_value.var:
                # Let v = (let vv=e: v) => # Let v = e
                bind_values.append(bind_value.value)
            else:
                bind_values.append(bind_value)
        if same_list(stmt.bind_values, bind_values):
            return stmt
        else:
            return LetStmt(stmt.bind_vars, bind_values, stmt.body)

    def visit_CallTileOp(self, call: CallTileOp):
        call = super().visit_CallTileOp(call)
        ret_type = self.type_infer(call)
        if isinstance(ret_type, TileType):
            assert isinstance(call.op, TileOp)
            v = var(hint=call.op.var_name_hint, dtype=ret_type)
            return Let(v, call, v)
        else:
            return call


class FlattenLetChainRewriter(IRRewriter):
    def visit_LetStmt(self, stmt: LetStmt):
        stmt = super().visit_LetStmt(stmt)
        if isinstance(stmt.body, LetStmt):
            bind_vars = stmt.bind_vars + stmt.body.bind_vars
            bind_values = stmt.bind_values + stmt.body.bind_values
            body = stmt.body.body
            return LetStmt(bind_vars, bind_values, body)
        else:
            return stmt


class ConvertTileExprToLetPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        return convert_to_let(func)


def convert_to_let(node: Union[IRModule, Function]):
    rewrites = [
        TileDeclareToLetRewriter(),
        ConvertTileExprToLetRewriter(),
        LetExprExpander(),
        FlattenLetChainRewriter(),
    ]
    for r in rewrites:
        node = r(node)
    return node


def convert_tile_expr_to_let_pass() -> TileFunctionPass:
    return ConvertTileExprToLetPass()


raise ValueError()
