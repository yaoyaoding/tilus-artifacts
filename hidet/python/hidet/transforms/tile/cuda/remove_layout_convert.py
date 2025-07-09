from collections import defaultdict
from typing import Dict, Optional, Type

from hidet.ir.expr import Var, Expr
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import EvaluateStmt, Stmt
from hidet.ir.tile.expr import CallTileOp, TileOp
from hidet.ir.tile.layout import BlockDotOperandLayout
from hidet.ir.tile.ops import Broadcast, BinaryTileOp, ReduceOp, Dot, ExpandDims, StoreBaseOp, Load
from hidet.ir.tile.ops import Create, convert_layout, ConvertLayout, CastOp, DebugPrint, UnaryTileOp
from hidet.ir.tile.ops import InsertSliceAsync, ExtractSlice
from hidet.ir.tile.stmt import PureForStmt, YieldStmt
from hidet.ir.tile.type import TileLayout
from hidet.ir.tile.type import TileType
from hidet.transforms.base import TileFunctionPass
from hidet.transforms.tile.analyzers import VarUsage, UsageAnalyzer, LevelAnalyzer
from hidet.transforms.tile.generic.canonicalize_to_ssa import canonicalize_to_ssa
from hidet.transforms.tile.generic.dead_code_elimination import DeadCodeEliminationRewriter
from hidet.transforms.tile.generic.pattern_transform import PatternTransform, Pattern
from hidet.transforms.tile.utils.pattern import TilePattern
from hidet.utils import same_list


class IdentityConvertLayoutTransform(PatternTransform):
    """
    convert_layout(tile with layout1, layout1) -> tile with layout1
    """

    def __init__(self):
        self.x = self.any_tile()
        self.cvt = self.convert_layout(self.x)

    def source(self) -> TilePattern:
        return self.cvt

    def target(self, matched: Dict[Pattern, Expr], var2call: Dict[Var, CallTileOp]) -> Optional[Expr]:
        x = matched[self.x]
        cvt: ConvertLayout = self.get_tile_op(self.cvt, matched, var2call)
        if isinstance(x, Var) and isinstance(x.type, TileType) and x.type.layout == cvt.layout:
            return x
        else:
            return None


class FoldConvertLayoutTransform(PatternTransform):
    """
    convert_layout(convert_layout(x, layout1), layout2) -> convert_layout(x, layout2)
    """

    def __init__(self):
        self.x = self.any_tile()
        self.cvt1 = self.convert_layout(self.x)
        self.cvt2 = self.convert_layout(self.cvt1)

    def source(self) -> TilePattern:
        return self.cvt2

    def target(self, matched: Dict[Pattern, Expr], var2call: Dict[Var, CallTileOp]) -> Optional[Expr]:
        x = matched[self.x]
        cvt2: ConvertLayout = self.get_tile_op(self.cvt2, matched, var2call)
        return ConvertLayout(x, cvt2.layout, cvt2.scope).make_call()


class ConvertConstructLayoutTransform(PatternTransform):
    """
    convert_layout(construct(..., layout1), layout2) -> construct(..., layout2)
    """

    def __init__(self):
        self.cst = self.construct()
        self.cvt = self.convert_layout(self.cst)

    def source(self) -> TilePattern:
        return self.cvt

    def target(self, matched: Dict[Pattern, Expr], var2call: Dict[Var, CallTileOp]) -> Optional[Expr]:
        cst: Create = self.get_tile_op(self.cst, matched, var2call)
        cvt: ConvertLayout = self.get_tile_op(self.cvt, matched, var2call)

        updated_cst = Create(value=cst.value, shape=cst.shape, axes=cst.axes, layout=cvt.layout)
        return updated_cst.make_call()


class PushConvertLayoutForUnaryOpTransform(PatternTransform):
    """
    convert_layout(op(x), layout) -> op(convert_layout(x, layout))
    """

    def __init__(self):
        self.x = self.any_tile()
        self.y = self.unary(self.x)
        self.cvt = self.convert_layout(self.y)

    def source(self) -> TilePattern:
        return self.cvt

    def target(self, matched: Dict[Pattern, Expr], var2call: Dict[Var, CallTileOp]) -> Optional[Expr]:
        x = matched[self.x]
        op: UnaryTileOp = self.get_tile_op(self.y, matched, var2call)
        cvt: ConvertLayout = self.get_tile_op(self.cvt, matched, var2call)
        if isinstance(cvt.layout, BlockDotOperandLayout):
            # do not push convert_layout for BlockDotOperandLayout because it consumes too many registers
            return None
        return op.reforward(args=[convert_layout(x, cvt.layout)]).make_call()


class PushConvertLayoutForBinaryOpTransform(PatternTransform):
    """
    convert_layout(op(x, y), layout) -> op(convert_layout(x, layout), convert_layout(y, layout))

    where the output of op(x, y) is not in shared scope
    """

    def __init__(self):
        self.x = self.any_tile()
        self.y = self.any_tile()
        self.z = self.binary(self.x, self.y)
        self.cvt = self.convert_layout(self.z)

    def source(self) -> TilePattern:
        return self.cvt

    def target(self, matched: Dict[Pattern, Expr], var2call: Dict[Var, CallTileOp]) -> Optional[Expr]:
        x = matched[self.x]
        y = matched[self.y]
        op: BinaryTileOp = self.get_tile_op(self.z, matched, var2call)
        cvt: ConvertLayout = self.get_tile_op(self.cvt, matched, var2call)
        # if isinstance(cvt.layout, BlockDotOperandLayout):
        #     # do not push convert_layout for BlockDotOperandLayout because it consumes too many registers
        #     return None
        if cvt.scope is not None and cvt.scope.is_shared():
            # do not push convert_layout when converting to shared memory scope
            return None
        return op.reforward(args=[convert_layout(x, cvt.layout), convert_layout(y, cvt.layout)]).make_call()


class FoldConvertLayoutBeforeAndAfterCast(PatternTransform):
    """
    convert_layout(cast(convert_layout(x, layout1)), layout2) -> cast(convert_layout(x, layout2))
    """

    def __init__(self):
        self.x = self.any_tile()
        self.cvt1 = self.convert_layout(self.x)
        self.cst = self.cast(self.cvt1)
        self.cvt2 = self.convert_layout(self.cst)

    def source(self) -> TilePattern:
        return self.cvt2

    def target(self, matched: Dict[Pattern, Expr], var2call: Dict[Var, CallTileOp]) -> Optional[Expr]:
        from hidet.ir.tile.ops import cast  # pylint: disable=reimported

        x = matched[self.x]
        cst: CastOp = self.get_tile_op(self.cst, matched, var2call)
        cvt2: ConvertLayout = self.get_tile_op(self.cvt2, matched, var2call)
        return cast(convert_layout(x, cvt2.layout), cst.dtype)


class ExchangeConvertLayoutForBinaryOpTransform(PatternTransform):
    """
    op(convert_layout(x, y's layout), y) -> op(x, convert_layout(y, x's layout))
    if
    y's level is smaller than x's level (level indicate the number of nested loop the var is defined in)
    and
    x is not in shared scope
    """

    def __init__(self, var2level: Dict[Var, int]):
        self.x = self.any_tile()
        self.y = self.any_tile()
        self.z1 = self.convert_layout(self.x)
        self.z2 = self.binary(self.z1, self.y)

        self.var2level: Dict[Var, int] = var2level

    def source(self) -> TilePattern:
        return self.z2

    def target(self, matched: Dict[Pattern, Expr], var2call: Dict[Var, CallTileOp]) -> Optional[Expr]:
        x = matched[self.x]
        y = matched[self.y]
        op: BinaryTileOp = self.get_tile_op(self.z2, matched, var2call)

        if not (isinstance(x, Var) and isinstance(y, Var) and isinstance(x.type, TileType)):
            return None

        if not x.type.scope.is_register():
            return None

        assert x in self.var2level, 'can not find level for {}'.format(x)
        assert y in self.var2level, 'can not find level for {}'.format(y)

        if self.var2level[x] <= self.var2level[y]:
            return None

        return op.reforward(args=[x, convert_layout(y, x.type.layout)]).make_call()


class ExchangeConvertLayoutForBinaryOpRewriter(IRRewriter):
    def visit_Function(self, func: Function):
        level_analyzer = LevelAnalyzer()

        level_analyzer(func)
        # print('\n'.join('{}: {}'.format(k, v) for k, v in level_analyzer.var2level.items()))
        transforms = [ExchangeConvertLayoutForBinaryOpTransform(var2level=level_analyzer.var2level)]
        for transform in transforms:
            func = transform(func)
        return func


class ChangeForArgLayoutRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.usages: Dict[Var, VarUsage] = {}

    def anchor_priority(self, op: Type[TileOp]):
        order = [
            Dot,
            Load,
            StoreBaseOp,
            ReduceOp,
            InsertSliceAsync,
            ExtractSlice,
            Broadcast,
            ExpandDims,
            BinaryTileOp,
            ConvertLayout,
            DebugPrint,
        ]
        for idx, cls in enumerate(order):
            if issubclass(op, cls):
                return len(order) - idx
        raise NotImplementedError(op)

    def stmt_priority(self, stmt: Type[Stmt]):
        if isinstance(stmt, PureForStmt):
            return 100
        elif isinstance(stmt, YieldStmt):
            return 90
        else:
            raise NotImplementedError()

    def visit_Function(self, func: Function):
        usage_analyzer = UsageAnalyzer()
        usage_analyzer.visit(func)
        self.memo.clear()
        self.usages = usage_analyzer.usages
        updated_func = super().visit_Function(func)
        if updated_func is not func:
            updated_func = canonicalize_to_ssa(updated_func)
        return updated_func

    def get_usage_priority(self, usage: VarUsage):
        ret = -1
        for let_usage in usage.call_op_let_usages():
            p = self.anchor_priority(type(let_usage.op))
            ret = max(ret, p)
        for stmt_usage in usage.stmt_usages:
            s = stmt_usage.stmt
            if isinstance(s, EvaluateStmt):
                if isinstance(s.expr, CallTileOp):
                    ret = max(ret, self.anchor_priority(type(s.expr.op)))
                else:
                    raise NotImplementedError()
            else:
                ret = max(ret, self.stmt_priority(type(s)))
        return ret

    def visit_PureForStmt(self, stmt: PureForStmt):
        arg2layout: Dict[Var, TileLayout] = {}
        for arg in stmt.args:
            if not isinstance(arg.type, TileType):
                continue
            layout = arg.type.layout

            usage: VarUsage = self.usages[arg]

            # the mapping from the layout to the list of tile operators that require the arg to have the layout
            layout2priority: Dict[TileLayout, int] = defaultdict(int)

            for let_usage in usage.call_op_let_usages():
                op = let_usage.op
                if isinstance(op, ConvertLayout):
                    bind_var = let_usage.bind_var
                    layout2priority[op.layout] = max(
                        layout2priority[op.layout], self.get_usage_priority(self.usages[bind_var])
                    )
            layout2priority[layout] = max(layout2priority[layout], self.get_usage_priority(usage))

            # find the layout that has the anchor with the highest priority
            best_layout = max(layout2priority.keys(), key=lambda l: layout2priority[l])
            if best_layout == layout:
                continue
            arg2layout[arg] = best_layout

        if len(arg2layout) == 0:
            return super().visit_PureForStmt(stmt)

        # update the layout
        args = []
        values = []
        let_vars = []
        for orig_arg, let_var, value in zip(stmt.args, stmt.let_vars, stmt.values):
            value = self.visit(value)
            if orig_arg in arg2layout:
                assert isinstance(orig_arg, Var) and isinstance(orig_arg.type, TileType)
                tp = orig_arg.type
                orig_layout = tp.layout
                args.append(Var(orig_arg.hint, TileType(tp.type, tp.shape, arg2layout[orig_arg])))
                let_vars.append(Var(let_var.hint, TileType(tp.type, tp.shape, arg2layout[orig_arg])))
                values.append(convert_layout(value, arg2layout[orig_arg]))
                self.memo[orig_arg] = convert_layout(args[-1], orig_layout)
                self.memo[let_var] = convert_layout(let_vars[-1], orig_layout)
            else:
                args.append(orig_arg)
                values.append(value)
                let_vars.append(let_var)

        loop_var = self.visit(stmt.loop_var)
        extent = self.visit(stmt.extent)
        self.pure_for_stmts.append(stmt)
        body = self.visit(stmt.body)
        self.pure_for_stmts.pop()
        let_body = self.visit(stmt.let_body)
        return PureForStmt(
            args=args, values=values, loop_var=loop_var, extent=extent, body=body, let_vars=let_vars, let_body=let_body
        )

    def visit_YieldStmt(self, stmt: YieldStmt):
        for_stmt = self.pure_for_stmts[-1]
        yields = self.visit(stmt.values)
        updated_yields = []
        for arg, yield_value in zip(for_stmt.args, yields):
            if arg is not self.memo[arg]:
                call_cvt: CallTileOp = self.memo[arg]
                assert isinstance(call_cvt.op, ConvertLayout)
                cvt: ConvertLayout = call_cvt.op
                assert isinstance(cvt.x, Var)
                updated_arg = cvt.x
                assert isinstance(updated_arg, Var) and isinstance(updated_arg.type, TileType)
                updated_yields.append(convert_layout(yield_value, updated_arg.type.layout))
            else:
                updated_yields.append(yield_value)
        if same_list(updated_yields, yields):
            return stmt
        else:
            return YieldStmt(updated_yields)


class RemoveLayoutConvertPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        return remove_layout_convert(func)


def _apply_transforms(transforms, func) -> Function:
    # idx = 0
    while True:
        orig_func = func
        for transform in transforms:
            if isinstance(transform, list):
                func = _apply_transforms(transform, func)
            else:
                func = transform(func)
            if orig_func is not func:
                func = canonicalize_to_ssa(func)
        if orig_func is func:
            break
    return func


def remove_layout_convert(func: Function) -> Function:

    level_analyzer = LevelAnalyzer()
    level_analyzer(func)

    return _apply_transforms(
        [
            ChangeForArgLayoutRewriter(),
            ExchangeConvertLayoutForBinaryOpRewriter(),
            [
                IdentityConvertLayoutTransform(),
                ConvertConstructLayoutTransform(),
                FoldConvertLayoutTransform(),
                PushConvertLayoutForUnaryOpTransform(),
                PushConvertLayoutForBinaryOpTransform(),
                FoldConvertLayoutBeforeAndAfterCast(),
            ],
            DeadCodeEliminationRewriter(),
        ],
        func,
    )


def remove_layout_convert_pass() -> TileFunctionPass:
    return RemoveLayoutConvertPass()
