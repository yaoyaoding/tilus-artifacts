from typing import List
from hidet.ir.expr import Expr
from hidet.ir.tile.type import TileType
from hidet.ir.tile.expr import CallTileOp, TileOp
from hidet.ir.tile.stmt import PureForStmt, YieldStmt
from hidet.ir.tile.layout import TileLayout
from hidet.ir.tile.ops.creation import Create
from hidet.ir.tile.ops.memory import Load, StoreBaseOp
from hidet.ir.tile.ops.transform import Broadcast, ExpandDims, CastOp
from hidet.ir.tile.ops.convert_layout import ConvertLayout
from hidet.ir.tile.ops.arthimatic import UnaryTileOp, BinaryTileOp
from hidet.ir.tile.ops.reduce import ReduceOp
from hidet.ir.tile.ops.debug import DebugPrint
from hidet.ir.tile.ops.dot import Dot
from hidet.ir.tile.ops.assign import Assign
from hidet.ir.tile.ops.smem import AllocTensor, InsertSliceAsync, ExtractSlice, ProcedureOp
from hidet.utils import same_list
from .base_functor import BaseFunctor, BaseVisitor, BaseRewriter


class TileFunctor(BaseFunctor):
    def __init__(self, use_memo=True):
        super().__init__(use_memo)
        self.pure_for_stmts: List[PureForStmt] = []

    def visit_dispatch(self, node):
        if isinstance(node, TileType):
            return self.visit_TileType(node)
        elif isinstance(node, TileLayout):
            return self.visit_TileLayout(node)
        elif isinstance(node, CallTileOp):
            return self.visit_CallTileOp(node)
        elif isinstance(node, PureForStmt):
            return self.visit_PureForStmt(node)
        elif isinstance(node, YieldStmt):
            return self.visit_YieldStmt(node)
        elif isinstance(node, UnaryTileOp):
            return self.visit_UnaryTileOp(node)
        elif isinstance(node, BinaryTileOp):
            return self.visit_BinaryTileOp(node)
        elif isinstance(node, Load):
            return self.visit_Load(node)
        elif isinstance(node, StoreBaseOp):
            return self.visit_StoreBaseOp(node)
        elif isinstance(node, Broadcast):
            return self.visit_Broadcast(node)
        elif isinstance(node, Create):
            return self.visit_Create(node)
        elif isinstance(node, ConvertLayout):
            return self.visit_ConvertLayout(node)
        elif isinstance(node, ExpandDims):
            return self.visit_ExpandDims(node)
        elif isinstance(node, CastOp):
            return self.visit_CastOp(node)
        elif isinstance(node, DebugPrint):
            return self.visit_DebugPrint(node)
        elif isinstance(node, ReduceOp):
            return self.visit_ReduceOp(node)
        elif isinstance(node, Dot):
            return self.visit_Dot(node)
        elif isinstance(node, Assign):
            return self.visit_Assign(node)
        elif isinstance(node, AllocTensor):
            return self.visit_AllocTensor(node)
        elif isinstance(node, InsertSliceAsync):
            return self.visit_InsertSliceAsync(node)
        elif isinstance(node, ExtractSlice):
            return self.visit_ExtractSlice(node)
        elif isinstance(node, ProcedureOp):
            return self.visit_ProcedureOp(node)
        elif isinstance(node, TileOp):
            raise NotImplementedError(
                'Rewriter for the following tile op is not implemented: \n{}'.format(node.op_name())
            )
        else:
            return NotImplemented

    def visit_CallTileOp(self, call: CallTileOp):
        raise NotImplementedError()

    def visit_TileType(self, t: TileType):
        raise NotImplementedError()

    def visit_TileLayout(self, t: TileLayout):
        raise NotImplementedError()

    def visit_UnaryTileOp(self, e: UnaryTileOp):
        raise NotImplementedError()

    def visit_BinaryTileOp(self, e: BinaryTileOp):
        raise NotImplementedError()

    def visit_Create(self, e: Create):
        raise NotImplementedError()

    def visit_Load(self, e: Load):
        raise NotImplementedError()

    def visit_StoreBaseOp(self, e: StoreBaseOp):
        raise NotImplementedError()

    def visit_Broadcast(self, e: Broadcast):
        raise NotImplementedError()

    def visit_ExpandDims(self, e: ExpandDims):
        raise NotImplementedError()

    def visit_CastOp(self, e: CastOp):
        raise NotImplementedError()

    def visit_ConvertLayout(self, e: ConvertLayout):
        raise NotImplementedError()

    def visit_ReduceOp(self, e: ReduceOp):
        raise NotImplementedError()

    def visit_Dot(self, e: Dot):
        raise NotImplementedError()

    def visit_Assign(self, e: Assign):
        raise NotImplementedError()

    def visit_AllocTensor(self, e: AllocTensor):
        raise NotImplementedError()

    def visit_InsertSliceAsync(self, e: InsertSliceAsync):
        raise NotImplementedError()

    def visit_ExtractSlice(self, e: ExtractSlice):
        raise NotImplementedError()

    def visit_ProcedureOp(self, e: ProcedureOp):
        raise NotImplementedError()

    def visit_DebugPrint(self, e: DebugPrint):
        raise NotImplementedError()

    def visit_PureForStmt(self, e: PureForStmt):
        raise NotImplementedError()

    def visit_YieldStmt(self, e: YieldStmt):
        raise NotImplementedError()


class TileVisitor(TileFunctor, BaseVisitor):
    def visit_TileType(self, t: TileType):
        self.visit(t.type)

    def visit_TileLayout(self, t: TileLayout):
        pass

    def visit_CallTileOp(self, call: CallTileOp):
        self.visit(call.op)

    def visit_UnaryTileOp(self, e: UnaryTileOp):
        self.visit(e.args)

    def visit_BinaryTileOp(self, e: BinaryTileOp):
        self.visit(e.args)

    def visit_Create(self, e: Create):
        self.visit(e.args)

    def visit_Load(self, e: Load):
        self.visit(e.args)

    def visit_StoreBaseOp(self, e: StoreBaseOp):
        self.visit(e.args)

    def visit_Broadcast(self, e: Broadcast):
        self.visit(e.x)

    def visit_ExpandDims(self, e: ExpandDims):
        self.visit(e.x)

    def visit_CastOp(self, e: CastOp):
        self.visit(e.x)

    def visit_ConvertLayout(self, e: ConvertLayout):
        self.visit(e.x)

    def visit_ReduceOp(self, e: ReduceOp):
        self.visit(e.x)

    def visit_Dot(self, e: Dot):
        self.visit(e.args)

    def visit_Assign(self, e: Assign):
        self.visit(e.args)

    def visit_DebugPrint(self, e: DebugPrint):
        self.visit(e.x)

    def visit_AllocTensor(self, e: AllocTensor):
        pass

    def visit_InsertSliceAsync(self, e: InsertSliceAsync):
        self.visit(e.args)

    def visit_ExtractSlice(self, e: ExtractSlice):
        self.visit(e.args)

    def visit_ProcedureOp(self, e: ProcedureOp):
        pass

    def visit_PureForStmt(self, stmt: PureForStmt):
        self.visit(stmt.args)
        self.visit(stmt.values)
        self.visit(stmt.loop_var)
        self.visit(stmt.extent)
        self.pure_for_stmts.append(stmt)
        self.visit(stmt.body)
        self.pure_for_stmts.pop()
        self.visit(stmt.let_vars)
        self.visit(stmt.let_body)

    def visit_YieldStmt(self, stmt: YieldStmt):
        self.visit(stmt.values)


class TileRewriter(TileFunctor, BaseRewriter):
    def visit_TileType(self, t: TileType):
        tp = self.visit(t.type)
        if tp is t.type:
            return t
        else:
            return TileType(tp, shape=t.shape, layout=t.layout)

    def visit_TileLayout(self, t: TileLayout):
        return t

    def visit_CallTileOp(self, call: CallTileOp):
        op = self.visit(call.op)
        if op is call.op:
            return call
        else:
            if isinstance(op, TileOp):
                return op.make_call()
            else:
                # we allow the visitor of TileOp to return an Expr
                assert isinstance(op, Expr)
                return op

    def visit_UnaryTileOp(self, e: UnaryTileOp):
        x = self.visit(e.x)
        if x is e.x:
            return e
        else:
            return e.reforward([x])

    def visit_BinaryTileOp(self, e: BinaryTileOp):
        x = self.visit(e.x)
        y = self.visit(e.y)
        if x is e.x and y is e.y:
            return e
        else:
            return e.reforward([x, y])

    def visit_Create(self, e: Create):
        value = self.visit(e.value)
        if value is e.value:
            return e
        else:
            return e.reforward([value])

    def visit_Load(self, e: Load):
        ptr = self.visit(e.ptr)
        mask = self.visit(e.mask)
        other = self.visit(e.other)
        if ptr is e.ptr and mask is e.mask and other is e.other:
            return e
        else:
            return e.reforward([ptr, mask, other])

    def visit_StoreBaseOp(self, e: StoreBaseOp):
        ptr = self.visit(e.ptr)
        value = self.visit(e.value)
        mask = self.visit(e.mask)
        if ptr is e.ptr and mask is e.mask and value is e.value:
            return e
        else:
            return e.reforward([ptr, value, mask])

    def visit_Broadcast(self, e: Broadcast):
        x = self.visit(e.x)
        if x is e.x:
            return e
        else:
            return e.reforward([x])

    def visit_ExpandDims(self, e: ExpandDims):
        x = self.visit(e.x)
        if x is e.x:
            return e
        else:
            return e.reforward([x])

    def visit_CastOp(self, e: CastOp):
        x = self.visit(e.x)
        if x is e.x:
            return e
        else:
            return e.reforward([x])

    def visit_ConvertLayout(self, e: ConvertLayout):
        x = self.visit(e.x)
        if x is e.x:
            return e
        else:
            return e.reforward([x])

    def visit_ReduceOp(self, e: ReduceOp):
        x = self.visit(e.x)
        if x is e.x:
            return e
        else:
            return e.reforward([x])

    def visit_Dot(self, e: Dot):
        a = self.visit(e.a)
        b = self.visit(e.b)
        c = self.visit(e.c)
        if a is e.a and b is e.b and c is e.c:
            return e
        else:
            return e.reforward([a, b, c])

    def visit_Assign(self, e: Assign):
        src = self.visit(e.src)
        dst = self.visit(e.dst)
        if src is e.src and dst is e.dst:
            return e
        else:
            return e.reforward([dst, src])

    def visit_DebugPrint(self, e: DebugPrint):
        x = self.visit(e.x)
        if x is e.x:
            return e
        else:
            return e.reforward([x])

    def visit_AllocTensor(self, e: AllocTensor):
        return e

    def visit_InsertSliceAsync(self, e: InsertSliceAsync):
        args = self.visit(e.args)
        if args is e.args:
            return e
        else:
            return e.reforward(args)

    def visit_ExtractSlice(self, e: ExtractSlice):
        args = self.visit(e.args)
        if args is e.args:
            return e
        else:
            return e.reforward(args)

    def visit_ProcedureOp(self, e: ProcedureOp):
        return e

    def visit_PureForStmt(self, stmt: PureForStmt):
        self.pure_for_stmts.append(stmt)
        args = self.visit(stmt.args)
        values = self.visit(stmt.values)
        loop_var = self.visit(stmt.loop_var)
        extent = self.visit(stmt.extent)
        body = self.visit(stmt.body)
        self.pure_for_stmts.pop()
        let_vars = self.visit(stmt.let_vars)
        let_body = self.visit(stmt.let_body)
        if (  # pylint: disable=too-many-boolean-expressions
            same_list(args, stmt.args)
            and loop_var is stmt.loop_var
            and extent is stmt.extent
            and body is stmt.body
            and same_list(let_vars, stmt.let_vars)
            and let_body is stmt.let_body
        ):
            return stmt
        else:
            return PureForStmt(
                args=args,
                values=values,
                loop_var=loop_var,
                extent=extent,
                body=body,
                let_vars=let_vars,
                let_body=let_body,
            )

    def visit_YieldStmt(self, stmt: YieldStmt):
        yields = self.visit(stmt.values)
        if yields is stmt.values:
            return stmt
        else:
            return YieldStmt(values=yields)
