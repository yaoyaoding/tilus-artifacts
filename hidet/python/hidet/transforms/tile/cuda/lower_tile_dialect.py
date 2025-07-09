from typing import List, Dict, Union, Optional

from hidet.ir.expr import Var, Expr, tensor_var, tensor_pointer_var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import LetStmt, DeclareStmt, ForStmt, AssignStmt
from hidet.ir.stmt import Stmt, SeqStmt, EvaluateStmt
from hidet.ir.tools import IRPrinter
from hidet.ir.tile.expr import TileOp, CallTileOp
from hidet.ir.tile.layout import TileLayout
from hidet.ir.tile.type import TileType, TileScope
from hidet.ir.tile.stmt import PureForStmt, YieldStmt
from hidet.ir.tools import TypeInfer
from hidet.ir.type import BaseType
from hidet.ir.type import DataType, PointerType
from hidet.transforms.base import TileFunctionPass
from hidet.transforms.declare_to_let import DeclareToLetRewriter, UpliftLetBodyRewriter
from hidet.transforms.tile.analyzers import UsageAnalyzer, VarUsage, TensorInfo, ValueInfo, analyze_value
from hidet.transforms.tile.cuda.lower_ops.assign import AssignImpl
from hidet.ir.primitives.debug import comment
from .lower_ops import Buffer, implement_tile_op


class LowerTileDialectRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        # the mapping from the defined var to the corresponding buffer
        # the defined var can be either the var in LetStmt/DeclareStmt, or the one created in Buffer
        self.num_warps: Optional[int] = None
        self.usages: Dict[Var, VarUsage] = {}
        self.var2buffer: Dict[Var, Buffer] = {}
        self.stmts: List[Stmt] = []
        self.type_infer = TypeInfer()
        self.var2value: Dict[Var, ValueInfo] = {}
        self.printer = IRPrinter()

    def alloc_buffer(self, hint: str, tile_op_or_type: Union[TileOp, TileType]) -> Buffer:
        if isinstance(tile_op_or_type, TileOp):
            ttype: TileType = self.type_infer(CallTileOp(tile_op_or_type))
        else:
            ttype: TileType = tile_op_or_type
        assert ttype.layout is not None
        layout: TileLayout = ttype.layout
        shape: List[int] = ttype.shape
        dtype: Union[DataType, PointerType] = ttype.type
        scope: TileScope = ttype.scope
        local_shape = layout.local_shape()

        if scope.is_register():
            buf_var: Var = tensor_var(hint=hint, shape=local_shape, dtype=dtype)
        elif scope.is_shared():
            buf_var: Var = tensor_pointer_var(hint=hint, shape=local_shape, dtype=dtype)
        else:
            raise NotImplementedError()

        self.append_stmt(DeclareStmt(buf_var))

        buf = Buffer(buf_var=buf_var, dtype=dtype, shape=shape, scope=scope, local_shape=local_shape, layout=layout)
        return buf

    def append_stmt(self, stmt: Union[Stmt, Expr]):
        if isinstance(stmt, Expr):
            stmt = EvaluateStmt(stmt)
        self.stmts.append(stmt)

    def flush_stmts(self):
        stmts = self.stmts
        self.stmts = []
        return stmts

    def assign_buffer(self, dst: Buffer, src: Buffer):
        if dst.scope == src.scope == TileScope.Register:
            assign_impl = AssignImpl(self.num_warps)
            assign_impl.implement(None, args=[dst, src], output=None)
            self.stmts.append(assign_impl.finish())
        elif src.scope == dst.scope == TileScope.Shared:
            self.append_stmt(AssignStmt(dst.var, src.var))
        else:
            raise NotImplementedError()

    def visit_CallTileOp(self, call: CallTileOp):
        # add a comment statement before implementing the tile op
        self.append_stmt(comment(str(call), style='/*'))

        args: List[Union[Expr, Buffer]] = []
        for arg in call.op.args:
            arg_type = self.type_infer(arg)
            if isinstance(arg_type, TileType):
                assert isinstance(arg, Var)
                args.append(self.var2buffer[arg])
            else:
                args.append(self.visit(arg))

        output_type: BaseType = self.type_infer(call)
        if output_type.is_void():
            output: Optional[Buffer] = None
        elif isinstance(output_type, TileType):
            output: Optional[Buffer] = self.alloc_buffer(call.op.name, output_type)
        else:
            raise NotImplementedError()
        self.append_stmt(implement_tile_op(call.op, args=args, output=output, num_warps=self.num_warps))

        return output

    def visit_LetStmt(self, stmt: LetStmt):
        stmts: List[Stmt] = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            if isinstance(bind_value, CallTileOp):
                # tile expression
                buf = self.visit(bind_value)
                if not isinstance(buf, Buffer):
                    raise NotImplementedError(
                        'The following tile expression has not been lowered to Buffer:\n'
                        + '  {}'.format(type(bind_value.op).__name__)
                    )
                self.var2buffer[bind_var] = buf
                buf.info = self.var2value[bind_var] if bind_var in self.var2value else TensorInfo.from_shape(buf.shape)
                buf.var.hint = bind_var.hint
            elif isinstance(bind_value, Var) and isinstance(bind_value.type, TileType):
                self.memo[bind_var] = bind_value
            else:
                # scalar expression
                self.append_stmt(DeclareStmt(bind_var, self.visit(bind_value)))
            stmts.extend(self.flush_stmts())
        stmts.append(self.visit(stmt.body))
        if len(stmts) == 1:
            return stmts[0]
        else:
            return SeqStmt(stmts)

    def visit_PureForStmt(self, stmt: PureForStmt):
        stmts = []

        # copy the argument initial values to argument buffers
        for arg, value in zip(stmt.args, stmt.values):
            if isinstance(arg.type, TileType):
                assert isinstance(value, Var)
                if self.usages[value].count() == 1:
                    # the value is only used as the argument of the loop, so we can directly use the buffer
                    self.var2buffer[arg] = self.var2buffer[value]
                else:
                    # otherwise, we need to create a buffer for the arg and copy the value to the buffer
                    arg_buf: Buffer = self.alloc_buffer(arg.hint, arg.type)
                    self.assign_buffer(dst=arg_buf, src=self.var2buffer[value])
                    self.var2buffer[arg] = arg_buf
                    arg_buf.info = (
                        self.var2value[arg] if arg in self.var2value else TensorInfo.from_shape(arg_buf.shape)
                    )
            else:
                self.append_stmt(DeclareStmt(arg, init=self.visit(value)))
            stmts.extend(self.flush_stmts())

        # implement the loop body
        self.pure_for_stmts.append(stmt)
        body = self.visit(stmt.body)
        stmts.append(ForStmt(stmt.loop_var, stmt.extent, body))
        self.pure_for_stmts.pop()

        # mapping the let_vars to the buffer
        for let_var, arg in zip(stmt.let_vars, stmt.args):
            if isinstance(arg.type, TileType):
                self.var2buffer[let_var] = self.var2buffer[arg]
            else:
                self.memo[let_var] = arg

        # implement the let body
        stmts.append(self.visit(stmt.let_body))
        return SeqStmt(stmts)

    def visit_YieldStmt(self, stmt: YieldStmt):
        for_stmt = self.pure_for_stmts[-1]
        stmts = []
        for arg, yield_value in zip(for_stmt.args, stmt.values):
            assert isinstance(yield_value, Var)
            if isinstance(arg.type, TileType):
                self.assign_buffer(dst=self.var2buffer[arg], src=self.var2buffer[yield_value])
            else:
                self.append_stmt(AssignStmt(arg, self.visit(yield_value)))
            stmts.extend(self.flush_stmts())
        return SeqStmt(stmts)

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        if isinstance(stmt.expr, CallTileOp):
            ret = self.visit(stmt.expr)
            assert isinstance(ret, Buffer) or ret is None
            stmts = self.flush_stmts()
            if len(stmts) == 1:
                return stmts[0]
            else:
                return SeqStmt(stmts)
        else:
            return super().visit_EvaluateStmt(stmt)

    def visit_Function(self, func: Function):
        usage_analyzer = UsageAnalyzer()
        usage_analyzer.visit(func)
        self.usages = usage_analyzer.usages
        self.var2value = analyze_value(func)
        self.num_warps = func.attrs['cuda.block_dim'] // 32
        ret = super().visit_Function(func)
        if ret.kind == 'cuda_tile':
            ret.kind = 'cuda_kernel'
        return ret


class LowerTileDialectPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        return self.apply_transforms(
            func, [LowerTileDialectRewriter(), DeclareToLetRewriter(), UpliftLetBodyRewriter()]
        )


def lower_tile_dialect_pass() -> TileFunctionPass:
    return LowerTileDialectPass()
