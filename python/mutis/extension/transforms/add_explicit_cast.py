from typing import Optional, List, Sequence

from hidet.ir import BufferStoreStmt
from hidet.ir.type import FuncType, BaseType, PointerType, DataType, type_equal, get_base_type
from hidet.ir.expr import Expr, Call, cast
from hidet.ir.stmt import SeqStmt, LaunchKernelStmt, AssertStmt, AssignStmt, LetStmt
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.tools import TypeInfer
from hidet.transforms.base import FunctionPass
from hidet.utils import same_list


class AddExplicitCastRewriter(IRRewriter):
    def __init__(self):
        super().__init__(use_memo=False)
        self.type_infer = TypeInfer()

    def process(self, expr: Expr, target_type: BaseType):
        source_type = self.type_infer(expr)

        # If we are doing pointer cast and the two types are different
        cond1 = isinstance(target_type, PointerType) and not type_equal(source_type, target_type)

        # If we are assigning one data type to another data type
        cond2 = (
            isinstance(source_type, DataType)
            and target_type.is_data_type()
            and not type_equal(source_type, target_type)
        )

        perform_explicit_cast = cond1 or cond2

        if perform_explicit_cast:
            processed = cast(expr, target_type)
        else:
            processed = expr
        return processed

    def process_list(self, exprs: Sequence[Expr], target_types: List[BaseType]):
        return [self.process(expr, target_type) for expr, target_type in zip(exprs, target_types)]

    def visit_Call(self, e: Call):
        func_type: FuncType = e.func_var.type
        args = self.process_list(self.visit(e.args), func_type.param_types)
        if same_list(args, e.args):
            return e
        else:
            return Call(e.func_var, tuple(args))

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        func_type: FuncType = stmt.func_var.type
        args = self.process_list(self.visit(stmt.args), func_type.param_types)
        if same_list(args, stmt.args):
            return stmt
        else:
            return LaunchKernelStmt(
                func_var=stmt.func_var,
                args=args,
                grid_dim=stmt.grid_dim,
                block_dim=stmt.block_dim,
                shared_mem=stmt.shared_mem_bytes,
                target=stmt.target,
            )

    def visit_AssignStmt(self, stmt: AssignStmt):
        value = self.process(stmt.value, stmt.var.type)
        if value is stmt.value:
            return stmt
        else:
            return AssignStmt(stmt.var, value)

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        value = self.process(stmt.value, get_base_type(self.type_infer(stmt.buf)))
        if value is stmt.value:
            return stmt
        else:
            return BufferStoreStmt(self.visit(stmt.buf), self.visit(stmt.indices), value)

    def visit_LetStmt(self, stmt: LetStmt):
        bind_values = self.process_list(stmt.bind_values, [bind_var.type for bind_var in stmt.bind_vars])
        body = self.visit(stmt.body)
        if same_list(bind_values, stmt.bind_values) and body is stmt.body:
            return stmt
        else:
            return LetStmt(stmt.bind_vars, bind_values, body)


class AddExplicitPointerCastPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        rewriter = AddExplicitCastRewriter()
        return rewriter.visit_Function(func)


def add_explicit_cast() -> FunctionPass:
    return AddExplicitPointerCastPass()
