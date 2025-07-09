from typing import Optional

from hidet.ir import AssertStmt
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import SeqStmt
from hidet.transforms.base import FunctionPass


class FilterAssertionRewriter(IRRewriter):
    def __init__(self, filter_device_assertion: bool):
        super().__init__()
        self.current_func_kind: Optional[str] = None
        self.filter_device_assertion: bool = filter_device_assertion

    def visit_Function(self, func: Function):
        self.current_func_kind = func.kind
        return super().visit_Function(func)

    def visit_AssertStmt(self, stmt: AssertStmt):
        if self.filter_device_assertion and self.current_func_kind in ['cuda_internal', 'cuda_kernel']:
            return SeqStmt([])
        return stmt


class FilterAssertionPass(FunctionPass):
    def __init__(self, filter_device_assertion: bool):
        super().__init__()

        self.filter_device_assertion: bool = filter_device_assertion

    def process_func(self, func: Function) -> Function:
        rewriter = FilterAssertionRewriter(self.filter_device_assertion)
        func = rewriter(func)
        return func


def filter_assertion_pass(filter_device_assertion: bool = True):
    return FilterAssertionPass(filter_device_assertion)
