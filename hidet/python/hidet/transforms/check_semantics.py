from typing import List, Sequence, Dict, Hashable
from hidet.ir.node import Node
from hidet.ir.type import BaseType, PointerType, TensorPointerType
from hidet.ir.expr import TensorElement, Expr
from hidet.ir.stmt import BufferStoreStmt
from hidet.ir.module import IRModule
from hidet.ir.functors import IRVisitor
from hidet.ir.tools import TypeInfer, IRPrinter
from hidet.utils.doc import Doc, Text, NewLine
from hidet.utils.py import red
from hidet.transforms.base import Pass


class DiagnosticIRPrinter(IRPrinter):
    def __init__(self, errors: Dict[Node, str]):
        super().__init__()
        self.errors: Dict = errors
        self.messages: List[str] = []

    def astext(self, node: Node):
        ret = super().astext(node)
        if len(self.messages) > 0:
            msg_doc = Doc()
            for idx, msg in enumerate(self.messages):
                msg_doc += NewLine() + Text('  {} {}'.format(red(idx, '[%{}]'), msg))
            ret = ret + msg_doc
            self.messages = []
        return ret

    def visit(self, node):
        ret = super().visit(node)
        if isinstance(node, Hashable) and node in self.errors:
            ret = ret + Text(red(' [%{}]'.format(len(self.messages))))
            self.messages.append(self.errors[node])
        return ret


class SemanticsChecker(IRVisitor):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()
        self.errors: Dict[Node, str] = {}

    def check(self, ir_module: IRModule):
        self.visit(ir_module)

        if len(self.errors) > 0:
            printer = DiagnosticIRPrinter(self.errors)
            print(printer.astext(ir_module), flush=True)
            raise ValueError('Found {} errors in semantics checking'.format(len(self.errors)))

    def append_error(self, node: Node, message: str):
        self.errors[node] = message

    def check_tensor_access(self, e: Node, buf: Expr, indices: Sequence[Expr]):
        buf_type: BaseType = self.type_infer(buf)

        if buf_type.is_tensor():
            ttype = buf_type.as_tensor_type()
            if len(indices) != len(ttype.shape):
                self.append_error(
                    e, 'Accessing a {}-rank tensor with {}-rank indices'.format(len(ttype.shape), len(indices))
                )
        elif buf_type.is_pointer():
            if isinstance(buf_type, PointerType):
                if len(indices) != 1:
                    self.append_error(e, 'Pointer access must have exactly one index')
            elif isinstance(buf_type, TensorPointerType):
                if len(indices) != len(buf_type.tensor_type.shape):
                    self.append_error(
                        e,
                        'Tensor pointer access expect {} indices, but got {}.'.format(
                            len(buf_type.tensor_type.shape), len(indices)
                        ),
                    )
        elif buf_type.is_array():
            if len(indices) != 1:
                self.append_error(e, 'Array access must have exactly one index')
        else:
            self.append_error(e, 'Indexing a non-tensor/non-pointer variable')

    def visit_TensorElement(self, e: TensorElement):
        super().visit_TensorElement(e)
        self.check_tensor_access(e, e.base, e.indices)

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        super().visit_BufferStoreStmt(stmt)
        self.check_tensor_access(stmt, stmt.buf, stmt.indices)


class SemanticsCheckerPass(Pass):
    def process_module(self, ir_module: IRModule) -> IRModule:
        checker = SemanticsChecker()
        checker.check(ir_module)
        return ir_module


def check_semantics_pass():
    return SemanticsCheckerPass()
