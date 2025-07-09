from hidet.utils import same_list
from mutis.ir import Operator
from mutis.ops.arithmatic import ElementwiseUnary, ElementwiseBinary
from mutis.ops.attention import Attention
from mutis.ops.matmul import Matmul
from mutis.ops.ldst import Load, Store
from .functor import Functor


class Rewriter(Functor):
    def __init__(self):
        super().__init__()

    def visit_Operator(self, node: Operator):
        inputs = [self.visit(v) for v in node.inputs]
        if same_list(inputs, node.inputs):
            return node
        else:
            return node.reforward(inputs, update_attrs={})
