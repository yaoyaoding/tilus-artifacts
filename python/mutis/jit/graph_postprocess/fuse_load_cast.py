from typing import List
from hidet.ir.expr import Var, Constant, DataType
from hidet.ir.type import base_type_of_pointer, sizeof
from hidet.ir.tools import infer_type
from hidet.ir.utils.index_transform import index_multiply, index_sum
from mutis.ir import Operator
from mutis.ops.ldst import Load
from mutis.ir.graph import Graph
from mutis.ops.transform import Cast
from mutis.ops.ldst import Load
from mutis.jit.graph_postprocess.base import GraphTransform
from mutis.ir.functor import GraphRewriter
from mutis.ir.tools import eliminate_dead_code


class FuseLoadCastRewriter(GraphRewriter):
    def visit_Operator(self, op: Operator):
        if isinstance(op, Cast) and isinstance(op.inputs[0].producer, Load):
            load_op: Load = op.inputs[0].producer
            cast_op: Cast = op
            return Load(
                ptr=load_op.ptr,
                dtype=load_op.dtype,
                shape=load_op.shape,
                strides=load_op.strides,
                cast_dtype=cast_op.casted_type,
            )
        else:
            return super().visit_Operator(op)


class FuseLoadCastTransform(GraphTransform):
    def transform(self, graph: Graph) -> Graph:
        rewriter = FuseLoadCastRewriter()
        return eliminate_dead_code(rewriter.visit(graph))


def fuse_load_cast_transform() -> GraphTransform:
    return FuseLoadCastTransform()
