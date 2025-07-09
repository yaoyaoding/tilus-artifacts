from typing import Dict, Any
from mutis.ir import Operator, Graph
from mutis.ops.arithmatic import ElementwiseUnary, ElementwiseBinary
from mutis.ops.attention import Attention
from mutis.ops.matmul import Matmul
from mutis.ops.ldst import Load, Store
from hidet.utils import same_list


class Functor:
    def __init__(self):
        self.memo: Dict[Operator, Any] = {}

    def visit(self, node):
        if node in self.memo:
            return self.memo[node]

        # get the method to dispatch the node
        assert isinstance(node, Operator), 'node should be an instance of Operator'
        class_name = node.__class__.__name__
        method_name = 'visit_{}'.format(class_name)
        method = getattr(self, method_name, None)
        if method is None:
            raise NotImplementedError('Can not dispatch node {}'.format(class_name))

        result = method(node)
        self.memo[node] = result
        return result

    def visit_Graph(self, graph: Graph):
        nodes = list(reversed([self.visit(nd) for nd in reversed(graph.nodes)]))
        if same_list(nodes, graph.nodes):
            return graph
        else:
            return Graph(graph.name, graph.params, nodes)

    def visit_Operator(self, node: Operator):
        raise NotImplementedError()

    def visit_ElementwiseUnary(self, node: ElementwiseUnary):
        return self.visit_Operator(node)

    def visit_ElementwiseBinary(self, node: ElementwiseBinary):
        return self.visit_Operator(node)

    def visit_Attention(self, node: Attention):
        return self.visit_Operator(node)

    def visit_Matmul(self, node: Matmul):
        return self.visit_Operator(node)

    def visit_Load(self, node: Load):
        return self.visit_Operator(node)

    def visit_Store(self, node: Store):
        return self.visit_Operator(node)
