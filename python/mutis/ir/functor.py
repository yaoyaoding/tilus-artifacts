from typing import Dict, Union, Any
from hidet.ir.type import BaseType
from hidet.ir.expr import Expr
from mutis.ir.graph import Graph, Tensor, Operator
from mutis.utils import same_list


Node = Union[Tensor, Operator, Graph]


class GraphFunctor:
    def __init__(self):
        self.memo: Dict[int, Any] = {}

    def visit(self, node):
        key = id(node)
        if key in self.memo:
            return self.memo[key]

        if isinstance(node, Graph):
            ret = self.visit_Graph(node)
        elif isinstance(node, Tensor):
            ret = self.visit_Tensor(node)
        elif isinstance(node, Operator):
            ret = self.visit_Operator(node)
        elif isinstance(node, list):
            ret = self.visit_list(node)
        elif isinstance(node, tuple):
            ret = self.visit_tuple(node)
        elif isinstance(node, dict):
            ret = self.visit_dict(node)
        elif isinstance(node, (str, float, int, type(None))):
            ret = self.visit_PyConstant(node)
        elif isinstance(node, BaseType):
            ret = self.visit_Type(node)
        elif isinstance(node, Expr):
            ret = self.visit_Expr(node)
        else:
            raise ValueError(f"Unknown node type {type(node)}")

        self.memo[key] = ret
        return ret

    def visit_list(self, nodes):
        return [self.visit(node) for node in nodes]

    def visit_tuple(self, nodes):
        return tuple(self.visit_list(nodes))

    def visit_dict(self, nodes):
        return {key: self.visit(value) for key, value in nodes.items()}

    def visit_PyConstant(self, node):
        return node

    def visit_Type(self, node: BaseType):
        return node

    def visit_Expr(self, node: Expr):
        return node

    def visit_Graph(self, graph: Graph):
        raise NotImplementedError()

    def visit_Tensor(self, tensor: Tensor):
        raise NotImplementedError()

    def visit_Operator(self, op: Operator):
        raise NotImplementedError()


class GraphVisitor(GraphFunctor):
    def visit_Graph(self, graph: Graph):
        self.visit(graph.nodes)
        self.visit(graph.params)

    def visit_Operator(self, op: Operator):
        self.visit(op.inputs)
        self.visit(op.attrs)

    def visit_Tensor(self, tensor: Tensor):
        self.visit(tensor.shape)
        self.visit(tensor.elem_type)
        self.visit(tensor.producer)


class GraphRewriter(GraphFunctor):
    def visit_Graph(self, graph: Graph):
        nodes = self.visit(graph.nodes)
        params = self.visit(graph.params)
        param2attrs = {self.visit(p): graph.param2attrs[p] for p in graph.param2attrs}
        if same_list(nodes, graph.nodes) and same_list(params, graph.params):
            return graph
        else:
            return Graph(name=graph.name, params=params, param2attrs=param2attrs, nodes=nodes)

    def visit_Operator(self, op: Operator):
        inputs = self.visit(op.inputs)
        attrs = self.visit(op.attrs)
        if same_list(inputs, op.inputs) and same_list(attrs.values(), op.attrs.values()):
            return op
        else:
            updated_op = op.reforward(inputs=inputs, update_attrs=attrs)
            self.memo[id(op.output)] = updated_op.output
            return updated_op

    def visit_Tensor(self, tensor: Tensor):
        if tensor.producer is None:
            return tensor
        else:
            producer: Operator = self.visit(tensor.producer)
            return producer.output
