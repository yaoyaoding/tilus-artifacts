from typing import List, Set

from hidet.utils import same_list
from mutis.ir import Graph, Operator
from mutis.ir.functor import GraphRewriter
from mutis.ops.ldst import Store


class DeadCodeEliminationRewriter(GraphRewriter):
    def visit_Graph(self, graph: Graph):
        anchor_classes = (Store,)
        anchor_operators: List[Operator] = []

        for node in graph.nodes:
            if isinstance(node, anchor_classes):
                anchor_operators.append(node)

        visited: Set[Operator] = set(anchor_operators)
        queue: List[Operator] = list(anchor_operators)

        while queue:
            op: Operator = queue.pop()

            for t in op.inputs:
                if t.producer and t.producer not in visited:
                    visited.add(t.producer)
                    queue.append(t.producer)

        updated_nodes = [node for node in graph.nodes if node in visited]

        if same_list(updated_nodes, graph.nodes):
            return graph
        else:
            return Graph(name=graph.name, params=graph.params, param2attrs=graph.param2attrs, nodes=updated_nodes)


def eliminate_dead_code(graph: Graph) -> Graph:
    return DeadCodeEliminationRewriter().visit(graph)
