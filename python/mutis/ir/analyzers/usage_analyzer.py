from typing import Tuple, Dict, List
from collections import defaultdict
from mutis.ir.graph import Graph, Operator, Tensor


def analyze_usage(graph: Graph) -> Dict[Tensor, List[Tuple[Operator, int]]]:
    usage = defaultdict(list)
    for node in graph.nodes:
        assert isinstance(node, Operator)
        for i, t in enumerate(node.inputs):
            assert isinstance(t, Tensor)
            usage[t].append((node, i))
    return usage
