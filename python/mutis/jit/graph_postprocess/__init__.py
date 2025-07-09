from mutis.ir.graph import Graph
from mutis.jit.graph_postprocess.resolve_weight_size import resolve_weight_size_transform


def graph_postprocess(graph: Graph) -> Graph:
    transforms = [resolve_weight_size_transform()]

    for transform in transforms:
        graph = transform.transform(graph)
    return graph
