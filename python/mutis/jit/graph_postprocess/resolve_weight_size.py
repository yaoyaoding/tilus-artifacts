from typing import List
from hidet.ir.expr import Var, Constant, DataType
from hidet.ir.type import base_type_of_pointer, sizeof
from hidet.ir.tools import infer_type
from hidet.ir.utils.index_transform import index_multiply, index_sum
from mutis.ops.ldst import Load
from mutis.ir.graph import Graph
from mutis.jit.graph_postprocess.base import GraphTransform


class WeightSizeResolverTransform(GraphTransform):
    def transform(self, graph: Graph) -> Graph:
        weight_params: List[Var] = [param for param in graph.params if graph.param2attrs[param].is_weight]

        for weight_param in weight_params:
            if graph.param2attrs[weight_param].weight_nbytes is not None:
                # skip
                continue
            loads: List[Load] = [node for node in graph.nodes if isinstance(node, Load) and node.ptr is weight_param]
            loads = [load for load in loads if all(isinstance(s, Constant) for s in load.shape + load.strides)]
            if len(loads) == 0:
                msg = (
                    f'Cannot resolve weight size for {weight_param} since there is not a load operator with static '
                    f'shape and strides to load this weight parameter'
                )
                raise ValueError(msg)
            nbytes: List[int] = []
            for load in loads:
                dtype = load.dtype
                shape: List[int] = [int(s) for s in load.shape]
                strides: List[int] = [int(s) for s in load.strides]
                assert all(s >= 0 for s in strides)
                indices = [a - 1 for a in shape]

                num_elements = int(index_sum(index_multiply(indices, strides)) + 1)
                if isinstance(dtype, DataType):
                    nbytes.append(num_elements * dtype.nbits // 8)  # to support sub-byte type
                else:
                    nbytes.append(num_elements * sizeof(dtype))

            if any(n != nbytes[0] for n in nbytes):
                msg = (
                    f'Cannot resolve weight size for {weight_param} since there are multiple load operators with '
                    f'different sizes for this weight parameter: {nbytes}'
                )
                raise ValueError(msg)
            graph.param2attrs[weight_param].weight_nbytes = nbytes[0]
        return graph


def resolve_weight_size_transform() -> WeightSizeResolverTransform:
    return WeightSizeResolverTransform()
