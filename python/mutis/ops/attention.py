from typing import Tuple, Union, List
from hidet.ir.expr import Expr, convert
from hidet.ir.type import DataType, PointerType
from mutis.ir import Tensor, Operator
from mutis import utils


class Attention(Operator):
    """
    query: [bs, q_heads, q_tokens, head_size]
    key: [bs, kv_heads, kv_tokens, head_size]
    value: [bs, kv_heads, kv_tokens, head_size]
    causal: bool
    where q_heads % kv_heads == 0

    output: [bs, q_heads, q_tokens, head_size]
    """

    def __init__(self, query: Tensor, key: Tensor, value: Tensor, causal: bool):
        self.causal: bool = causal
        super().__init__(inputs=[query, key, value], attrs={'causal': causal})

    def infer_type(self) -> Tuple[Union[DataType, PointerType], List[Expr]]:
        query = self.get_input(0)
        key = self.get_input(1)
        value = self.get_input(2)
        assert len(query.shape) == 4, 'Expect query has 4 dimensions, got {}'.format(query.shape)
        assert len(key.shape) == 4, 'Expect key has 4 dimensions, got {}'.format(key.shape)
        assert len(value.shape) == 4, 'Expect value has 4 dimensions, got {}'.format(value.shape)
        assert (
            query.shape[0] == key.shape[0] == value.shape[0]
        ), 'Expect the batch size of query, key, and value are the same, got {}, {}, {}'.format(
            query.shape[0], key.shape[0], value.shape[0]
        )
        assert (
            query.shape[3] == key.shape[3] == value.shape[3]
        ), 'Expect the head size of query, key, and value are the same, got {}, {}, {}'.format(
            query.shape[3], key.shape[3], value.shape[3]
        )
        assert key.shape[1] == value.shape[1], 'Expect kv_heads of key and value are the same, got {}, {}'.format(
            key.shape[1], value.shape[1]
        )
        assert query.shape[1] % key.shape[1] == 0, 'Expect q_heads % kv_heads == 0, got {}, {}'.format(
            query.shape[1], key.shape[1]
        )
        assert (
            query.elem_type == key.elem_type == value.elem_type
        ), 'Expect query, key, and value have the same element type, got {}, {}, {}'.format(
            query.elem_type, key.elem_type, value.elem_type
        )

        return query.elem_type, list(query.shape)

    def tile_schemes(self):
        return [(['B, N', 'B, N', 'B, N', 'B, N'], {'B': [32, 64, 128, 256]})]


def attention(query: Tensor, key: Tensor, value: Tensor, causal: bool) -> Tensor:
    op = Attention(query, key, value, causal)
    GraphContext.current().append(op)
    return op.get_output()
