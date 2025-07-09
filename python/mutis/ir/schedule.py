from typing import Optional, Dict, List, Union, Any
from hidet.ir.expr import Var
from mutis.ir.tile import GraphTile, TensorTile
from mutis.ir.layout import Layout
from mutis.ir.graph import Tensor, Operator
from mutis.vm.ir.program import BlockMapping


class Partition:

    INTER_BLOCK_REDUCE_PARTITION = 'inter_block_reduce'
    BLOCK_KIND = 'block'
    REDUCE_KIND = 'reduce'
    UNROLL_KIND = 'unroll'

    def __init__(self, members, axes: List[Var], kind: str):
        self.members: List[Union[Partition, Operator]] = members
        self.nodes: List[Operator] = []
        self.axes: List[Var] = axes
        self.kind: str = kind

        for member in self.members:
            if isinstance(member, Partition):
                self.nodes.extend(member.nodes)
            else:
                assert isinstance(member, Operator)
                self.nodes.append(member)

    def astext(self, op2name: Optional[Dict[Operator, str]] = None):
        if op2name is None:
            op2name = {op: str(idx) for idx, op in enumerate(self.nodes)}

        items = []
        for member in self.members:
            if isinstance(member, Operator):
                items.append(op2name[member])
            else:
                items.append(member.astext(op2name))
        return '{}{{{}: {}}}'.format(self.kind, ', '.join(str(item) for item in self.axes), ', '.join(items))

    def copy(self):
        return Partition(members=self.members.copy(), axes=self.axes.copy(), kind=self.kind)


class Variants:

    SUPPORTED_VARIANTS = {
        'Matmul': {
            'inst': {'mma', 'simt'},
            'warp_spatial': Any,  # (int, int, int)
            'warp_repeat': Any,  # (int, int, int)
            # specific to inst 'mma'
            'mma_inst': {
                mma_inst.format(vk=vec_k)
                for mma_inst in ['m16n8k16v{vk}_f16_f16', 'm16n8k16v{vk}_f16_f32', 'm16n8k16v{vk}_bf16_f32']
                for vec_k in [1, 2, 3, 4]
            },
            # specific to inst 'simt'
            'simt_thread_spatial': Any,  # (int, int)
            'simt_thread_repeat': Any,  # (int, int)
        },
        'Load': {
            'stages': {'gmem->regs', 'gmem->smem->regs'},
            'pipeline': {None, 2, 3, 4, 5},  # pipeline > 1 only supports inst 'cp.async'
            'shared_layout_hint': Any,  # None, or SharedLayout
            'g2s_layout_hint': Any,  # None, or Layout
        },
    }

    def __init__(self, op2variant: Dict[Operator, Dict[str, Any]]):
        self.op2variant: Dict[Operator, Dict[str, Any]] = op2variant

    def set_variant(self, op: Operator, name: str, value: Any):
        assert isinstance(op, Operator)
        cls_name = op.__class__.__name__
        if cls_name not in Variants.SUPPORTED_VARIANTS or name not in Variants.SUPPORTED_VARIANTS[cls_name]:
            raise ValueError('Unsupported variant {} for operator {}'.format(name, cls_name))
        candidates = Variants.SUPPORTED_VARIANTS[cls_name][name]
        if candidates is not Any and value not in candidates:
            raise ValueError(
                'Unsupported value {} for variant {} of operator {}, candidates: {}'.format(
                    value, name, cls_name, candidates
                )
            )

        if op not in self.op2variant:
            self.op2variant[op] = {}
        self.op2variant[op][name] = value

    def get_variant(self, op: Operator, name: str, default: Optional[Any] = None) -> Any:
        if not self.has_variant(op, name):
            return default
        return self.op2variant[op][name]

    def has_variant(self, op: Operator, name: str):
        if op not in self.op2variant:
            return False
        if name not in self.op2variant[op]:
            return False
        return True

    def copy(self):
        return Variants({k: v.copy() for k, v in self.op2variant.items()})


class Schedule:
    def __init__(self, *, anchor_op=None, num_warps=None, graph_tile=None, layouts=None, variants=None, partition=None):
        self.anchor_op: Optional[Operator] = anchor_op
        self.num_warps: Optional[int] = num_warps
        self.graph_tile: Optional[GraphTile] = graph_tile
        self.layouts: Optional[Dict[Tensor, Layout]] = layouts
        self.partition: Optional[Partition] = partition
        self.variants: Optional[Variants] = variants

    def copy(self):
        sch = Schedule(
            anchor_op=self.anchor_op if self.anchor_op is not None else None,
            num_warps=self.num_warps if self.num_warps is not None else None,
            graph_tile=self.graph_tile.copy() if self.graph_tile is not None else None,
            layouts=self.layouts.copy() if self.layouts is not None else None,
            variants=self.variants.copy() if self.variants is not None else None,
            partition=self.partition.copy() if self.partition is not None else None,
        )
        return sch

    def tile_of(self, tensor: Tensor) -> TensorTile:
        assert self.graph_tile is not None, 'graph_tile is None'
        return self.graph_tile.tensor2tile[tensor]
