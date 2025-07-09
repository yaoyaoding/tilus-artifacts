from __future__ import annotations

from collections import defaultdict
from typing import Optional, Union, Dict, Any, List, Type, Tuple

from hidet.ir.expr import Var, Expr, convert
from hidet.ir.tools import TypeInfer
from mutis.ir.graph import Graph, Operator, Tensor
from mutis.ir.layout import Layout
from mutis.ir.schedule import Schedule, Partition, Variants, GraphTile
from mutis.ir.tile import TensorTile
from mutis.vm.ir.value import Value
from mutis.vm.ir.inst import Instruction
from mutis.vm.ir.program import VirtualMachineProgram, WeightTransform
from mutis.vm.ir.stmt import Stmt, SeqStmt, ForStmt
from mutis.vm.ir.value import RegisterValue
from mutis.exceptions import CompilationFailedError
from mutis.target import Target, gpgpu_any, get_current_target


class NotSupportedEmitter(Exception):
    pass


class BaseEmitter:
    # op -> (priority -> emitter)
    REGISTRY: Dict[Type[Operator], Dict[Target, List[EmitterRegistry]]] = defaultdict(lambda: defaultdict(list))

    def __init__(self, codegen, op: Operator, variant: Dict[str, Any]):
        self.codegen: Codegen = codegen
        self.op: Operator = op
        self.variant: Dict[str, Any] = variant

    # utility methods used to simplify emitter implementation
    @property
    def tensor2value(self) -> Dict[Tensor, Value]:
        return self.codegen.tensor2value

    @property
    def tensor2layout(self) -> Dict[Tensor, Layout]:
        return self.codegen.tensor2layout

    @property
    def tensor2tile(self) -> Dict[Tensor, TensorTile]:
        return self.codegen.tensor2tile

    @property
    def graph(self):
        return self.codegen.graph

    @property
    def schedule(self):
        return self.codegen.schedule

    @property
    def num_tiles_map(self):
        return self.schedule.graph_tile.num_tiles_map

    # methods should be implemented by subclasses
    def emit_weight_transform(self) -> Optional[Tuple[Var, List[WeightTransform]]]:
        return None

    def emit(self) -> Union[Stmt, Instruction]:
        """
        Emit the operator to a sequence of instructions or statements
        """
        raise NotImplementedError(self.__class__.__name__)


class GroupReduceEmitter:
    def __init__(self, emitters: List[ReduceEmitter]):
        self.emitters: List[ReduceEmitter] = emitters

    def init(self):
        pass

    def emit_reduce_prologue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        return None

    def emit_reduce_body_prologue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        return None

    def emit_reduce_body_epilogue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        return None

    def emit_reduce_epilogue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        return None


class GroupUnrollEmitter:
    def __init__(self, emitters: List[UnrollEmitter]):
        self.emitters: List[UnrollEmitter] = emitters

    def init(self):
        pass

    def emit_unroll_prologue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        return None

    def emit_unroll_body_prologue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        return None

    def emit_unroll_body_epilogue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        return None

    def emit_unroll_epilogue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        return None


class InterBlockReduceEmitter(BaseEmitter):
    def emit_inter_block_reduce_prologue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        return None

    def emit_inter_block_reduce_epilogue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        return None


class ReduceEmitter(BaseEmitter):
    @staticmethod
    def emitter_group_class() -> Optional[Type[GroupReduceEmitter]]:
        return None

    def emit_reduce_prologue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        return None

    def emit_reduce_body_prologue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        return None

    def emit_reduce_body_epilogue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        return None

    def emit_reduce_epilogue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        return None


class UnrollEmitter(BaseEmitter):
    @staticmethod
    def emitter_group_class() -> Optional[Type[GroupUnrollEmitter]]:
        return None

    def emit_unroll_prologue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        return None

    def emit_unroll_body_prologue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        return None

    def emit_unroll_body_epilogue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        return None

    def emit_unroll_epilogue(self, axis: Var) -> Optional[Union[Stmt, Instruction]]:
        return None


class EmitterRegistry:
    def __init__(self, priority: int, variant: Dict[str, Any], emitter_cls: Type[BaseEmitter]):
        self.priority: int = priority
        self.variant: Dict[str, Any] = variant
        self.emitter_cls: Type[BaseEmitter] = emitter_cls

    def match(self, variant: Dict[str, Any]):
        for name, value in self.variant.items():
            if name not in variant or value != variant[name]:
                return False
        return True


def register_emitter(
    op_cls: Type[Operator],
    *,
    priority: int = 0,
    variant: Optional[Dict[str, Any]] = None,
    target: Optional[Target] = None,
):
    if target is None:
        target = gpgpu_any

    def decorator(emitter_cls: Type[BaseEmitter]):
        registry = EmitterRegistry(priority=priority, variant=variant if variant else {}, emitter_cls=emitter_cls)
        BaseEmitter.REGISTRY[op_cls][target].append(registry)
        return emitter_cls

    return decorator


class Codegen:
    def __init__(self):
        self.type_infer = TypeInfer()
        self.current_partition_kind: Optional[str] = None
        self.ctx: Dict[Operator, Dict[str, Any]] = defaultdict(dict)

        self.graph: Optional[Graph] = None
        self.schedule: Optional[Schedule] = None
        self.variants: Optional[Variants] = None
        self.graph_tile: Optional[GraphTile] = None
        self.tensor2tile: Optional[Dict[Tensor, TensorTile]] = None
        self.tensor2layout: Optional[Dict[Tensor, Layout]] = None
        self.tile_size_map: Optional[Dict[Var, int]] = None
        self.num_tiles_map: Optional[Dict[Var, Expr]] = None
        self.tensor2value: Dict[Tensor, RegisterValue] = {}

        self.param2transforms: Dict[Var, List[WeightTransform]] = {}
        self.op2emitter: Dict[Operator, BaseEmitter] = {}  # each operator instance has an emitter

    def get_emitter(self, op: Operator):
        if op not in self.op2emitter:

            def create_emitter() -> BaseEmitter:
                cur_target = get_current_target()
                op_cls = type(op)
                valid_registries: List[EmitterRegistry] = []
                for target, registries in BaseEmitter.REGISTRY[op_cls].items():
                    if target.supports(cur_target):
                        valid_registries.extend(registries)
                sorted_registries = sorted(valid_registries, key=lambda reg: reg.priority, reverse=True)
                variant = self.variants.op2variant.get(op, {})
                for registry in sorted_registries:
                    if not registry.match(variant):
                        continue
                    emitter_class = registry.emitter_cls
                    try:
                        emitter = emitter_class(self, op, variant)
                    except NotSupportedEmitter:
                        # this emitter does not support this variant
                        continue
                    return emitter
                raise CompilationFailedError(
                    'can not find an emitter that supports:\n'
                    + '  operator: {}\n'.format(op)
                    + '  variant: {}\n'.format(variant)
                    + ('  tiling: {}\n'.format(self.tensor2tile[op.output]) if op.output else '')
                )

            self.op2emitter[op] = create_emitter()
        return self.op2emitter[op]

    def generate(self, graph: Graph, schedule: Schedule) -> VirtualMachineProgram:
        self.graph = graph
        self.schedule = schedule
        self.variants = schedule.variants
        self.graph_tile = schedule.graph_tile
        self.tensor2tile: Dict[Tensor, TensorTile] = schedule.graph_tile.tensor2tile
        self.tensor2layout: Dict[Tensor, Layout] = schedule.layouts
        self.tile_size_map: Dict[Var, int] = schedule.graph_tile.tile_size_map
        self.num_tiles_map: Dict[Var, Union[Expr, int]] = schedule.graph_tile.num_tiles_map
        body = self.generate_partition(schedule.partition)

        assert schedule.graph_tile.block_mapping is not None, 'block mapping has not been set in schedule'
        return VirtualMachineProgram(
            name=graph.name,
            params=graph.params,
            param2attrs=graph.param2attrs,
            num_warps=schedule.num_warps,
            block_axes=schedule.graph_tile.block_axes,
            num_blocks=[self.num_tiles_map[axis] for axis in schedule.graph_tile.block_axes],
            body=body,
            block_mapping=schedule.graph_tile.block_mapping,
            weight_transforms=self.param2transforms,
            var2divisibility={},
            annotations={},
        )

    def generate_partition(self, partition: Partition) -> Stmt:
        old_kind = self.current_partition_kind
        self.current_partition_kind = partition.kind
        if partition.kind == Partition.BLOCK_KIND:
            ret = self.generate_block_partition(partition)
        elif partition.kind == Partition.INTER_BLOCK_REDUCE_PARTITION:
            ret = self.generate_inter_block_reduce_partition(partition)
        elif partition.kind == Partition.REDUCE_KIND:
            ret = self.generate_reduce_partition(partition)
        elif partition.kind == Partition.UNROLL_KIND:
            ret = self.generate_unroll_partition(partition)
        else:
            raise ValueError()
        self.current_partition_kind = old_kind
        return ret

    def generate_block_partition(self, partition) -> Stmt:
        seq = []
        for op in partition.nodes:
            emitter = self.get_emitter(op)
            emitted = emitter.emit_weight_transform()
            if emitted is not None:
                param, weight_transforms = emitted
                if param not in self.param2transforms:
                    self.param2transforms[param] = []
                self.param2transforms[param].extend(weight_transforms)
        for node in partition.members:
            if isinstance(node, Partition):
                seq.append(self.generate_partition(node))
            elif isinstance(node, Operator):
                emitter = self.get_emitter(node)
                seq.append(emitter.emit())
            else:
                assert False
        return SeqStmt(seq=seq)

    def generate_inter_block_reduce_partition(self, partition) -> Stmt:
        inter_block_reduce_axis: Var = partition.axes[0]
        seq: List[Union[Stmt, Instruction]] = []

        for node in partition.nodes:
            emitter = self.get_emitter(node)
            if isinstance(emitter, InterBlockReduceEmitter):
                emitted = emitter.emit_inter_block_reduce_prologue(axis=inter_block_reduce_axis)
                if emitted is not None:
                    seq.append(emitted)

        for node in partition.members:
            if isinstance(node, Partition):
                seq.append(self.generate_partition(node))
            elif isinstance(node, Operator):
                emitter = self.get_emitter(node)
                seq.append(emitter.emit())
            else:
                assert False

        for node in partition.nodes:
            emitter = self.get_emitter(node)
            if isinstance(emitter, InterBlockReduceEmitter):
                emitted = emitter.emit_inter_block_reduce_epilogue(axis=inter_block_reduce_axis)
                if emitted is not None:
                    seq.append(emitted)

        return SeqStmt(seq)

    def generate_reduce_partition(self, partition) -> Stmt:
        reduce_axis: Var = partition.axes[0]
        iter_var: Var = reduce_axis
        extent: Expr = convert(self.num_tiles_map[partition.axes[0]])
        prologue_seq: List[Union[Stmt, Instruction]] = []
        body_prologue_seq: List[Union[Stmt, Instruction]] = []
        body_seq: List[Union[Stmt, Instruction]] = []
        body_epilogue_seq: List[Union[Stmt, Instruction]] = []
        epilogue_seq: List[Union[Stmt, Instruction]] = []

        # get emitters for operator and operator group
        op_emitters: List[BaseEmitter] = [self.get_emitter(op) for op in partition.nodes]
        group2emitter: Dict[Type[GroupReduceEmitter], GroupReduceEmitter] = {}
        reduce_emitters: List[Union[ReduceEmitter, GroupReduceEmitter]] = []
        for emitter in op_emitters:
            if isinstance(emitter, ReduceEmitter):
                group_class: Optional[Type[GroupReduceEmitter]] = emitter.emitter_group_class()
                if group_class is None:
                    reduce_emitters.append(emitter)
                else:
                    if group_class not in group2emitter:
                        group2emitter[group_class] = group_class([])
                        reduce_emitters.append(group2emitter[group_class])
                    group2emitter[group_class].emitters.append(emitter)

        for group_emitter in group2emitter.values():
            group_emitter.init()

        # reduce prologue
        for emitter in reduce_emitters:
            emitted = emitter.emit_reduce_prologue(axis=reduce_axis)
            if emitted is not None:
                prologue_seq.append(emitted)
        for emitter in reduce_emitters:
            emitted = emitter.emit_reduce_body_prologue(axis=reduce_axis)
            if emitted is not None:
                body_prologue_seq.append(emitted)
        for node in partition.members:
            if isinstance(node, Partition):
                body_seq.append(self.generate_partition(node))
            elif isinstance(node, Operator):
                emitter = self.get_emitter(node)
                body_seq.append(emitter.emit())
            else:
                assert False
        for emitter in reduce_emitters:
            emitted = emitter.emit_reduce_body_epilogue(axis=reduce_axis)
            if emitted is not None:
                body_epilogue_seq.append(emitted)
        for emitter in reduce_emitters:
            emitted = emitter.emit_reduce_epilogue(axis=reduce_axis)
            if emitted is not None:
                epilogue_seq.append(emitted)
        for_stmt = ForStmt(
            iter_var=iter_var,
            extent=extent,
            body=SeqStmt(body_prologue_seq + body_seq + body_epilogue_seq),
            unroll_factor=1,
        )
        return SeqStmt(prologue_seq + [for_stmt] + epilogue_seq)

    def generate_unroll_partition(self, partition) -> Stmt:
        assert len(partition.axes) == 1
        unroll_axis = partition.axes[0]

        # do similar things as reduce partition
        iter_var: Var = unroll_axis
        extent: Expr = convert(self.num_tiles_map[unroll_axis])
        prologue_seq = []
        body_prologue_seq = []
        body_seq = []
        body_epilogue_seq = []
        epilogue_seq = []

        # get emitters for operator and operator group
        op_emitters: List[BaseEmitter] = [self.get_emitter(op) for op in partition.nodes]
        group2emitter: Dict[Type[GroupUnrollEmitter], GroupUnrollEmitter] = {}
        unroll_emitters: List[Union[GroupUnrollEmitter, UnrollEmitter]] = []
        for emitter in op_emitters:
            if isinstance(emitter, UnrollEmitter):
                group_class = emitter.emitter_group_class()
                if group_class is None:
                    unroll_emitters.append(emitter)
                else:
                    if group_class not in group2emitter:
                        group2emitter[group_class] = group_class([])
                        unroll_emitters.append(group2emitter[group_class])
                    group2emitter[group_class].emitters.append(emitter)

        for group_emitter in group2emitter.values():
            group_emitter.init()

        # unroll prologue
        for emitter in unroll_emitters:
            emitted = emitter.emit_unroll_prologue(axis=unroll_axis)
            if emitted is not None:
                prologue_seq.append(emitted)
        for emitter in unroll_emitters:
            emitted = emitter.emit_unroll_body_prologue(axis=unroll_axis)
            if emitted is not None:
                body_prologue_seq.append(emitted)
        for node in partition.members:
            if isinstance(node, Operator):
                emitter = self.get_emitter(node)
                body_seq.append(emitter.emit())
            else:
                assert False
        for emitter in unroll_emitters:
            emitted = emitter.emit_unroll_body_epilogue(axis=unroll_axis)
            if emitted is not None:
                body_epilogue_seq.append(emitted)
        for emitter in unroll_emitters:
            emitted = emitter.emit_unroll_epilogue(axis=unroll_axis)
            if emitted is not None:
                epilogue_seq.append(emitted)
        for_stmt = ForStmt(
            iter_var=iter_var,
            extent=extent,
            body=SeqStmt(body_prologue_seq + body_seq + body_epilogue_seq),
            unroll_factor=-1,
        )
        return SeqStmt(prologue_seq + [for_stmt] + epilogue_seq)

        # iter_var: Var = unroll_axis
        # extent: Expr = convert(self.num_tiles_map[unroll_axis])
        # body_seq: List[Union[Stmt, Instruction]] = []
        # for node in partition.nodes:
        #     emitter = self.get_emitter(node)
        #     emitted = emitter.emit()
        #     if emitted:
        #         body_seq.append(emitted)
        # for_stmt = ForStmt(iter_var=iter_var, extent=extent, body=SeqStmt(body_seq), unroll=None)
        # return for_stmt


def generate_virtual_machine_program(graph: Graph, schedule: Schedule) -> VirtualMachineProgram:
    codegen = Codegen()
    return codegen.generate(graph, schedule)
