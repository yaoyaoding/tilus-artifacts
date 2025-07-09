from typing import List, Dict, Optional

from mutis.ir.graph import Graph, Tensor
from mutis.ir.layout import Layout
from mutis.ir.schedule import Schedule
from mutis.scheduling.scheduler import BaseScheduler, resolve_scheduler


def determine_layout_for_anchor(graph: Graph, schedules: List[Schedule]) -> List[Schedule]:
    new_schedules = []
    for sch in schedules:
        anchor_op = sch.anchor_op
        scheduler = resolve_scheduler(anchor_op)
        new_schedules.extend(scheduler.determine_layout_as_anchor(graph, anchor_op, sch))
    return new_schedules


def propagate_layout(graph: Graph, schedules: List[Schedule]):

    for sch in schedules:
        tensor2layout: Dict[Tensor, Layout] = sch.layouts

        while True:
            updated: bool = False
            for op in graph.nodes:
                scheduler: BaseScheduler = resolve_scheduler(op)
                tensors = op.inputs + ([op.output] if op.output else [])
                layouts: List[Optional[Layout]] = [tensor2layout.get(x, None) for x in tensors]

                if all(layout is not None for layout in layouts):
                    # all tensors associated with this operator have layout
                    continue
                if all(layout is None for layout in layouts):
                    # all tensors associated with this operator do not have layout
                    continue

                updated_layouts = scheduler.propagate_layout(op, sch, layouts)

                for tensor, layout, updated_layout in zip(tensors, layouts, updated_layouts):
                    if layout and updated_layout:
                        assert layout is updated_layout, 'layout can not be changed'
                    elif layout is None and updated_layout:
                        tensor2layout[tensor] = updated_layout
                        updated = True

            if not updated:
                break

        # check if all tensors have layout
        for op in graph.nodes:
            tensors = op.inputs + ([op.output] if op.output else [])
            for tensor in tensors:
                if tensor not in tensor2layout:
                    raise ValueError(f'Tensor {tensor} does not have layout')

    return schedules


def determine_variant_for_load(graph: Graph, schedules: List[Schedule]) -> List[Schedule]:

    for op in graph.nodes:
        scheduler = resolve_scheduler(op)
        updated_schedules = []
        for sch in schedules:
            updated_schedules.extend(scheduler.determine_variant(graph, op, sch))
        schedules = updated_schedules

    return schedules


def determine_layout(graph: Graph, schedules: List[Schedule]) -> List[Schedule]:
    schedules = determine_layout_for_anchor(graph, schedules)
    schedules = propagate_layout(graph, schedules)
    schedules = determine_variant_for_load(graph, schedules)
    return schedules
