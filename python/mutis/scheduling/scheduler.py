from typing import List, Optional, Dict, Type, Tuple, Union, Sequence
from mutis.ir.graph import Operator, Graph
from mutis.ir.schedule import GraphTile, Schedule
from mutis.ir.layout import Layout
from mutis.target import Target, gpgpu_any, get_current_target


class BaseScheduler:

    _current_space = 0

    def tile(self, graph: Graph, op: Operator) -> List[GraphTile]:
        raise NotImplementedError('Tiling for {}'.format(self.__class__.__name__))

    def determine_layout_as_anchor(self, graph: Graph, op: Operator, sch: Schedule) -> List[Schedule]:
        raise NotImplementedError('Determine layout for {}'.format(self.__class__.__name__))

    def propagate_layout(self, op: Operator, sch: Schedule, layouts: List[Optional[Layout]]) -> List[Optional[Layout]]:
        raise NotImplementedError('Propagate layout for {}'.format(self.__class__.__name__))

    def determine_variant(self, graph: Graph, op: Operator, sch: Schedule) -> List[Schedule]:
        # by default, no variant
        return [sch]


_scheduler_registry: Dict[Type[Operator], Dict[Target, BaseScheduler]] = {}


def register_scheduler(op_cls: Type[Operator], target: Optional[Target] = None):
    if target is None:
        target = gpgpu_any

    def decorator(scheduler_cls: Type[BaseScheduler]):
        if op_cls not in _scheduler_registry:
            _scheduler_registry[op_cls] = {}

        if target in _scheduler_registry[op_cls]:
            raise RuntimeError(f"Scheduler for {op_cls} and {target} already registered")
        _scheduler_registry[op_cls][target] = scheduler_cls()
        return scheduler_cls

    return decorator


def resolve_scheduler(op: Operator) -> BaseScheduler:
    op_cls = type(op)
    if op_cls not in _scheduler_registry:
        raise RuntimeError("Cannot resolve scheduler for {}.".format(op_cls.__name__))
    cur_target = get_current_target()
    available_targets = [
        target
        for target in _scheduler_registry[op_cls].keys()
        if target == gpgpu_any
        or target.kind == cur_target.kind
        and target.properties.compute_capability <= cur_target.properties.compute_capability
    ]
    if not available_targets:
        raise RuntimeError(
            "No available scheduler for {} on target {}, registered targets:\n{}".format(
                op_cls.__name__, cur_target, "\n".join([str(target) for target in _scheduler_registry[op_cls].keys()])
            )
        )
    found_target = max(
        available_targets, key=lambda target: (target != gpgpu_any, target.properties.compute_capability)
    )

    return _scheduler_registry[op_cls][found_target]


def set_current_space(space: int):
    BaseScheduler._current_space = space


def get_current_space() -> int:
    return BaseScheduler._current_space
