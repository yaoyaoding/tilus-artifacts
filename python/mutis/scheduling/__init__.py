from typing import List, Optional, Sequence
import os
from mutis.ir.graph import Graph
from mutis.ir.schedule import Schedule
from mutis.scheduling import schedulers
from mutis.scheduling.scheduler import set_current_space
from mutis.scheduling.tiling import tile_graph
from mutis.scheduling.determine_layout import determine_layout
from mutis.scheduling.partition import partition_graph


def _dump_schedules(graph: Graph, schedules: Sequence[Schedule], stage: str, dump_dir: Optional[str] = None):
    if dump_dir is None:
        return
    os.makedirs(dump_dir, exist_ok=True)
    graph.dump_schedule_summary(schedules, summary_path=os.path.join(dump_dir, '{}.txt'.format(stage)))


def generate_schedules(space: int, graph: Graph, dump_ir_dir: Optional[str] = None) -> List[Schedule]:
    set_current_space(space)
    schedules = tile_graph(graph)
    _dump_schedules(graph, schedules, '0_tiling', dump_ir_dir)

    schedules = determine_layout(graph, schedules)
    _dump_schedules(graph, schedules, '1_layouts', dump_ir_dir)

    schedules = partition_graph(graph, schedules)
    _dump_schedules(graph, schedules, '2_partition', dump_ir_dir)

    return schedules
