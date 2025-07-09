from typing import List, Optional

from mutis.ir import Operator
from mutis.ir.layout import Layout
from mutis.ir.schedule import Schedule
from mutis.ops.transform import Cast
from mutis.scheduling.scheduler import BaseScheduler, register_scheduler
from mutis.scheduling.schedulers.arthmatic import ElementwiseBaseScheduler


@register_scheduler(Cast)
class CastScheduler(ElementwiseBaseScheduler):
    pass
