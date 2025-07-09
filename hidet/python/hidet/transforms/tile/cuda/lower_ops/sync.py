from typing import List, Union, Optional

from hidet.ir.expr import Expr
from hidet.ir.primitives.cuda.sync import syncthreads
from hidet.ir.tile.ops.sync import SyncThreads
from .registry import TileOpImpl, Buffer, register_impl


@register_impl(SyncThreads)
class SyncThreadsImpl(TileOpImpl):
    def implement(self, op: SyncThreads, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        self.append(syncthreads())
