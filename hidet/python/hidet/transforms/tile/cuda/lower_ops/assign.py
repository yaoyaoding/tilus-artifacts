from typing import List, Union, Optional

from hidet.ir.expr import Expr
from hidet.ir.tile.ops.assign import Assign
from .registry import TileOpImpl, Buffer, register_impl


@register_impl(Assign)
class AssignImpl(TileOpImpl):
    def implement(self, op: Optional[Assign], args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        dst: Buffer = args[0]
        src: Buffer = args[1]

        assert dst.layout == src.layout and src.scope.is_register()

        def f_compute(local_indices, global_indices, not_duplicated):
            return src.at_local(local_indices)

        self.iterate_dist_buffer_and_compute(dst, f_compute)
