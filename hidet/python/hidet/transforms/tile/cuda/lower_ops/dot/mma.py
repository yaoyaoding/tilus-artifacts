from typing import List, Union

from hidet.ir.expr import Expr
from hidet.ir.primitives.cuda import threadIdx
from hidet.ir.primitives.cuda.mma import mma_sync
from hidet.ir.tile.ops.dot import MmaDot
from hidet.ir.tile.layout import MmaOutputLayout, MmaDotOperandLayout
from hidet.ir.mapping import row_spatial
from ..registry import TileOpImpl, Buffer, register_impl


@register_impl(MmaDot)
class MmaDotImpl(TileOpImpl):
    def implement(self, op: MmaDot, args: List[Union[Buffer, Expr]], output: Buffer):
        a: Buffer = args[0]
        b: Buffer = args[1]
        c: Buffer = args[2]
        d: Buffer = output
        a_layout, b_layout, c_layout, d_layout = a.layout, b.layout, c.layout, d.layout

        # check layout
        assert (
            isinstance(a_layout, MmaDotOperandLayout)
            and isinstance(b_layout, MmaDotOperandLayout)
            and isinstance(c_layout, MmaOutputLayout)
            and isinstance(d_layout, MmaOutputLayout)
            and a_layout.mma == c_layout
            and b_layout.mma == c_layout
            and c_layout == d_layout
        )

        mma: MmaOutputLayout = d_layout

        assert len(mma.local_shape()) == 1
        with self.for_range(mma.local_shape()[0]) as s:
            self.local_store(d, [s], c.at_local([s]))

        assert mma.warps_k == 1
        warp_id = threadIdx.x // 32
        with self.for_mapping(row_spatial(mma.warps_m, mma.warps_n), worker=warp_id) as (wi, wj):
            with self.for_grid([mma.repeat_m, mma.repeat_n]) as (ri, rj):
                for rk in range(mma.repeat_k):
                    i = (wi * mma.repeat_m + ri) * mma.inst_m
                    j = (wj * mma.repeat_n + rj) * mma.inst_n
                    k = rk * mma.inst_k
                    a_ptr = ~a.at_logical([i, k])
                    b_ptr = ~b.at_logical([k, j])
                    c_ptr = ~d.at_logical([i, j])
                    self.append(mma_sync(mma.config.config, a_ptr, b_ptr, c_ptr))
