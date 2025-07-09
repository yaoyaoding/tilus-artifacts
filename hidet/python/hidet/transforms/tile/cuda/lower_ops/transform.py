from typing import List, Union, Optional

from hidet.ir.expr import Expr
from hidet.ir.tile.ops.transform import ExpandDims, Broadcast, CastOp
from .registry import TileOpImpl, Buffer, register_impl


@register_impl(ExpandDims)
class ExpandDimsImpl(TileOpImpl):
    def implement(self, op: ExpandDims, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        src: Buffer = args[0]
        dst: Buffer = output

        if src.is_flatten_block() and dst.is_block() and src.flatten_block_layout.parent == dst.layout:
            assert src.flatten_block_layout.axis == op.axis

            def f_compute(local_indices, global_indices, not_duplicated):
                return src.at_local(local_indices)

            self.iterate_dist_buffer_and_compute(dst, f_compute)
        else:
            raise NotImplementedError()


@register_impl(Broadcast)
class BroadcastImpl(TileOpImpl):
    def implement(self, op: Broadcast, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        src: Buffer = args[0]
        dst: Buffer = output

        broadcast_dims = [i for i in range(len(dst.shape)) if dst.shape[i] != src.shape[i]]

        if src.scope.is_register() and dst.scope.is_register() and src.layout == dst.layout:

            def f_compute(local_indices, global_indices, not_duplicated):
                local_indices = [idx if i not in broadcast_dims else 0 for i, idx in enumerate(local_indices)]
                return src[local_indices]

            self.iterate_dist_buffer_and_compute(dst, f_compute)
            assert False, 'Stale path'
        else:

            def f_compute(local_indices, logical_indices, not_duplicated):
                logical_indices = [idx if i not in broadcast_dims else 0 for i, idx in enumerate(logical_indices)]
                return src.at_logical(logical_indices)

            self.iterate_dist_buffer_and_compute(dst, f_compute)


@register_impl(CastOp)
class CastOpImpl(TileOpImpl):
    def implement(self, op: CastOp, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        src: Buffer = args[0]
        dst: Buffer = output
        if src.scope.is_register() and dst.scope.is_register() and src.layout == dst.layout:

            def f_compute(local_indices, global_indices, not_duplicated):
                return src.at_local(local_indices)

            self.iterate_dist_buffer_and_compute(dst, f_compute)
        else:
            raise NotImplementedError()
