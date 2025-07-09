from typing import List, Union

from hidet.ir.expr import Expr
from hidet.ir.tile.type import TileScope
from hidet.ir.tile.ops.arthimatic import UnaryTileOp, BinaryTileOp
from .registry import TileOpImpl, Buffer, register_impl


@register_impl(UnaryTileOp)
class UnaryTileOpImpl(TileOpImpl):
    def implement(self, op: UnaryTileOp, args: List[Union[Buffer, Expr]], output: Buffer):
        src: Buffer = args[0]

        assert src.scope == TileScope.Register

        self.iterate_dist_buffer_and_compute(
            output, lambda local_indices, global_indices, not_duplicated: op.apply_scalar(src.at_local(local_indices))
        )


@register_impl(BinaryTileOp)
class BinaryTileOpImpl(TileOpImpl):
    def implement(self, op: BinaryTileOp, args: List[Union[Buffer, Expr]], output: Buffer):
        lhs: Buffer = args[0]
        rhs: Buffer = args[1]

        assert lhs.scope == rhs.scope == TileScope.Register, 'Scope not supported for op: {} ({} vs {})'.format(
            op, lhs.scope, rhs.scope
        )

        if lhs.layout == rhs.layout:
            self.iterate_dist_buffer_and_compute(
                output,
                lambda local_indices, global_indices, not_duplicated: op.apply_scalar(
                    lhs.at_local(local_indices), rhs.at_local(local_indices)
                ),
            )
        else:
            raise NotImplementedError()
