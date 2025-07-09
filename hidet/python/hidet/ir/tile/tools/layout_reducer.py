from typing import Optional
from hidet.ir.tile.layout import TileLayout, AtomLayout
from hidet.ir.tile.tools.layout_functor import TileLayoutRewriter
from hidet.utils import same_list


class LayoutReduceRewriter(TileLayoutRewriter):
    def __init__(self):
        self.dim: Optional[int] = None
        self.keep_dim: Optional[bool] = None

    def reduce(self, layout: TileLayout, dim: int, keep_dim: bool) -> TileLayout:
        self.dim = dim
        self.keep_dim = keep_dim
        return self.visit(layout)

    def visit_AtomLayout(self, layout: AtomLayout):
        squeeze_dims = layout.squeeze_dims.copy() if layout.squeeze_dims is not None else []
        dims = [i for i in range(len(layout.shape)) if i not in squeeze_dims]
        actual_dim = dims[self.dim]

        shape = [layout.shape[i] if i != actual_dim else 1 for i in range(len(layout.shape))]

        if self.keep_dim and same_list(shape, layout.shape):
            return layout
        else:
            if not self.keep_dim:
                squeeze_dims.append(dims[self.dim])
            return AtomLayout(
                shape=shape,
                worker_shape=layout.worker_shape,
                ranks=layout.ranks,
                worker_ranks=layout.worker_ranks,
                flatten_local=layout.flatten_local,
                squeeze_dims=squeeze_dims,
            )


def reduce_layout(layout: TileLayout, dim: int, keep_dim: bool = False) -> TileLayout:
    rewriter = LayoutReduceRewriter()
    return rewriter.reduce(layout, dim, keep_dim)
