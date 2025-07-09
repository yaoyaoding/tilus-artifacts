from hidet.ir.tile.layout import (
    TileLayout,
    SpatialLayout,
    RepeatLayout,
    AtomLayout,
    ComposedLayout,
    ParameterizedTileLayout,
)


class TileLayoutFunctor:
    def __call__(self, layout: TileLayout):
        return self.visit(layout)

    def visit(self, layout: TileLayout):
        if isinstance(layout, ComposedLayout):
            ret = self.visit_ComposedLayout(layout)
        elif isinstance(layout, SpatialLayout):
            ret = self.visit_SpatialLayout(layout)
        elif isinstance(layout, RepeatLayout):
            ret = self.visit_RepeatLayout(layout)
        elif isinstance(layout, AtomLayout):
            ret = self.visit_AtomLayout(layout)
        elif isinstance(layout, ParameterizedTileLayout):
            ret = self.visit_ParameterizedTileLayout(layout)
        else:
            raise NotImplementedError(layout.__class__)
        return ret

    def visit_SpatialLayout(self, layout: SpatialLayout):
        return self.visit_AtomLayout(layout)

    def visit_RepeatLayout(self, layout: RepeatLayout):
        return self.visit_AtomLayout(layout)

    def visit_AtomLayout(self, layout: AtomLayout):
        raise NotImplementedError()

    def visit_ComposedLayout(self, layout: ComposedLayout):
        raise NotImplementedError()

    def visit_ParameterizedTileLayout(self, layout: ParameterizedTileLayout):
        raise NotImplementedError()


class TileLayoutRewriter(TileLayoutFunctor):
    def visit_AtomLayout(self, layout: AtomLayout):
        return layout

    def visit_ComposedLayout(self, layout: ComposedLayout):
        outer = self.visit(layout.outer)
        inner = self.visit(layout.inner)
        if outer is layout.outer and inner is layout.inner:
            return layout
        else:
            return ComposedLayout(outer, inner)

    def visit_ParameterizedTileLayout(self, layout: ParameterizedTileLayout):
        updated_layout = self.visit(layout.layout)
        if updated_layout == layout.layout:
            return layout
        else:
            return updated_layout
