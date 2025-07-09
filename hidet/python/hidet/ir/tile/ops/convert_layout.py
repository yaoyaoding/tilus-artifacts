from typing import List, Union, Optional

from hidet.ir.expr import Expr
from hidet.ir.tile.expr import TileOp
from hidet.ir.tile.type import tile_type, TileLayout, TileType, TileScope, resolve_scope
from hidet.ir.type import BaseType


class ConvertLayout(TileOp):
    def __init__(self, x: Expr, layout: TileLayout, scope: Optional[Union[TileScope, str]] = None):
        super().__init__(args=[x], attrs={"layout": layout, "scope": scope})
        self.x: Expr = x
        self.layout: TileLayout = layout
        self.scope: TileScope = resolve_scope(scope, layout)

    @property
    def var_name_hint(self):
        return 'cvt'

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        a_type = arg_types[0]
        assert isinstance(a_type, TileType)
        return tile_type(elem_type=a_type.type, shape=a_type.shape, scope=self.scope, layout=self.layout)


def convert_layout(x: Expr, layout: TileLayout, scope: Optional[Union[TileScope, str]] = None):
    return ConvertLayout(x, layout, scope).make_call()
