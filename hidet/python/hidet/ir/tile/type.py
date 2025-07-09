from __future__ import annotations

from typing import List, Optional, Union

from hidet.ir.tile.layout import TileLayout
from hidet.ir.stmt import DeclareScope as TileScope
from hidet.ir.type import BaseType, PointerType, DataType


def resolve_scope(given_scope: Optional[Union[TileScope, str]], layout: Optional[TileLayout]) -> Optional[TileScope]:
    if layout is None:
        return None

    if given_scope is None:
        if layout.num_workers() == 1:
            return TileScope.Shared
        else:
            return TileScope.Register
    else:
        assert not given_scope.is_shared() ^ (layout.num_workers() == 1)
        return given_scope


class TileType(BaseType):
    def __init__(
        self,
        elem_type: Union[PointerType, DataType],
        shape: List[int],
        layout: Optional[TileLayout] = None,
        scope: Optional[Union[TileScope, str]] = None,
    ):
        self.type: Union[PointerType, DataType] = elem_type
        self.shape: List[int] = shape
        self.layout: Optional[TileLayout] = layout
        self.scope: TileScope = resolve_scope(scope, layout)

        # if self.layout is not None and any(a != b for a, b in zip(self.shape, self.layout.logical_shape())):
        #     raise ValueError('Tile shape does not match layout logical shape: {} vs. {}'.format(self.shape, self.layout.logical_shape()))


def tile_type(
    elem_type,  # Union[PointerType, DataType]
    shape: List[int],
    layout: Optional[TileLayout] = None,
    scope: Optional[Union[TileScope, str]] = None,
):
    assert isinstance(elem_type, (PointerType, DataType))
    return TileType(elem_type, shape, layout, scope)
