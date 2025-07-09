from __future__ import annotations
from typing import List, Dict, Union
from hidet.ir.node import Node
from hidet.ir.type import BaseType
from hidet.ir.expr import Expr, convert
from hidet.ir.tile.type import TileLayout


_ScalarConst = Union[str, int, float, bool, BaseType, TileLayout]
CConst = Union[_ScalarConst, List[_ScalarConst]]  # compile-time constant


class TileOp(Node):
    def __init__(self, args: List[Expr] = None, attrs: Dict[str, CConst] = None):
        self.args: List[Expr] = args if args is not None else []
        self.attrs: Dict[str, CConst] = attrs if attrs is not None else {}
        self.annotations: Dict[str, CConst] = {}

        # canonicalize args
        self.args = [arg if isinstance(arg, Expr) else convert(arg) for arg in self.args]

        # annotations are different from attrs:
        #        attrs: attrs will determine the semantics of the operator
        #  annotations: annotations are used to pass information between passes
        # we introduce annotations to avoid polluting the attrs with information that is only used in passes
        # we will also show the annotations when we print the IR

    @classmethod
    def op_name(cls):
        # camel to snake (e.g., CamelName -> camel_name)
        camel_name = cls.__name__
        snake_name = "".join(["_" + c.lower() if c.isupper() else c for c in camel_name]).lstrip("_")
        return snake_name

    @property
    def name(self):
        return self.op_name()

    @property
    def var_name_hint(self):
        return self.name

    def reforward(
        self, args: List[Expr], attr_update: Dict[str, CConst] = None, annotations_update: Dict[str, CConst] = None
    ) -> TileOp:
        attrs = self.attrs.copy()
        annotations = self.annotations.copy()
        if attr_update is not None:
            attrs.update(attr_update)
        if annotations_update is not None:
            annotations.update(annotations_update)
        ret = self.__class__(*args, **attrs)
        ret.annotations = annotations
        return ret

    def make_call(self):
        return CallTileOp(self)

    def write_memory_op(self) -> bool:
        return False

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        raise NotImplementedError(
            "'infer_type' method has not been implemented for the following operator: \n{}".format(type(self).__name__)
        )


class CallTileOp(Expr):
    def __init__(self, op: TileOp):
        self.op: TileOp = op


def call_tile_op(top: TileOp):
    return CallTileOp(top)
