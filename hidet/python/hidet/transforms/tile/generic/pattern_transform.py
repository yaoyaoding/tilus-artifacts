from typing import List, Dict, Optional

from hidet.ir.expr import Expr, Var
from hidet.ir.tile.expr import CallTileOp
from hidet.ir.tile.ops import Dot
from hidet.ir.func import Function
from hidet.transforms.base import TileFunctionPass
from hidet.transforms.tile.utils.pattern import PatternTransform, Pattern
from .dead_code_elimination import DeadCodeEliminationRewriter


class DotAddTransform(PatternTransform):
    """
    add(dot(a, b, 0), c) => dot(a, b, c)
    """

    def __init__(self):
        super().__init__()
        self.a = self.any_tile()
        self.b = self.any_tile()
        self.c = self.any_tile()
        self.zero = self.any_tile()

        self.dot_result = self.dot(self.a, self.b, self.zero)

        self.pattern = self.add(self.dot_result, self.c)

    def source(self) -> Pattern:
        return self.pattern

    def target(self, matched: Dict[Pattern, Expr], var2call: Dict[Var, CallTileOp]) -> Optional[CallTileOp]:
        a = matched[self.a]
        b = matched[self.b]
        c = matched[self.c]
        zero = matched[self.zero]
        dot = self.get_tile_op(self.dot_result, matched, var2call)
        dot_cls = dot.__class__
        if not self.is_zero(zero, var2call):
            return None
        return dot_cls(a, b, c).make_call()


class PatternTransformPass(TileFunctionPass):
    def __init__(self, transforms: List[PatternTransform]):
        super().__init__()
        self.transforms: List[PatternTransform] = transforms

    def process_tile_func(self, func: Function) -> Function:
        rewriter = DeadCodeEliminationRewriter()
        func = self.apply_transforms(func, self.transforms, repeat_limit=-1)
        func = rewriter(func)
        return func


def pattern_transform_pass() -> TileFunctionPass:
    transforms = [DotAddTransform()]
    return PatternTransformPass(transforms)
