from typing import List, Tuple

from hidet.ir.expr import Var, Expr
from hidet.ir.dtypes import int32
from hidet.ir.tools import rewrite, simplify_to_int


class Formula:
    def __init__(self, x: Var, y: Expr):
        self.x: Var = x
        self.y: Expr = y

    def __str__(self):
        return str(self.y)

    @staticmethod
    def from_lambda(func):
        x = Var('x', type=int32)
        y = func(x)
        return Formula(x, y)

    def satisfy(self, x: List[int], y: List[int]) -> bool:
        for x_val, y_expected in zip(x, y):
            y_actual = simplify_to_int(rewrite(self.y, {self.x: x_val}))
            if y_actual != y_expected:
                return False
        return True


predefined_formulas = [
    Formula.from_lambda(lambda x: x + 1),
    Formula.from_lambda(lambda x: x / 4),
    Formula.from_lambda(lambda x: x % 4),
    Formula.from_lambda(lambda x: x % 32 / 4),
    Formula.from_lambda(lambda x: x % 4 == 0),
    Formula.from_lambda(lambda x: x % 4 == 0),
    Formula.from_lambda(lambda x: x / 32),
    Formula.from_lambda(lambda x: 8 + x % 32 / 4),
    Formula.from_lambda(lambda x: 16 + x % 32 / 4),
    Formula.from_lambda(lambda x: 24 + x % 32 / 4),
    Formula.from_lambda(lambda x: x / 32 == 0),
    Formula.from_lambda(lambda x: x / 32 + x % 4 == 0),
    Formula.from_lambda(lambda x: x % 32 / 4 * 4),
    Formula.from_lambda(lambda x: 32 + x % 32 / 4 * 4),
    Formula.from_lambda(lambda x: 64 + x % 32 / 4 * 4),
    Formula.from_lambda(lambda x: 96 + x % 32 / 4 * 4),
    Formula.from_lambda(lambda x: 96 + x % 32 / 4 * 4),
    Formula.from_lambda(lambda x: x / 64 * 16),
    Formula.from_lambda(lambda x: x / 32 % 2 == 1),
    Formula.from_lambda(lambda x: x % 32 / 4 * 2 + x / 64 * 32),
    Formula.from_lambda(lambda x: x % 32 / 4 * 2 + x / 64 * 32 + 16),
    Formula.from_lambda(lambda x: x / 32 % 2 == 0),
    Formula.from_lambda(lambda x: x % 32 / 4 + x / 64 * 16),
    Formula.from_lambda(lambda x: x % 32 / 4 + x / 64 * 16 + 8),
    Formula.from_lambda(lambda x: x % 4 + x / 32 % 2 == 0),
]


def solve(x: List[int], y: List[int]) -> Tuple[Var, Expr]:
    for formula in predefined_formulas:
        if formula.satisfy(x, y):
            return formula.x, formula.y
    buf = ['can not solve:']
    buf.append('thread id: ' + ' '.join('{:3}'.format(v) for v in x))
    buf.append('  warp id: ' + ' '.join('{:3}'.format(v // 32) for v in x))
    buf.append('  lane id: ' + ' '.join('{:3}'.format(v % 32) for v in x))
    buf.append('   target: ' + ' '.join('{:3}'.format(v) for v in y))
    raise NotImplementedError('\n'.join(buf))
