from typing import Sequence, List


class Vector:
    def __init__(self, init: Sequence[int]):
        self.v: List[int] = list(init)

    def __len__(self):
        return len(self.v)

    def __iter__(self):
        return iter(self.v)

    @staticmethod
    def _convert(n, other):
        if isinstance(other, Vector):
            assert len(other) == n
        elif isinstance(other, int):
            other = Vector([other] * n)
        elif isinstance(other, (list, tuple)):
            assert len(other) == n
            other = Vector(other)
        return other

    @staticmethod
    def _bin(lhs, op, rhs):
        if isinstance(lhs, Vector) and isinstance(rhs, Vector):
            assert len(lhs) == len(rhs)
        elif isinstance(lhs, Vector):
            rhs = Vector._convert(len(lhs), rhs)
        elif isinstance(rhs, Vector):
            lhs = Vector._convert(len(rhs), lhs)
        else:
            raise RuntimeError(f"Cannot {op} {lhs} and {rhs}")
        return Vector([op(x, y) for x, y in zip(lhs, rhs)])

    def __eq__(self, other):
        other = Vector._convert(len(self), other)
        return all(x == y for x, y in zip(self, other))

    def __add__(self, other):
        return Vector._bin(self, lambda x, y: x + y, other)

    def __sub__(self, other):
        return Vector._bin(self, lambda x, y: x - y, other)

    def __mul__(self, other):
        return Vector._bin(self, lambda x, y: x * y, other)

    def __floordiv__(self, other):
        return Vector._bin(self, lambda x, y: x // y, other)

    def __mod__(self, other):
        return Vector._bin(self, lambda x, y: x % y, other)

    def __radd__(self, other):
        return Vector._bin(other, lambda x, y: x + y, self)

    def __rsub__(self, other):
        return Vector._bin(other, lambda x, y: x - y, self)

    def __rmul__(self, other):
        return Vector._bin(other, lambda x, y: x * y, self)

    def __rfloordiv__(self, other):
        return Vector._bin(other, lambda x, y: x // y, self)

    def __rmod__(self, other):
        return Vector._bin(other, lambda x, y: x % y, self)
