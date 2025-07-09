from typing import TypeVar, Generic
from hidet.ir.dtypes import float8_e4m3, float16, float32, bfloat16
from hidet.ir.type import void_p, void

T = TypeVar('T')


class ConstantType:
    def __init__(self, base_type):
        self.base_type = base_type


class ConstantTypeMaker(Generic[T]):
    def __getitem__(self, item: T) -> T:
        return ConstantType(item)


class WeightType:
    def __init__(self, base_pointer_type):
        self.base_pointer_type = base_pointer_type


class WeightTypeMaker(Generic[T]):
    def __getitem__(self, item: T) -> T:
        return WeightType(item)


class OptionalType:
    def __init__(self, base_type):
        self.base_type = base_type


class OptionalTypeMaker(Generic[T]):
    def __getitem__(self, item: T) -> T:
        return OptionalType(item)


constant = ConstantTypeMaker()
weight = WeightTypeMaker()
optional = OptionalTypeMaker()
