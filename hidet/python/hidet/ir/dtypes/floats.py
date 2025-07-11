# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any
from functools import cached_property
from dataclasses import dataclass
import warnings
import numpy as np
from hidet.ir.type import DataType


@dataclass
class FloatInfo:
    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float
    dtype: DataType


class FloatType(DataType):
    def __init__(self, name, short_name, nbytes, min_value, max_value, eps, smallest_normal):
        super().__init__(name, short_name, nbytes)

        self._min_value: float = min_value
        self._max_value: float = max_value
        self._eps: float = eps
        self._smallest_normal: float = smallest_normal

    def is_integer_subbyte(self) -> bool:
        return False

    def is_float_subbyte(self) -> bool:
        return False

    def is_float(self) -> bool:
        return True

    def is_integer(self) -> bool:
        return False

    def is_complex(self) -> bool:
        return False

    def is_vector(self) -> bool:
        return False

    def is_subbyte(self) -> bool:
        return False

    def constant(self, value: Any):
        from hidet.ir.expr import Constant, constant

        if isinstance(value, Constant):
            value = value.value
        value = float(value)

        if value > self._max_value:
            warnings.warn(
                (
                    'Constant value {} is larger than the maximum value {} of data type {}. '
                    'Truncated to maximum value of {}.'
                ).format(value, self._max_value, self.name, self.name)
            )
            value = self._max_value

        if value < self._min_value:
            warnings.warn(
                (
                    'Constant value {} is smaller than the minimum value {} of data type {}. '
                    'Truncated to minimum value of {}.'
                ).format(value, self._min_value, self.name, self.name)
            )
            value = self._min_value

        return constant(value, self)

    @cached_property
    def one(self):
        return self.constant(1.0)

    @cached_property
    def zero(self):
        return self.constant(0.0)

    @property
    def min_value(self):
        return self.constant(self._min_value)

    @property
    def max_value(self):
        return self.constant(self._max_value)

    @property
    def exponent_nbits(self):
        # todo: refactor the FloatType to make everyone has _exponent_nbits and _mantissa_nbits
        name2result = {'float64': 11, 'float32': 8, 'float16': 5, 'bfloat16': 8, 'float8_e5m2': 5, 'float8_e4m3': 4}
        return name2result[self.name]

    @property
    def mantissa_nbits(self):
        name2result = {'float64': 52, 'float32': 23, 'float16': 10, 'bfloat16': 7, 'float8_e5m2': 2, 'float8_e4m3': 3}
        return name2result[self.name]

    def finfo(self) -> FloatInfo:
        return FloatInfo(
            bits=self.nbytes * 8,
            eps=self._eps,
            max=self._max_value,
            min=self._min_value,
            smallest_normal=self._smallest_normal,
            dtype=self,
        )


float8_e4m3 = FloatType(
    'float8_e4m3', 'f8e4m3', 1, min_value=float(-448), max_value=float(448), eps=2 ** (-2), smallest_normal=2 ** (-6)
)
float8_e5m2 = FloatType(
    'float8_e5m2',
    'f8e5m2',
    1,
    min_value=float(-57344),
    max_value=float(57344),
    eps=2 ** (-2),
    smallest_normal=2 ** (-14),
)
float16 = FloatType(
    'float16',
    'f16',
    2,
    np.finfo(np.float16).min,
    np.finfo(np.float16).max,
    np.finfo(np.float16).eps,
    np.finfo(np.float16).tiny,
)
float32 = FloatType(
    'float32',
    'f32',
    4,
    np.finfo(np.float32).min,
    np.finfo(np.float32).max,
    np.finfo(np.float32).eps,
    np.finfo(np.float32).tiny,
)
float64 = FloatType(
    'float64',
    'f64',
    8,
    np.finfo(np.float64).min,
    np.finfo(np.float64).max,
    np.finfo(np.float64).eps,
    np.finfo(np.float64).tiny,
)
bfloat16 = FloatType('bfloat16', 'bf16', 2, -3.4e38, 3.4e38, None, None)  # TODO: find correct values
tfloat32 = FloatType('tfloat32', 'tf32', 4, -3.4e38, 3.4e38, None, None)

f8e4m3 = float8_e4m3
f8e5m2 = float8_e5m2
f16 = float16
f32 = float32
f64 = float64
bf16 = bfloat16
tf32 = tfloat32
