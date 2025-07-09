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
from typing import Dict
from hidet.ir.type import DataType
from .integer import int8, int16, int32, int64, uint8, uint16, uint32, uint64
from .integer import i8, i16, i32, i64, u8, u16, u32, u64
from .integer_subbyte import int7b, int6b, int5b, int4b, int3b, int2b, int1b
from .integer_subbyte import uint7b, uint6b, uint5b, uint4b, uint3b, uint2b, uint1b
from .integer_subbyte import i7, i6, i5, i4, i3, i2, i1, u7, u6, u5, u4, u3, u2, u1
from .floats import float8_e4m3, float8_e5m2, float16, float32, float64, bfloat16, tfloat32
from .floats import f8e4m3, f8e5m2, f16, f32, f64, bf16, tf32
from .floats_subbyte import float7_e5m1, float7_e4m2, float7_e3m3, float7_e2m4, float6_e4m1, float6_e3m2, float6_e2m3
from .floats_subbyte import float5_e3m1, float5_e2m2, float4_e2m1, float3_e1m1
from .floats_subbyte import f7e5m1, f7e4m2, f7e3m3, f7e2m4, f6e4m1, f6e3m2, f6e2m3, f5e3m1, f5e2m2, f4e2m1, f3e1m1
from .boolean import boolean
from .vector import float16x1, float16x2, float16x4, float16x8, float32x1, float32x2, float32x4, float32x8
from .vector import f16x1, f16x2, f16x4, f16x8, f32x1, f32x2, f32x4, f32x8, i4x2, i4x8, u4x2, u4x8
from .vector import int8x4, uint8x1, uint8x2, uint8x4, int4bx8, uint4bx8, vectorize
from .vector import uint16x1, uint16x2, uint16x4
from .vector import uint32x1, uint32x2, uint32x4
from .vector import uint64x1, uint64x2, uint64x4
from .complex import complex64, complex128
from .promotion import promote_type
from .utils import dtype_to_numpy, finfo, iinfo
from hidet.utils import initialize

registered_dtypes = (
    # float dtypes
    float64,
    float32,
    float16,
    tfloat32,
    bfloat16,
    float8_e5m2,
    float8_e4m3,
    float7_e5m1,
    float7_e4m2,
    float7_e3m3,
    float7_e2m4,
    float6_e4m1,
    float6_e3m2,
    float6_e2m3,
    float5_e3m1,
    float5_e2m2,
    float4_e2m1,
    float3_e1m1,
    # signed integer dtypes
    int64,
    int32,
    int16,
    int8,
    boolean,
    # unsigned integer dtypes
    uint64,
    uint32,
    uint16,
    uint8,
    # sub-bype integer dtypes
    int7b,
    int6b,
    int5b,
    int4b,
    int3b,
    int2b,
    int1b,
    uint7b,
    uint6b,
    uint5b,
    uint4b,
    uint3b,
    uint2b,
    uint1b,
    # complex dtypes
    complex64,
    complex128,
    # float vector dtypes
    float32x1,
    float32x2,
    float32x4,
    float32x8,
    float16x1,
    float16x2,
    float16x4,
    float16x8,
    # signed integer vector dtypes
    int8x4,
    int4bx8,
    # unsigned integer vector dtypes
    uint8x1,
    uint8x2,
    uint8x4,
    uint16x1,
    uint16x2,
    uint16x4,
    uint32x1,
    uint32x2,
    uint32x4,
    uint64x1,
    uint64x2,
    uint64x4,
    uint4bx8,
)

# dtype name -> dtype
name2dtype: Dict[str, DataType] = {}

# dtype short name -> dtype
sname2dtype: Dict[str, DataType] = {}


@initialize()
def _initialize_name_mapping():
    for dtype in registered_dtypes:
        name2dtype[dtype.name] = dtype
        sname2dtype[dtype.short_name] = dtype


default_int_dtype = int32
default_index_dtype = int64
default_float_dtype = float32


def supported(name: str) -> bool:
    return name in name2dtype
