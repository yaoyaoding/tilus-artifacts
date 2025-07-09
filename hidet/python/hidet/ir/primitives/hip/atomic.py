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
from typing import Union, List

from hidet.ir import PointerType
from hidet.ir.dtypes import f16, f16x2, bf16, f32, u8, u16, u32, u64, i32, i64
from hidet.ir.type import DataType
from hidet.ir.expr import Expr, cast, deref
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.ir.type import FuncType, data_type
from hidet.utils import initialize


@initialize()
def register_functions():
    register_primitive_function(
        'hip_atomic_cas_int32', func_or_type=FuncType([~i32, i32, i32], i32), codegen_name='atomicCAS'
    )
    register_primitive_function(
        'hip_atomic_cas_uint32', func_or_type=FuncType([~u32, u32, u32], u32), codegen_name='atomicCAS'
    )


def atomic_cas(addr: Expr, compare: Union[Expr, int], value: Union[Expr, int]):
    from hidet.ir.tools import infer_type

    addr_type = infer_type(addr)

    assert isinstance(addr_type, PointerType)
    dtype = addr_type.base_type
    assert isinstance(dtype, DataType)
    if dtype not in [i32, u32]:
        raise ValueError(f"Unsupported atomic type {addr_type.base_type} for atomic_cas")

    return call_primitive_func('hip_atomic_cas_{}'.format(dtype.name), [addr, compare, value])
