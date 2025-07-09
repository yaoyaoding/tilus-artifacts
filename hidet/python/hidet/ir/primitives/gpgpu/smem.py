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
from typing import Union, Optional

from hidet.ir.expr import Expr, Call, cast
from hidet.ir.type import DataType, PointerType, data_type, void_p
from hidet.ir.func import Function
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.utils import initialize


@initialize()
def register_functions():
    from hidet.lang import script, attrs

    for dtype in ['void', 'bool', 'int8', 'uint8', 'uint32', 'int32', 'float16', 'float32']:
        func_name = f'gpgpu_dynamic_shared_memory_{dtype}'
        if dtype == 'void':
            ret_type = void_p
        elif dtype == 'void_p':
            ret_type = ~void_p
        else:
            ret_type = ~data_type(dtype)

        @script
        def gpgpu_dynamic_shared_memory(byte_offset: int) -> ret_type:
            attrs.func_kind = 'gpgpu_internal'
            attrs.func_name = func_name
            dynamic_smem = PointerType(base_type='uint8', specifiers=['extern', '__shared__'], use_bracket=True)
            return cast(~dynamic_smem[byte_offset], ret_type)

        assert isinstance(gpgpu_dynamic_shared_memory, Function)
        register_primitive_function(gpgpu_dynamic_shared_memory.name, gpgpu_dynamic_shared_memory)


def dynamic_shared_memory(
    byte_offset: Union[Expr, int], dtype: Optional[Union[DataType, PointerType, str]] = None
) -> Call:
    if dtype is None:
        suffix = 'void'
    elif isinstance(dtype, PointerType):
        suffix = 'void_p'
    else:
        suffix: str = data_type(dtype).name
    func_name = f'gpgpu_dynamic_shared_memory_{suffix}'
    if dtype is None:
        return cast(call_primitive_func(func_name, [byte_offset]), void_p)
    elif isinstance(dtype, PointerType):
        return cast(call_primitive_func(func_name, [byte_offset]), ~dtype)
    else:
        return call_primitive_func(func_name, [byte_offset])
