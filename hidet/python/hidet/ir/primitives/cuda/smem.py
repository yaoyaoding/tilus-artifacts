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
from hidet.ir.expr import Var
from hidet.ir.stmt import BlackBoxStmt, Stmt
from hidet.ir.primitives.gpgpu.smem import dynamic_shared_memory


def set_kernel_max_dynamic_smem_bytes(func: Var, max_dynamic_smem_bytes: Union[Expr, int]) -> Stmt:
    from hidet.ir.expr import convert

    max_dynamic_smem_bytes = convert(max_dynamic_smem_bytes)
    template_string = r'cudaFuncSetAttribute({}, cudaFuncAttributeMaxDynamicSharedMemorySize, {});'
    return BlackBoxStmt(template_string, func, max_dynamic_smem_bytes)
