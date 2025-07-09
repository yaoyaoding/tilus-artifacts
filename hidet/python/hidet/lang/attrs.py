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
# pylint: disable=pointless-string-statement
from typing import Optional, Union, Tuple
from hidet.ir.expr import Expr


"""
general attributes
"""
# The label of the scope
label: Optional[str] = None

"""
function attributes
"""
# The name of the function. The default hidet function name is the name of wrapped python function.
# Please set this attribute if we want to have a different name
func_name: Optional[str] = None

# The kind of this function. Candidates: 'cuda_kernel', 'cuda_internal', 'cpu_kernel', 'packed_func'
func_kind: Optional[str] = None


# If the func_kind == packed_func, then this attribute should be set to the var to function to be packed.
# packed_func: Optional[Var] = None


class cuda:
    Int = Union[Expr, int]
    Dim3 = Union[Int, Tuple[Int, Int], Tuple[Int, Int, Int]]

    # The grid dimension of a cuda kernel, specifying the number of thread blocks
    grid_dim: Dim3 = 1

    # The block dimension of a cuda kernel, specifying the number of threads per block
    block_dim: Dim3 = 1

    # A hint to nvcc compiler the minimal number of thread blocks should be executed on
    # the same streaming processor (SM). This attribute will influence the register allocation
    # strategy adopted by nvcc.
    min_blocks: int = 1

    # The size of dynamic shared memory allocated to the cuda kernel.
    dynamic_smem_bytes: Int = 0

    class opt:
        pipe_stages: int = 3


class hip:
    Int = Union[Expr, int]
    Dim3 = Union[Int, Tuple[Int, Int], Tuple[Int, Int, Int]]

    # The grid dimension of a cuda kernel, specifying the number of thread blocks
    grid_dim: Dim3 = 1

    # The block dimension of a cuda kernel, specifying the number of threads per block
    block_dim: Dim3 = 1


class gpgpu:
    Int = Union[Expr, int]
    Dim3 = Union[Int, Tuple[Int, Int], Tuple[Int, Int, Int]]

    # The grid dimension of a cuda kernel, specifying the number of thread blocks
    grid_dim: Dim3 = 1

    # The block dimension of a cuda kernel, specifying the number of threads per block
    block_dim: Dim3 = 1
