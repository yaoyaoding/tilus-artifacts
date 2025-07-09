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
"""
Propagate the launch configuration cuda.block_dim and cuda.grid_dim from cuda_kernel functions
to cuda_internal functions.
"""
from typing import Optional, Union, Sequence

from hidet.ir.expr import Constant, Expr, convert
from hidet.ir.func import Function
from hidet.ir.module import IRModule
from hidet.transforms.base import Pass, FunctionPass


class PropagateLaunchBoundPass(FunctionPass):
    # currently, we only deal with the module that has a single cuda kernel
    # and leave the support for multiple cuda kernels in the future.
    # please note that propagate the launch bound would help optimize the model,
    # but has no influence on the correctness.
    def __init__(self):
        super().__init__()
        self.kernel_function: Optional[Function] = None

    def process_module(self, ir_module: IRModule) -> IRModule:
        kernel_function = None
        for func in ir_module.functions.values():
            if func.kind == 'cuda_kernel':
                if kernel_function is not None:
                    return ir_module
                kernel_function = func
        if kernel_function is None:
            return ir_module
        self.kernel_function = kernel_function
        return FunctionPass.process_module(self, ir_module)

    def normalize_dims(self, dims: Union[int, Expr, Sequence[Union[int, Expr]]]):
        if not isinstance(dims, Sequence):
            dims = [dims]
        assert 1 <= len(dims) <= 3
        return tuple([convert(dim) for dim in dims])

    def process_func(self, func: Function) -> Function:
        if func.kind == 'cuda_internal':
            attrs = func.attrs.copy()
            block_dims = self.normalize_dims(self.kernel_function.attrs['cuda.block_dim'])
            grid_dims = self.kernel_function.attrs['cuda.grid_dim']
            if all(isinstance(v, Constant) for v in block_dims):
                attrs['cuda.block_dim'] = block_dims
            if all(isinstance(v, Constant) for v in grid_dims):
                attrs['cuda.grid_dim'] = grid_dims
            func = Function(
                name=func.name, params=func.params, body=func.body, ret_type=func.ret_type, kind=func.kind, attrs=attrs
            )
        return func


def propagate_launch_bound_pass() -> Pass:
    return PropagateLaunchBoundPass()
