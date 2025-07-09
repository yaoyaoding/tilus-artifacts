import os
import os

import torch

import hidet.cuda
import mutis.option
from hidet.graph.frontend.torch.utils import dtype_to_torch
from hidet.ir.dtypes import float6_e3m2
from hidet.ir.dtypes import uint32x4, uint32x2, uint32, uint16, uint8
from hidet.ir.type import DataType
from hidet.runtime import CompiledModule
from hidet.runtime.compiled_module import CompiledModuleLoadError
from mutis.ir.layout import Layout, repeat, column_repeat, auto_layout_for_efficient_ldst
from mutis.vm.ir.builder import VirtualMachineBuilder
from mutis.vm.ir.shared_layout import shared_repeat


class Config:
    def __init__(
        self,
        k: int,
        n: int,
        dtype: DataType,
        layout: Layout,
        transformed_dtype: DataType,
        transformed_layout: Layout,
        output_dtype: DataType,
    ):
        self.k: int = k
        self.n: int = n
        self.dtype: DataType = dtype
        self.layout: Layout = layout
        self.transformed_dtype: DataType = transformed_dtype
        self.transformed_layout: Layout = transformed_layout
        self.output_dtype: DataType = output_dtype

        block_k, block_n = layout.shape
        assert k % block_k == 0 and n % block_n == 0
        assert self.layout.num_workers == self.transformed_layout.num_workers
        assert (
            self.layout.local_size * self.dtype.nbits
            == self.transformed_layout.local_size * self.transformed_dtype.nbits
        )

    def build(self) -> CompiledModule:
        vm = VirtualMachineBuilder()

        with vm.program(
            'matmul_weight_decode', num_warps=1, params={'y_ptr': ~self.output_dtype, 'x_ptr': ~self.dtype}
        ) as (y_ptr, x_ptr):
            block_k, block_n = self.layout.shape
            k_blocks, n_blocks = self.k // block_k, self.n // block_n
            bk, bj = vm.virtual_blocks([k_blocks, n_blocks])
            block_id = bk * n_blocks + bj
            transformed_elements_per_block = block_k * block_n * self.dtype.nbits // self.transformed_dtype.nbits
            x = vm.load_global(
                dtype=self.transformed_dtype,
                layout=self.transformed_layout,
                ptr=x_ptr,
                f_offset=lambda axes: block_id * transformed_elements_per_block + axes[0],
            )
            x = vm.view(x, layout=self.layout, dtype=self.dtype)
            x = vm.cast(x, dtype=self.output_dtype)
            smem = vm.allocate_shared(dtype=self.output_dtype, shared_layout=shared_repeat(block_k, block_n))
            vm.store_shared(dst=smem, src=x)
            vm.syncthreads()
            x = vm.load_shared(
                src=smem,
                register_layout=auto_layout_for_efficient_ldst(
                    num_threads=32, shape=[block_k, block_n], dtype_nbits=self.output_dtype.nbits
                ),
            )
            vm.store_global(
                x=x, ptr=y_ptr, f_offset=lambda axes: (bk * block_k + axes[0]) * self.n + bj * block_n + axes[1]
            )
            vm.free_shared(smem)

        program = vm.built_program
        return program.build()


_in_memory_cache = {}


def make_kernel(k: int, n: int, dtype: DataType, output_dtype: DataType) -> CompiledModule:
    task_name = '{}-{}-{}-{}'.format(k, n, dtype.name, output_dtype.name)

    # try to use memory cache
    if task_name in _in_memory_cache:
        return _in_memory_cache[task_name]

    # # try to load from disk cache
    # module_dir = os.path.join(mutis.option.get_option('cache_dir'), 'decode', task_name)
    # try:
    #     print(module_dir)
    #     compiled_module = hidet.runtime.load_compiled_module(module_dir)
    #     _in_memory_cache[task_name] = compiled_module
    #     return compiled_module
    # except CompiledModuleLoadError:
    #     pass

    # build
    layout = repeat(1, 4).column_repeat(8, 1).column_spatial(4, 8).repeat(2, 1)
    total_bits = layout.local_size * dtype.nbits

    def get_transformed_dtype():
        dtype_list = [uint32x4, uint32x2, uint32, uint16, uint8]
        for t_dtype in dtype_list:
            if total_bits % t_dtype.nbits == 0:
                return t_dtype
        assert False

    transformed_dtype = get_transformed_dtype()
    transformed_layout = repeat(total_bits // transformed_dtype.nbits).spatial(32)
    config = Config(
        k=k,
        n=n,
        dtype=dtype,
        layout=layout,
        transformed_dtype=transformed_dtype,
        transformed_layout=transformed_layout,
        output_dtype=output_dtype,
    )
    compiled_module = config.build()
    _in_memory_cache[task_name] = compiled_module
    return compiled_module


def matmul_mma_decode(k: int, n: int, dtype: DataType, output_dtype: DataType, x: torch.Tensor):
    out = torch.empty(k, n, dtype=dtype_to_torch(output_dtype), device='cuda')
    kernel = make_kernel(k=k, n=n, dtype=dtype, output_dtype=output_dtype)
    kernel(out, x)
    return out


def main():
    import mutis
    from mutis.utils import benchmark_func
    from hidet.ir.dtypes import uint4b, float16

    k, n = (8192, 57344)
    for dtype in [uint8, float6_e3m2, uint4b]:
        print(k, n, dtype)
        x = mutis.randn([k, n], dtype=dtype).storage
        func = lambda: matmul_mma_decode(k=k, n=n, dtype=dtype, output_dtype=float16, x=x)
        print(benchmark_func(func))


if __name__ == '__main__':
    main()
