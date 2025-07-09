import pytest
import hidet
from hidet.ir.dtypes import f16, f32


@pytest.mark.skipif(not hidet.cuda.available(), reason='CUDA is not available')
@pytest.mark.parametrize('dtype', [f16, f32])
def demo_reduce_add(dtype):
    from hidet.lang import attrs
    from hidet.ir.primitives.cuda.atomic import reduce_add

    with hidet.script_module() as script_module:

        @hidet.script
        def func_v1(a: ~dtype):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.block_dim = 1
            attrs.cuda.grid_dim = 1874

            reduce_add(dtype, addr=a, src_values=[dtype.one])

    op = script_module.build()
    a = hidet.zeros([1], dtype=dtype, device='cuda')
    op(a)
    assert a.item() == 1874.0
