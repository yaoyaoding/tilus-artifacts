import pytest
import hidet
import numpy as np


@pytest.mark.skipif(not hidet.cuda.available(), reason='CUDA is not available')
def test_for_and_increment():
    from hidet.lang.types import f32, int32
    from hidet.lang import attrs
    from hidet.lang import tile as ti

    with hidet.script_module() as script_module:

        @hidet.script
        def use_arange(b_ptr: ~int32):
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 128

            a = ti.arange(0, 16)
            for k in range(10):
                a += 1
            ti.store(b_ptr + ti.arange(0, 16), value=a)

    func = script_module.build()

    b = hidet.empty([16], dtype=hidet.int32, device='cuda')
    func(b)

    b_expected = np.arange(16, dtype=np.int32) + 10
    hidet.utils.assert_close(actual=b, expected=b_expected)
