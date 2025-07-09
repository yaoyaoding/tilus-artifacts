import pytest
import hidet
import numpy as np


@pytest.mark.skipif(not hidet.cuda.available(), reason='CUDA is not available')
def test_expand_dims():
    from hidet.lang.types import f32, int32
    from hidet.lang import attrs
    from hidet.lang import tile as ti

    with hidet.script_module() as script_module:

        @hidet.script
        def use_expand_dims(c_ptr: ~f32):
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 128

            a = ti.arange(0, 16)
            b = ti.arange(0, 16)
            c = ti.expand_dims(a, 1) * 16 + ti.expand_dims(b, 0)
            c_ptrs = c_ptr + c
            ti.store(c_ptrs, ti.cast(c, f32))

    c = hidet.empty([16, 16], dtype=hidet.float32, device='cuda')
    func = script_module.build()
    func(c)

    c_expected = np.arange(16 * 16, dtype=np.float32).reshape([16, 16])
    hidet.utils.assert_close(actual=c, expected=c_expected)
