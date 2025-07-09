import pytest
import hidet


@pytest.mark.skipif(not hidet.cuda.available(), reason='CUDA is not available')
def test_arange():
    from hidet.lang.types import f32, int32
    from hidet.lang import attrs
    from hidet.lang import tile as ti

    with hidet.script_module() as script_module:

        @hidet.script
        def use_arange(a_ptr: ~f32, b_ptr: ~f32, n: int32):
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 128

            a = ti.arange(0, 16)
            mask = a < n
            a_ptrs = a_ptr + a
            b_ptrs = b_ptr + a
            ti.store(b_ptrs, ti.load(a_ptrs, mask) + 1, mask)

    func = script_module.build()
    n = 16
    a = hidet.randn([16], dtype=hidet.float32, device='cuda')
    b = hidet.empty([16], dtype=hidet.float32, device='cuda')

    func(a, b, n)
    hidet.utils.assert_close(b, a + 1)
