import pytest
import hidet


@pytest.mark.skipif(not hidet.cuda.available(), reason='CUDA is not available')
def test_vector_add(n: int = 1024):
    from hidet.lang.types import f32
    from hidet.lang import attrs
    from hidet.lang import tile as ti

    block_size = 128

    with hidet.script_module() as script_module:

        @hidet.script
        def vec_add(a: ~f32, b: ~f32, c: ~f32):
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.block_dim = 256
            attrs.cuda.grid_dim = (n + block_size - 1) // block_size

            pid = ti.program_id()
            offsets = pid * block_size + ti.arange(0, block_size)
            mask = offsets < n

            result = ti.load(a + offsets, mask=mask) + ti.load(b + offsets, mask=mask)
            ti.store(c + offsets, result, mask=mask)

    a = hidet.randn([n], dtype=hidet.float32, device='cuda')
    b = hidet.randn([n], dtype=hidet.float32, device='cuda')
    c = hidet.zeros([n], dtype=hidet.float32, device='cuda')
    func = script_module.build()
    func(a, b, c)
    hidet.utils.assert_close(c, a + b)
