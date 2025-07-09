import pytest
import pytest
import hidet


@pytest.mark.skipif(not hidet.cuda.available(), reason='CUDA is not available')
def test_simt_dot(m=16, n=16, k=16):
    from hidet.lang.types import f32
    from hidet.lang import attrs
    from hidet.lang import tile as ti

    with hidet.script_module() as script_module:

        @hidet.script
        def dot(a_ptr: ~f32, b_ptr: ~f32, c_ptr: ~f32):
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 128

            a_ptrs = a_ptr + ti.expand_dims(ti.arange(0, m) * k, axis=1) + ti.arange(0, k)
            b_ptrs = b_ptr + ti.expand_dims(ti.arange(0, k) * n, axis=1) + ti.arange(0, n)
            c_ptrs = c_ptr + ti.expand_dims(ti.arange(0, m) * n, axis=1) + ti.arange(0, n)

            a = ti.load(a_ptrs)
            b = ti.load(b_ptrs)
            c = ti.dot(a, b)
            ti.store(c_ptrs, c)

    a = hidet.randn([m, k], dtype='float32', device='cuda')
    b = hidet.randn([k, n], dtype='float32', device='cuda')
    c1 = hidet.empty([m, n], dtype='float32', device='cuda')
    func = script_module.build()
    func(a, b, c1)

    c2 = a @ b

    hidet.utils.assert_close(c1, c2)


@pytest.mark.skipif(not hidet.cuda.available(), reason='CUDA is not available')
@pytest.mark.parametrize(
    'm_size, n_size, k_size',
    [(16, 16, 16), (16, 16, 32), (16, 32, 16), (32, 16, 16), (32, 32, 32), (32, 32, 64), (32, 64, 32), (64, 32, 32)],
)
def test_mma_dot(m_size, n_size, k_size):
    from hidet.lang import attrs
    from hidet.lang.types import f16
    from hidet.lang import tile as ti

    m_size = 16
    n_size = 16
    k_size = 32

    with hidet.script_module() as script_module:

        @hidet.script
        def mma_kernel(a_ptr: ~f16, b_ptr: ~f16, c_ptr: ~f16):
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.block_dim = 32 * 2
            attrs.cuda.grid_dim = 1

            a_ptrs = a_ptr + ti.grid([m_size, k_size], starts=[0, 0], strides=[k_size, 1])
            b_ptrs = b_ptr + ti.grid([k_size, n_size], starts=[0, 0], strides=[n_size, 1])
            c_ptrs = c_ptr + ti.grid([m_size, n_size], starts=[0, 0], strides=[n_size, 1])

            a = ti.load(a_ptrs)
            b = ti.load(b_ptrs)
            c = ti.dot(a, b)
            ti.store(c_ptrs, c)

    func = script_module.build()

    a = hidet.randn([m_size, k_size], dtype=f16, device='cuda')
    b = hidet.randn([k_size, n_size], dtype=f16, device='cuda')
    c = hidet.empty([m_size, n_size], dtype=f16, device='cuda')

    func(a, b, c)

    hidet.utils.assert_close(actual=c, expected=hidet.from_torch(a.torch() @ b.torch()), atol=1e-2, rtol=1e-2)
