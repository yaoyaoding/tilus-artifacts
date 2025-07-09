import pytest
import hidet


@pytest.mark.skipif(not hidet.cuda.available(), reason='CUDA is not available')
def test_ldst():
    from hidet.lang.types import f32, int32
    from hidet.lang import attrs
    from hidet.lang import tile as ti

    block_m = 128
    block_k = 16

    with hidet.script_module() as script_module:

        @hidet.script
        def matmul(a_ptr: ~f32, b_ptr: ~f32):
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = 1

            offsets = ti.expand_dims(ti.arange(0, block_m), axis=1) * block_k + ti.arange(0, block_k)
            a_ptrs = a_ptr + offsets
            b_ptrs = b_ptr + offsets
            ti.store(b_ptrs, ti.load(a_ptrs))

    a = hidet.randn([block_m, block_k], device='cuda')
    b = hidet.empty([block_m, block_k], device='cuda')
    func = script_module.build()
    func(a, b)
    hidet.utils.assert_close(a, b)


@pytest.mark.skipif(not hidet.cuda.available(), reason='CUDA is not available')
def test_ldgsts_lds128():
    from hidet.lang.types import f32, int32
    from hidet.lang import attrs, cast, shared_tensor, register_tensor
    from hidet.lang.cuda import threadIdx, cp_async, cp_async_wait_all

    with hidet.script_module() as script_module:

        @hidet.script
        def func(out: f32[256]):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.block_dim = 32
            attrs.cuda.grid_dim = 1

            s = 0.0
            tid = threadIdx.x
            a = shared_tensor('float32', shape=[256])

            # ldgsts
            # L1 exc, L1, L1 ideal, L2 exc, L2, L2 ideal
            # 0	4	4	0	16	16
            cp_async(dst=~a[tid * 4], src=~out[tid * 4], cp_size=16)
            cp_async_wait_all()

            # 4	8	4	16	32	16
            cp_async(dst=~a[((3 - (tid // 8)) * 8 + tid % 8) * 4], src=~out[tid * 8], cp_size=16)
            cp_async_wait_all()

            # 4	8	4	16	32	16
            cp_async(dst=~a[((3 - (tid // 8)) * 8 + tid % 8) * 8], src=~out[tid * 8], cp_size=16)
            cp_async_wait_all()

            # lds
            b = register_tensor('float32', shape=[4])
            c = register_tensor('float32', shape=[4])
            for i in range(4):
                b[i] = a[tid * 4 + i]
            for i in range(4):
                group_id = tid // 8
                id_in_group = tid % 8
                c[i] = a[((3 - group_id) * 8 + id_in_group) * 4 + i]
            for i in range(4):
                s += b[i]
            for i in range(4):
                s += c[i]

            for i in range(4):
                out[tid] = s

    func = script_module.build()
    out = hidet.empty([256], device='cuda')
    func(out)
