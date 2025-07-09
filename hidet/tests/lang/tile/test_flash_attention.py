import pytest
import torch
import hidet


def flash_attention(seq_heads=128 * 32, dim=128, q_tile=32, v_tile=128, num_warps=4):
    from hidet.lang import attrs
    from hidet.lang.types import f16, f32
    from hidet.lang import tile as ti

    assert seq_heads % q_tile == 0

    with hidet.script_module() as script_module:

        @hidet.script
        def flash_attention_kernel(
            q_ptr: ~f16, k_ptr: ~f16, v_ptr: ~f16, o_ptr: ~f16  # [seq, dim]  # [dim, seq]  # [seq, dim]  # [seq, dim]
        ):
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.block_dim = num_warps * 32
            attrs.cuda.grid_dim = seq_heads // q_tile

            pid = ti.program_id()

            q_ptrs = q_ptr + ti.grid(shape=[q_tile, dim], starts=[pid * q_tile, 0], strides=[dim, 1])
            k_ptrs = k_ptr + ti.grid(shape=[dim, v_tile], starts=[0, 0], strides=[seq_heads, 1])
            v_ptrs = v_ptr + ti.grid(shape=[v_tile, dim], starts=[0, 0], strides=[dim, 1])

            q = ti.load(q_ptrs)
            o = ti.zeros([q_tile, dim], dtype=f16)

            m = ti.full(shape=[q_tile, 1], value=-1e4, dtype=f32)
            l = ti.zeros(shape=[q_tile, 1], dtype=f32)

            for i in range(seq_heads // v_tile):
                # q @ k
                k = ti.load(k_ptrs)

                qk = ti.dot(q, k)  # [q_tile, v_tile]

                # online softmax(q @ k)
                m2 = ti.maximum(m, ti.max(qk, axis=1, keepdims=True))  # [q_tile, 1]
                p = ti.exp(qk - m2)
                scale = ti.exp(m - m2)
                l2 = l * scale + ti.sum(p, axis=1, keepdims=True)  # [q_tile, 1]
                score = ti.cast(p / l2, dtype=f16)
                o = o * ti.cast(l / l2 * scale, dtype=f16)  # [q_tile, dim]

                # update m and l
                m = m2
                l = l2

                # update o
                v = ti.load(v_ptrs)  # [v_tile, dim]
                o += ti.dot(score, v)  # [q_tile, dim]

                # advance pointers
                k_ptrs += v_tile
                v_ptrs += v_tile * dim

            o_ptrs = o_ptr + ti.grid(shape=[q_tile, dim], starts=[pid * q_tile, 0], strides=[dim, 1])
            ti.store(o_ptrs, o)

    return script_module.build()


def flash_attention_torch(seq, dim):
    import torch

    def func(q, k, v, o):
        q, k, v, o = [t if isinstance(t, torch.Tensor) else t.torch() for t in [q, k, v, o]]
        o.copy_(torch.softmax(q @ k, dim=-1) @ v)

    return func


@pytest.mark.skipif(not hidet.cuda.available(), reason='CUDA is not available')
@pytest.mark.parametrize('seq_heads', [128 * 32])
@pytest.mark.parametrize('dim', [128])
@pytest.mark.parametrize('q_tile', [32])
@pytest.mark.parametrize('v_tile', [32])
def test_flash_attention(seq_heads, dim, q_tile, v_tile):
    tilia_kernel = flash_attention(seq_heads=seq_heads, dim=dim, q_tile=q_tile, v_tile=v_tile, num_warps=4)
    torch_kernel = flash_attention_torch(seq_heads, dim)

    q = hidet.randn([seq_heads, dim], dtype='float16', device='cuda') / 10.0
    k = hidet.randn([dim, seq_heads], dtype='float16', device='cuda') / 10.0
    v = hidet.randn([seq_heads, dim], dtype='float16', device='cuda') / 10.0

    o1 = hidet.zeros([seq_heads, dim], dtype='float16', device='cuda')
    o2 = hidet.zeros([seq_heads, dim], dtype='float16', device='cuda')

    tilia_kernel(q, k, v, o1)
    hidet.cuda.synchronize()

    torch_kernel(q, k, v, o2)
    hidet.cuda.synchronize()

    torch.testing.assert_close(actual=o2.torch(), expected=o1.torch(), atol=2e-2, rtol=2e-2)
