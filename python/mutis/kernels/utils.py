import functools
import torch
import hidet


@functools.cache
def _unpack_int8_to_int4():
    from hidet.lang import attrs, printf
    from hidet.lang.types import void_p, int8, int4b
    from hidet.lang.cuda import threadIdx, blockIdx, blockDim

    with hidet.script_module() as script_module:

        @hidet.script
        def unpack_int8_to_int4_kernel(src_count: int, dst: ~int8, src: ~int8):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = (src_count + 127) // 128

            i = blockIdx.x * blockDim.x + threadIdx.x

            if i < src_count:
                dst[i << 1] = int8(src[i] << 4) >> 4  # signed shift extension
                dst[i << 1 | 1] = src[i] >> 4

    func = script_module.build()
    return func


@functools.cache
def _pack_int4_to_int8():
    from hidet.lang import attrs, printf
    from hidet.lang.types import void_p, int8, int4b
    from hidet.lang.cuda import threadIdx, blockIdx, blockDim

    with hidet.script_module() as script_module:

        @hidet.script
        def pack_int4_to_int8_kernel(dst_count: int, dst: ~int8, src: ~int8):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = (dst_count + 127) // 128

            i = blockIdx.x * blockDim.x + threadIdx.x

            if i < dst_count:
                dst[i] = (src[i << 1 | 1] << 4) | (src[i << 1] & 0xFF)

    func = script_module.build()
    return func


def unpack_int8_to_int4(w: torch.Tensor):
    kernel = _unpack_int8_to_int4()
    assert w.dtype == torch.int8, 'Input tensor must be int8'
    dst_shape = list(w.shape)
    dst_shape[-1] *= 2
    dst = torch.empty(dst_shape, dtype=torch.int8, device=w.device)
    kernel(w.numel(), dst, w)
    return dst


def pack_int4_to_int8(w: torch.Tensor):
    kernel = _pack_int4_to_int8()
    assert w.dtype == torch.int8, 'Input tensor must be int8'
    assert torch.all(torch.logical_and(w >= -8, w <= 7)), 'Input tensor must be in range of int4: [-8, 7]'
    dst_shape = list(w.shape)
    assert dst_shape[-1] % 2 == 0, 'Last dimension of input tensor must be even'
    dst_shape[-1] //= 2
    dst = torch.empty(dst_shape, dtype=torch.int8, device=w.device)
    kernel(dst.numel(), dst, w)
    return dst


def demo_unpack_int8_to_int4():
    w = torch.randint(-128, 127, size=(233,), dtype=torch.int8, device='cuda')
    dst = unpack_int8_to_int4(w)
    assert torch.allclose(dst[::2], w << 4 >> 4)
    assert torch.allclose(dst[1::2], w >> 4)


def demo_pack_int4_to_int8():
    w = torch.randint(-8, 7, size=(232,), dtype=torch.int8, device='cuda')
    dst = pack_int4_to_int8(w)
    assert torch.allclose(dst, (w[1::2] << 4) | (w[::2] & 0xFF))


if __name__ == '__main__':
    hidet.option.cache_dir('./outs')
    demo_unpack_int8_to_int4()
    demo_pack_int4_to_int8()
