import pytest
import hidet
import numpy as np


@pytest.mark.skipif(not hidet.cuda.available(), reason='CUDA is not available')
@pytest.mark.parametrize('m, n', [(32, 16), (16, 32), (128, 128), (512, 512)])
def test_reduce(m, n):
    from hidet.lang.types import int32
    from hidet.lang import attrs
    from hidet.lang import tile as ti

    with hidet.script_module() as script_module:

        @hidet.script
        def use_arange(sum0: ~int32, sum1: ~int32, min0: ~int32, min1: ~int32, max0: ~int32, max1: ~int32):
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 128

            a = ti.arange(0, m)
            b = ti.arange(0, n)
            c = ti.expand_dims(a, 1) * n + ti.expand_dims(b, 0)

            ti.store(sum0 + ti.expand_dims(ti.arange(0, n), axis=0), ti.sum(c, axis=0, keepdims=True))
            ti.store(sum1 + ti.expand_dims(ti.arange(0, m), axis=1), ti.sum(c, axis=1, keepdims=True))
            ti.store(min0 + ti.expand_dims(ti.arange(0, n), axis=0), ti.min(c, axis=0, keepdims=True))
            ti.store(min1 + ti.expand_dims(ti.arange(0, m), axis=1), ti.min(c, axis=1, keepdims=True))
            ti.store(max0 + ti.arange(0, n), ti.max(c, axis=0, keepdims=False))
            ti.store(max1 + ti.arange(0, m), ti.max(c, axis=1, keepdims=False))

    func = script_module.build()
    sum0 = hidet.empty([1, n], dtype=hidet.int32, device='cuda')
    sum1 = hidet.empty([m, 1], dtype=hidet.int32, device='cuda')
    min0 = hidet.empty([1, n], dtype=hidet.int32, device='cuda')
    min1 = hidet.empty([m, 1], dtype=hidet.int32, device='cuda')
    max0 = hidet.empty([n], dtype=hidet.int32, device='cuda')
    max1 = hidet.empty([m], dtype=hidet.int32, device='cuda')
    func(sum0, sum1, min0, min1, max0, max1)

    x = np.arange(0, m * n).reshape([m, n])
    actual_tensors = [sum0, sum1, min0, min1, max0, max1]
    expected_tensors = [
        x.sum(axis=0, keepdims=True),
        x.sum(axis=1, keepdims=True),
        x.min(axis=0, keepdims=True),
        x.min(axis=1, keepdims=True),
        x.max(axis=0, keepdims=False),
        x.max(axis=1, keepdims=False),
    ]
    for actual, expected in zip(actual_tensors, expected_tensors):
        hidet.utils.assert_close(actual=actual, expected=expected)
