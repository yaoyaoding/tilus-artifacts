# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# %%
import numpy as np
import pytest

import hidet
from hidet import ops
from hidet.testing import check_binary, check_binary_dynamic


@pytest.mark.skipif(not hidet.hip.available(), reason='HIP is not available')
@pytest.mark.parametrize(
    "a_shape, b_shape, dtype", [[[1, 333, 444], [1, 444, 555], "float32"], [[1, 333, 444], [1, 444, 555], "float16"]]
)
@pytest.mark.parametrize('mma', ['simt', 'mma'])
def test_batch_matmul(a_shape, b_shape, dtype, mma):
    if hidet.hip.capability().gcnArchName != 'gfx90a' and mma == 'mma':
        pytest.skip('mma is only supported on gfx90a')

    tolerance = {('float16', 'simt'): 0.5, ('float16', 'mma'): 0.5, ('float32', 'simt'): 1e-4, ('float32', 'mma'): 0.05}
    tol = tolerance[(dtype, mma)]
    check_binary(
        a_shape,
        b_shape,
        lambda x, y: np.matmul(x, y),
        lambda x, y: ops.cuda_batch_matmul(x, y, mma=mma),
        device='hip',
        dtype=dtype,
        atol=tol,
        rtol=tol,
    )


@pytest.mark.skipif(not hidet.hip.available(), reason='HIP is not available')
@pytest.mark.parametrize(
    "a_shape, b_shape, dtype",
    [
        [[1, ("n", 333), ("m", 444)], [1, ("m", 444), ("k", 555)], "float32"],
        [[("b", 1), ("m", 333), ("k", 444)], [("b", 1), ("k", 444), ("n", 555)], "float16"],
    ],
)
@pytest.mark.parametrize('mma', ['simt', 'mma'])
def test_batch_matmul_dynamic(a_shape, b_shape, dtype: str, mma: str):
    if hidet.hip.capability().gcnArchName != 'gfx90a' and mma == 'mma':
        pytest.skip('mma is only supported on gfx90a')
    tolerance = {('float16', 'simt'): 0.5, ('float16', 'mma'): 0.5, ('float32', 'simt'): 1e-4, ('float32', 'mma'): 0.05}
    tol = tolerance[(dtype, mma)]
    check_binary_dynamic(
        a_shape,
        b_shape,
        lambda x, y: np.matmul(x, y),
        lambda x, y: ops.cuda_batch_matmul(x, y, mma=mma),
        device='hip',
        dtype=dtype,
        atol=tol,
        rtol=tol,
    )


if __name__ == '__main__':
    pytest.main([__file__])
