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
from typing import Union, List

from hidet.graph import ops
from hidet.graph.nn.module import Module
from hidet.graph.tensor import Tensor, empty


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.running_mean = empty(shape=[num_features])
        self.running_var = empty(shape=[num_features])
        self.num_batches_tracked = empty(shape=[])
        if affine:
            self.weight: Tensor = empty(shape=[num_features])
            self.bias = empty(shape=[num_features])
        else:
            self.weight = None
            self.bias = None

    def extra_str(self) -> str:
        return 'eps={}, affine={}'.format(self.eps, self.affine)

    def forward(self, x: Tensor):
        x = ops.batch_norm_infer(x, self.running_mean, self.running_var, self.eps)
        if self.affine:
            x = x * self.weight.unsqueeze([0, 2, 3]) + self.bias.unsqueeze([0, 2, 3])
        return x


class LayerNorm(Module):
    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = empty(normalized_shape)
            self.bias = empty(normalized_shape)
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        x = ops.layer_norm(x)
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.num_groups = num_groups
        self.num_channels = num_channels
        if affine:
            # add extra dims for broadcast
            self.weight: Tensor = empty(shape=[num_channels, 1, 1])
            self.bias: Tensor = empty(shape=[num_channels, 1, 1])
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor):
        x = ops.group_norm(x, self.num_groups, self.eps)
        if self.affine:
            x = x * self.weight + self.bias

        return x
