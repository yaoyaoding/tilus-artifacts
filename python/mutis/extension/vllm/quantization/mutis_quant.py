from typing import List, Dict, Any, Optional, Union
import functools

import torch
from torch import nn
from torch.nn import Module
from torch.nn.parameter import Parameter
from xformers.ops.fmha.triton_splitk import num_groups

import mutis
import vllm
from hidet.utils import error_tolerance
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    QKVParallelLinear,
    RowParallelLinear,
    MergedColumnParallelLinear,
)
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod, QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs

from hidet.ir.type import DataType, data_type
from mutis.kernels.scaled_mm_generic import scaled_mm_generic, scaled_mm_generic_reference
from mutis.kernels.vm.matmul_mma import matmul_mma
from mutis.utils import idiv


class MutisConfig(QuantizationConfig):
    backend: str

    def __init__(
        self,
        weight_granularity: str,
        group_size: int,
        weight_dtype: DataType,
        scale_dtype: DataType,
        keep_original_weight: bool,
    ):
        self.weight_granularity: str = weight_granularity
        self.group_size: int = group_size
        self.weight_dtype: DataType = weight_dtype
        # self.scale_dtype: DataType = scale_dtype
        self.scale_dtype: DataType = mutis.float16
        self.keep_original_weight: bool = keep_original_weight

    def get_name(self) -> str:
        return 'mutis'

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        return user_quant

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> QuantizationConfig:
        return cls(
            weight_granularity=config['weight_granularity'],
            group_size=config['group_size'],
            weight_dtype=data_type(config['weight_dtype']),
            scale_dtype=data_type(config['scale_dtype']),
            keep_original_weight=config.get('keep_original_weight', False),
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            return MutisLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        raise NotImplementedError()


class MutisQuantConfig(MutisConfig):
    backend = 'mutis'


class MutisTritonQuantConfig(MutisConfig):
    backend = 'triton'


class MutisBitblasQuantConfig(MutisConfig):
    backend = 'bitblas'


class MutisLinearMethod(LinearMethodBase):
    def __init__(self, config: MutisConfig):
        super().__init__()
        self.config: MutisConfig = config
        self.backend: str = self.config.backend

    def tensor_quantize_scale_loader(
        self, layer, param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id: Optional[Union[int, str]] = None
    ):
        assert param.data.shape == loaded_weight.shape, (param.data.shape, loaded_weight.shape)

        is_loaded = getattr(param, 'is_loaded', False)

        if is_loaded:
            # make sure the scale is the same
            torch.testing.assert_close(actual=param.data.cuda(), expected=loaded_weight.cuda(), atol=0, rtol=0)

        param.data.copy_(loaded_weight)

        setattr(param, 'is_loaded', True)

    def channel_quantize_scale_loader(
        self, layer, param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id: Optional[Union[int, str]] = None
    ):

        if loaded_shard_id is not None:
            if isinstance(loaded_shard_id, str):
                loaded_shard_id = {'q': 0, 'k': 1, 'v': 2}[loaded_shard_id]
            start_idx = sum(layer.output_sizes[:loaded_shard_id])
            length = layer.output_sizes[loaded_shard_id]
            param.data.narrow(dim=1, start=start_idx, length=length).copy_(loaded_weight)
        else:
            param.data.copy_(loaded_weight)

    def weight_loader_for_qkv_parallel_linear(
        self, layer: QKVParallelLinear, param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):
        tp_rank = get_tensor_model_parallel_rank()
        assert tp_rank == 0, 'TODO: support multi-rank loading'

        loaded_shard_id = {'q': 0, 'k': 1, 'v': 2}[loaded_shard_id]

        output_size, input_size = layer.output_size, layer.input_size
        dtype = getattr(param, 'actual_dtype')
        mutis_param = mutis.from_torch(param.data).view(dtype=dtype, shape=[input_size, output_size])
        loaded_weight = mutis.from_torch(loaded_weight.cuda()).view(
            dtype=dtype, shape=[input_size, layer.output_sizes[loaded_shard_id]]
        )
        output_begin = sum(layer.output_sizes[:loaded_shard_id])
        output_end = output_begin + layer.output_sizes[loaded_shard_id]
        mutis_param[:, output_begin:output_end] = loaded_weight

    def weight_loader_for_row_parallel_linear(
        self, layer: RowParallelLinear, param: Parameter, loaded_weight: torch.Tensor
    ):
        tp_rank = get_tensor_model_parallel_rank()
        assert tp_rank == 0, 'TODO: support multi-rank loading'

        assert param.data.shape == loaded_weight.shape, (param.data.shape, loaded_weight.shape)
        param.data.copy_(loaded_weight)

    def weight_loader_for_merged_column_parallel_linear(
        self, layer: MergedColumnParallelLinear, param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int
    ):
        tp_rank = get_tensor_model_parallel_rank()
        assert tp_rank == 0, 'TODO: support multi-rank loading'
        output_size, input_size = layer.output_size, layer.input_size
        dtype = getattr(param, 'actual_dtype')
        param = mutis.from_torch(param.data).view(dtype=dtype, shape=[input_size, output_size])
        loaded_weight = mutis.from_torch(loaded_weight.cuda()).view(
            dtype=dtype, shape=[input_size, layer.output_sizes[loaded_shard_id]]
        )
        output_begin = sum(layer.output_sizes[:loaded_shard_id])
        output_end = output_begin + layer.output_sizes[loaded_shard_id]
        param[:, output_begin:output_end] = loaded_weight

    def get_weight_loader(self, layer):
        if isinstance(layer, QKVParallelLinear):
            return functools.partial(self.weight_loader_for_qkv_parallel_linear, layer)
        elif isinstance(layer, RowParallelLinear):
            return functools.partial(self.weight_loader_for_row_parallel_linear, layer)
        elif isinstance(layer, MergedColumnParallelLinear):
            return functools.partial(self.weight_loader_for_merged_column_parallel_linear, layer)
        else:
            raise NotImplementedError()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs
    ):
        # create quantized weight with uint8 as data type
        weight_nbytes = (input_size * output_size * self.config.weight_dtype.nbits + 7) // 8
        weight_quant = Parameter(torch.empty(weight_nbytes, dtype=torch.uint8, device='cuda'), requires_grad=False)
        layer.register_parameter('weight_quant', weight_quant)
        set_weight_attrs(
            weight_quant,
            {
                'weight_loader': self.get_weight_loader(layer),
                'actual_dtype': self.config.weight_dtype,
                'param_name': extra_weight_attrs['prefix'] + '.weight_quant',
            },
        )

        if self.config.group_size == -1:
            num_groups = 1
        else:
            num_groups = idiv(input_size, self.config.group_size)

        torch_scale_dtype = mutis.dtype_to_torch(self.config.scale_dtype)

        if self.config.weight_dtype.is_integer() and not self.config.weight_dtype.signedness():
            # unsigned integer quantization, need to load a weight bias
            weight_bias = Parameter(
                torch.empty([num_groups, output_size], dtype=torch_scale_dtype, device='cuda'), requires_grad=False
            )
            layer.register_parameter('weight_bias', weight_bias)
            set_weight_attrs(
                weight_bias,
                {
                    'weight_loader': functools.partial(self.channel_quantize_scale_loader, layer),
                    'param_name': extra_weight_attrs['prefix'] + '.weight_bias',
                },
            )
        else:
            layer.register_parameter('weight_bias', None)

        if self.config.keep_original_weight:
            weight = Parameter(
                torch.empty([input_size, output_size], dtype=torch.bfloat16, device='cpu'), requires_grad=False
            )
            layer.register_parameter('weight', weight)
            set_weight_attrs(
                weight,
                {
                    'weight_loader': self.get_weight_loader(layer),
                    'actual_dtype': mutis.dtype_from_torch(torch.bfloat16),
                    'param_name': extra_weight_attrs['prefix'] + '.weight',
                },
            )

        # create weight scale with float32 as data type
        if self.config.weight_granularity == 'tensor':
            weight_scale = Parameter(torch.empty([1], dtype=torch_scale_dtype), requires_grad=False)
            layer.register_parameter('weight_scale', weight_scale)
            set_weight_attrs(
                weight_scale,
                {
                    'weight_loader': functools.partial(self.tensor_quantize_scale_loader, layer),
                    'param_name': extra_weight_attrs['prefix'] + '.weight_scale',
                },
            )
        else:
            weight_scale = Parameter(
                torch.empty([num_groups, output_size], dtype=torch_scale_dtype), requires_grad=False
            )
            layer.register_parameter('weight_scale', weight_scale)
            set_weight_attrs(
                weight_scale,
                {
                    'weight_loader': functools.partial(self.channel_quantize_scale_loader, layer),
                    'param_name': extra_weight_attrs['prefix'] + '.weight_scale',
                },
            )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        k, n = layer.input_size, layer.output_size
        if self.backend == 'mutis':
            pass
        elif self.backend == 'triton':
            if self.config.weight_dtype.is_integer() and self.config.weight_dtype.nbits == 4:
                layer.register_parameter(
                    'triton_b',
                    Parameter(
                        torch.randint(0, 1, size=[k // 8, n], dtype=torch.int32, device='cuda'), requires_grad=False
                    ),
                )
            elif self.config.weight_dtype.is_integer() and self.config.weight_dtype.nbits == 8:
                layer.register_parameter(
                    'triton_b',
                    Parameter(torch.randint(0, 1, size=[k, n], dtype=torch.int8, device='cuda'), requires_grad=False),
                )
            else:
                raise NotImplementedError()
            layer.register_parameter(
                'triton_zeros',
                Parameter(torch.randint(0, 1, size=[k, n // 8], dtype=torch.int32, device='cuda'), requires_grad=False),
            )
            layer.register_parameter('weight_quant', None)
        elif self.backend == 'bitblas':
            import bitblas
            from mutis.kernels.vm.matmul_mma_demo import dtype_to_bitblas

            bitblas_matmul = bitblas.Linear(
                in_features=k,
                out_features=n,
                bias=False,
                A_dtype=dtype_to_bitblas(mutis.float16),
                W_dtype=dtype_to_bitblas(self.config.weight_dtype),
                accum_dtype=dtype_to_bitblas(mutis.float32),
                out_dtype=dtype_to_bitblas(mutis.float16),
                group_size=self.config.group_size,
                with_scaling=True,
                with_zeros=True,
                zeros_mode='original',
                opt_M=[get_m_size()],
            ).cuda()
            b = (
                mutis.from_torch(layer.weight_quant)
                .view(dtype=self.config.weight_dtype, shape=[k, n])
                .to(dtype=mutis.int8)
                .torch()
            )
            bitblas_matmul.load_and_transform_weight(
                weight=b.T.contiguous(),
                scales=layer.weight_scale.T.contiguous(),
                zeros=layer.weight_bias.T.contiguous(),
                bias=None,
            )
            layer.register_parameter('quant_weight', None)
            layer.register_parameter('weight_bias', None)
            layer.register_parameter('weight_scale', None)
            layer.register_module('bitblas_matmul', bitblas_matmul)
        else:
            raise NotImplementedError(self.backend)

    def apply(self, layer: LinearBase, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.size(0) != get_m_size():
            # used to detect memory usage
            return (torch.rand([x.size(0), layer.output_size], dtype=torch.float16, device='cuda') - 0.5) / 10.0
        if self.backend == 'mutis':
            return matmul_mma(
                m=x.size(0),
                n=layer.output_size,
                k=layer.input_size,
                group_size=self.config.group_size,
                a=x,
                b=layer.weight_quant,
                scales=layer.weight_scale,
                zeros=layer.weight_bias,
                a_dtype=mutis.dtype_from_torch(x),
                b_dtype=self.config.weight_dtype,
                c_dtype=mutis.dtype_from_torch(x),
            )
        elif self.backend == 'triton':
            from mutis.kernels.vm.matmul_mma_demo import triton_quantized_gemm

            return triton_quantized_gemm(
                a=x,
                w=layer.triton_b,
                scales=layer.weight_scale,
                zeros=layer.triton_zeros,
                group_size=self.config.group_size,
                b_dtype=self.config.weight_dtype,
            )
        elif self.backend == 'bitblas':
            return layer.bitblas_matmul(x)
        else:
            raise NotImplementedError()


# QUANTIZATION_METHODS['mutis'] = MutisQuantConfig
# QUANTIZATION_METHODS['mutis_triton'] = MutisTritonQuantConfig
# QUANTIZATION_METHODS['mutis_bitblas'] = MutisBitblasQuantConfig

_m_size: Optional[int] = None


def get_m_size() -> int:
    global _m_size
    return _m_size


def set_m_size(m_size):
    global _m_size
    _m_size = m_size
