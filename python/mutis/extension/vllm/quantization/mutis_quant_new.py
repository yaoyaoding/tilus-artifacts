from typing import List, Dict, Any, Optional

import torch
import vllm.model_executor.layers.quantization
from packaging.version import Version

from hidet.ir.type import DataType
from mutis.kernels.baselines import MatmulLayer
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, get_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import QuantizationConfig


class GlobalConfig:
    backend: str
    m_size: int
    a_dtype: DataType
    b_dtype: DataType
    group_size: int


global_config = GlobalConfig()


def set_mutis_config(backend: str, m_size: int, a_dtype: DataType, b_dtype: DataType, group_size: int):
    global_config.backend = backend
    global_config.m_size = m_size
    global_config.a_dtype = a_dtype
    global_config.b_dtype = b_dtype
    global_config.group_size = group_size



class MutisConfig(QuantizationConfig):
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
        return cls()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            return MutisLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        raise NotImplementedError()


class MutisLinearMethod(LinearMethodBase):
    def __init__(self, config: MutisConfig):
        super().__init__()
        self.config: MutisConfig = config

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
        layer.register_module(
            'impl_layer',
            MatmulLayer.create(
                runner_name=global_config.backend,
                a_dtype=global_config.a_dtype,
                b_dtype=global_config.b_dtype,
                group_size=global_config.group_size,
                m=global_config.m_size,
                k=input_size,
                n=output_size,
            ),
        )

    def apply(self, layer: LinearBase, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        return layer.impl_layer.run(x)


if Version(vllm.__version__) >= Version("0.7"):
    from vllm.model_executor.layers.quantization import register_quantization_config
    register_quantization_config('mutis')(MutisConfig)
else:
    QUANTIZATION_METHODS['mutis'] = MutisConfig
