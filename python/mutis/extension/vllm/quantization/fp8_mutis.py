from typing import List, Dict, Any, Optional
from abc import abstractmethod

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

import vllm
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from vllm import _custom_ops as ops

from mutis.kernels.scaled_mm_fp8 import scaled_mm_fp8_e4m3


class Fp8MutisConfig(Fp8Config):
    """
    Configuration for FP8 quantization method using Mutis kernels.
    """

    def get_name(self) -> str:
        return "fp8_mutis"

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional[QuantizeMethodBase]:
        """Get the quantize method to use for the quantized layer.

        Args:
            layer: The layer for the quant method.
            prefix: The full name of the layer in the state dict
        Returns:
            The quantize method. None if the given layer doesn't support quant
            method.
        """
        if isinstance(layer, LinearBase):
            return Fp8MutisLinearMethod(self)
        return None


class Fp8MutisLinearMethod(Fp8LinearMethod):
    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)
        self.use_marlin = False

    def process_weights_after_loading(self, layer: Module) -> None:
        super().process_weights_after_loading(layer)
        layer.weight = Parameter(layer.weight.contiguous(), requires_grad=False)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        out_dtype = x.dtype
        x, scale_x = ops.scaled_fp8_quant(input=x, scale=layer.input_scale)
        return scaled_mm_fp8_e4m3(
            a=x, b=layer.weight, scale_a=scale_x, scale_b=layer.weight_scale, bias=bias, out_dtype=out_dtype
        )


QUANTIZATION_METHODS['fp8_mutis'] = Fp8MutisConfig
