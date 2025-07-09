from typing import Tuple, List
import torch
from hidet.ir.dtypes import float6_e3m2
from mutis.utils import prod


def quantize_fp32_to_fp6e3m2(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    from mutis.kernels.scaled_mm_fp6 import cast_f32_to_f6e3m2, cast_f6e3m2_to_f32

    tensor = tensor.to(torch.float32)
    scale = tensor.abs().max() / float(float6_e3m2.max_value)
    tensor = torch.clip(tensor / scale, min=float(float6_e3m2.min_value), max=float(float6_e3m2.max_value))
    tensor = tensor + 1e-6
    tensor = cast_f32_to_f6e3m2(tensor)
    return tensor, scale


def dequantize_fp6e3m2_to_fp32(shape: List[int], tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    from mutis.kernels.scaled_mm_fp6 import cast_f6e3m2_to_f32

    tensor = cast_f6e3m2_to_f32(prod(shape), tensor).reshape(shape)
    tensor = tensor * scale
    return tensor
