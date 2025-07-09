from typing import Tuple

import torch


def to_float8_e4m3(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    finfo = torch.finfo(torch.float8_e4m3fn)

    down_scale = finfo.max / max(tensor.abs().max().item(), 1e-6)
    up_scale = torch.full(size=[], fill_value=1.0 / down_scale, dtype=torch.float32, device=tensor.device)

    tensor = torch.clamp(tensor * down_scale, min=finfo.min, max=finfo.max).to(dtype=torch.float8_e4m3fn)
    return tensor, up_scale


def quantize_fp32_to_fp8e4m3(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    finfo = torch.finfo(torch.float8_e4m3fn)

    down_scale = finfo.max / max(tensor.abs().max().item(), 1e-6)
    scale = torch.full(size=[], fill_value=1.0 / down_scale, dtype=torch.float32, device=tensor.device)

    tensor = torch.clamp(tensor * down_scale, min=finfo.min, max=finfo.max).to(dtype=torch.float8_e4m3fn)
    return tensor, scale
