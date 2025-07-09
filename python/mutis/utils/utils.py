from typing import List, Union, Optional
import importlib
import contextlib
import math
import functools

import numpy as np
import torch

import hidet.cuda
import mutis.option
from hidet.ir.dtypes import i32
from hidet.ir.expr import Expr, is_one
from mutis.target import get_current_target


def check_same_elem_type(*args, msg):
    for arg in args:
        if arg.elem_type != args[0].elem_type:
            raise ValueError('{}, got {}'.format(msg, ' '.join(str(v.elem_type) for v in args)))


def broadcast_shape(a_shape: List[Expr], b_shape: List[Expr]) -> List[Expr]:
    a_shape = list(a_shape)
    b_shape = list(b_shape)

    shape = []
    while len(a_shape) < len(b_shape):
        a_shape = [i32.one] + a_shape
    while len(a_shape) > len(b_shape):
        b_shape = [i32.one] + b_shape
    for sa, sb in zip(a_shape, b_shape):
        if is_one(sa):
            shape.append(sb)
        elif is_one(sb):
            shape.append(sa)
        else:
            # we do not support dynamic broadcast where both sa and sb are known at runtime where either can be 1
            shape.append(sa)  # or sb
    return shape


def clear_cache():
    import shutil

    cache_dir: str = mutis.option.get_option('cache_dir')

    shutil.rmtree(cache_dir, ignore_errors=True)


@functools.cache
def _cuda_sleep_kernel():
    from hidet.lang import attrs, script, script_module
    from hidet.lang.types import int64
    from hidet.ir.primitives.cuda.time import nano_sleep

    with script_module() as module:

        @script
        def cuda_sleep_kernel(nanoseconds: int64):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 1

            # since the nano_sleep has a upper bound to sleep, approximately 1 millisecond, we break the given
            # nanoseconds into multiple milliseconds
            for _ in range(nanoseconds // 1000000):
                nano_sleep(1000000)

            nano_sleep(nanoseconds % 1000000)

    return module.build()


def cuda_sleep(nanoseconds: int):
    """
    A sleep cuda kernel that will sleep for given nanoseconds.
    """
    kernel = _cuda_sleep_kernel()
    kernel(nanoseconds)


def benchmark_func(
    run_func,
    warmup=1,
    repeat=5,
    maximum_warmup_time: Optional[float] = None,
    maximum_repeat_time: Optional[float] = None,
    median=True,
    clear_l2_cache=True,
    nvtx_scope: Optional[str] = None,
) -> Union[List[float], float]:
    if maximum_warmup_time is None:
        maximum_warmup_time = 1e8
    if maximum_repeat_time is None:
        maximum_repeat_time = 1e8

    use_nvtx = nvtx_scope is not None and hidet.cuda.available()

    num_bytes = 128 * 1024 * 1024
    memory_slab = torch.empty(num_bytes, dtype=torch.int8, device='cuda')
    memory_slab[:] = 0

    assert repeat >= 1

    events = [torch.cuda.Event(enable_timing=True) for _ in range(2 * (repeat + warmup))]
    for event in events:
        event.record()
    cuda_sleep(0)

    nvtx_annotations = []
    for i in range(warmup + repeat):
        if use_nvtx:
            import nvtx

            stage = 'warmup' if i < warmup else 'run'
            stage_i = i if i < warmup else i - warmup
            name = f'{nvtx_scope} {stage} {stage_i}'.strip()
            nvtx_annotations.append(nvtx.annotate(name))
        else:
            nvtx_annotations.append(contextlib.nullcontext())

    def bench(start: int, end: int) -> List[float]:
        assert start <= end <= len(events) // 2
        torch.cuda.synchronize()
        cuda_sleep((end - start) * 150000)  # sleep 150 microseconds for each kernel launch
        for i in range(start, end):
            if clear_l2_cache:
                # hidet.cuda.memory.memset_async(addr=memory_slab.data_ptr(), value=0, num_bytes=num_bytes)
                memory_slab[:] = 0
            events[i * 2].record()
            with nvtx_annotations[i]:
                run_func()
            events[i * 2 + 1].record()
        torch.cuda.synchronize()
        return [events[i * 2].elapsed_time(events[i * 2 + 1]) for i in range(start, end)]

    # warmup
    torch.cuda.synchronize()
    for i in range(warmup):
        events[i * 2].record()
        run_func()
        events[i * 2 + 1].record()
        torch.cuda.synchronize()
        if events[0].elapsed_time(events[i * 2 + 1]) > maximum_warmup_time:
            estimate_latency = events[0].elapsed_time(events[i * 2 + 1]) / (i + 1)
            break
    else:
        estimate_latency = events[0].elapsed_time(events[warmup * 2 - 1]) / warmup

    repeat = min(repeat, max(int(math.ceil(maximum_repeat_time / estimate_latency)), 1))
    results = bench(start=warmup, end=warmup + repeat)

    if median:
        return float(np.median(results))
    else:
        return results


def unique_file_name(pattern: str) -> Optional[str]:
    """
    Given a pattern like './results/exp/report_%d.txt' and returns a unique file name like `./results/exp/report_1.txt`
    """
    import os

    if pattern.count('%d') == 0:
        os.makedirs(os.path.dirname(pattern), exist_ok=True)
        return pattern
    else:
        assert pattern.count('%d') == 1
        os.makedirs(os.path.dirname(pattern), exist_ok=True)

        i = 0
        while True:
            file_name = pattern % i
            if not os.path.exists(file_name):
                return file_name
            i += 1
