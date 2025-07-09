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
from typing import List, Optional, Callable, Tuple, Any, Dict, Union
import time
import math
import contextlib
from dataclasses import dataclass

import numpy as np


# copied from: https://github.com/openai/triton/blob/main/python/triton/testing.py
def do_bench(fn, warmup=25, rep=100, number=1, reduction_method='mean', percentiles=(0.2, 0.5, 0.8)):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param reduction_method: one of 'mean' 'percentile' 'raw'
    :type reduction_method: str
    :param percentiles: Performance percentile to return in addition to the median.
    :type percentiles: list[float]
    """

    assert reduction_method in ['mean', 'percentile', 'raw']
    if reduction_method == 'percentile':
        assert percentiles is not None

    # Estimate the runtime of the function
    import hidet

    cuda_available = hidet.cuda.available()
    hip_available = hidet.hip.available()

    if not cuda_available and not hip_available:
        raise RuntimeError("No GPU found")

    cur_stream = hidet.cuda.current_stream() if cuda_available else hidet.hip.current_stream()

    def sync():
        if cuda_available:
            cur_stream.synchronize()
        if hip_available:
            cur_stream.synchronize()

    def create_event():
        if cuda_available:
            return hidet.cuda.Event(enable_timing=True)
        if hip_available:
            return hidet.hip.Event(enable_timing=True)
        assert False

    fn()
    sync()
    start_event = create_event()
    end_event = create_event()
    start_event.record()
    for _ in range(5):
        fn()
    end_event.record()
    sync()
    estimate_ms = end_event.elapsed_time(start_event) / 5
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    start_event = [create_event() for i in range(n_repeat)]
    end_event = [create_event() for i in range(n_repeat)]

    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        start_event[i].record()
        for _ in range(number):
            fn()
        end_event[i].record()
    # Record clocks
    sync()
    times = [e.elapsed_time(s) / number for s, e in zip(start_event, end_event)]
    if reduction_method == 'raw':
        return times
    times = np.array(times)
    if reduction_method == 'percentile':
        percentiles = np.quantile(times, percentiles)
        return tuple(percentiles)
    else:
        return np.mean(times).item()


def benchmark_func(
    run_func, warmup=1, repeat=5, maximum_time=50.0, median=True, clear_l2_cache=True, nvtx_scope: Optional[str] = None
) -> Union[List[float], float]:
    """Benchmark given function.

    The given function ``run_func`` will be executed :math:`warmup + repeat * number` times. Each :math:`number` times
    of execution will be grouped and conducted together.

    Parameters
    ----------
    run_func: Callable[[], Any]
        Any callable function to be benchmarked.

    warmup: int
        The number of warm-up executions.

    repeat: int
        The number of repeat times of the group measurement.

    maximum_time: float
        The maximum time (in milliseconds) to run the benchmark. The actual number of repeats will be adjusted to
        ensure that the total time of the benchmark does not exceed this value.

    median: bool
        Whether the median latency is returned, instead of the latency.

    clear_l2_cache: bool
        Whether to clear the L2 cache before each run.

    nvtx_scope: Optional[str]
        The NVTX scope to use for the benchmark. If None, NVTX is not used.

    Returns
    -------
    ret: Union[float, List[float]]
        - When median == True, a single latency number is returned.
        - When median == False, the latency of each repeat is returned, as a list of floats.
    """
    import hidet
    import torch

    use_nvtx = nvtx_scope is not None and hidet.cuda.available()

    num_bytes = 256 * 1024 * 1024
    memory_slab = torch.empty(num_bytes, dtype=torch.int8, device='cuda')

    def flush_l2_cache():
        import hidet

        if hidet.cuda.available() and hidet.cuda.device_count() > 0:
            import hidet.cuda

            hidet.cuda.memset_async(memory_slab.data_ptr(), 0, num_bytes=num_bytes)
        else:
            import hidet.hip

            hidet.hip.memset_async(memory_slab.data_ptr(), 0, num_bytes=num_bytes)

    assert repeat >= 1

    events = [torch.cuda.Event(enable_timing=True) for _ in range(2 * (repeat + warmup))]

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

    # run warmup + 1 times, and use the last run as the latency estimate

    def bench(start: int, end: int) -> List[float]:
        assert start <= end <= len(events) // 2
        torch.cuda.synchronize()
        for i in range(start, end):
            if clear_l2_cache:
                flush_l2_cache()
            events[i * 2].record()
            with nvtx_annotations[i]:
                run_func()
            events[i * 2 + 1].record()
        torch.cuda.synchronize()
        return [events[i * 2].elapsed_time(events[i * 2 + 1]) for i in range(start, end)]

    latencies = bench(0, end=warmup + 1)
    repeat = min(repeat, max(int(math.ceil(maximum_time / latencies[-1])), 1))
    results = [latencies[-1]] + bench(start=warmup + 1, end=warmup + repeat)

    if median:
        return float(np.median(results))
    else:
        return results


@dataclass
class BenchData:
    x_vals: List[Any]
    x_name: str
    y_name: str
    kwargs: Dict[str, Any]
    data: Dict[str, Tuple[List[float], List[float], List[float]]]  # [t_min, t_avg, t_max]

    def show_plot(self, show=True, save_path=None, figsize=None, title=None):
        from matplotlib import pyplot as plt

        if all(isinstance(x, (float, int)) for x in self.x_vals):
            x_vals = self.x_vals
        else:
            x_vals = range(1, len(self.x_vals) + 1)

        plt.figure(figsize=figsize)
        ax = plt.subplot()
        for name, (t_min, t_avg, t_max) in self.data.items():
            p = ax.plot(x_vals, t_avg, label=name)
            color = p[0].get_color()
            ax.fill_between(x_vals, t_min, t_max, alpha=0.15, color=color)
        ax.legend()
        ax.set_xlabel(self.x_name)
        ax.set_ylabel(self.y_name)
        if title is not None:
            ax.set_title(title)
        ax.set_xticks(ticks=x_vals, labels=[str(x) for x in self.x_vals])
        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path)
        return self

    def to_dataframe(self):
        import pandas as pd

        columns = list(self.data.keys())
        df = pd.DataFrame(columns=columns, index=self.x_vals)
        for n in columns:
            df[n] = self.data[n][1]  # get t_avg
        return df

    def print_data(self):
        print(self.to_dataframe())


class Bench:
    def __init__(self, x_vals: List[Any], x_name: str, **kwargs):
        self.x_vals = x_vals
        self.x_name = x_name
        self.y_name = 'ms'
        self.byte_fn = None

        self.kwargs: Dict[str, Any] = kwargs
        self.bench_fns: List[Tuple[str, Callable]] = []
        self.bench_data: Dict[str, Tuple[List[float], List[float], List[float]]] = {}

    def measure_flops(self, byte_fn: Callable[[Any], int]):
        """
        set a function that takes in the config, and the current x_val and returns the number of bytes
        """
        self.byte_fn = byte_fn
        self.y_name = 'TFLOP/s'

    def bench(self, fn: Callable[[Any], Callable[[], Any]], name: Optional[str] = None):
        """
        add a function that takes in the config and int and returns a function to be benchmarked
        to the list of functions to be benchmarked.
        If the name argument is None, the the name for this particular line is fn.__name__
        """
        if name is None:
            if hasattr(fn, '__name__'):
                name = fn.__name__
            else:
                raise ValueError("cannot get name of function")
        self.bench_fns.append((name, fn))
        return self

    def run(self):
        """
        run all the functions that needs to be benchmarked, returning BenchData representing
        the collected results
        """
        for i in self.x_vals:
            for name, fn in self.bench_fns:
                if name not in self.bench_data:
                    self.bench_data[name] = ([], [], [])
                t_min, t_avg, t_max = self.bench_data[name]

                bench_fn = fn(i, **self.kwargs)
                lo, avg, hi = do_bench(bench_fn, reduction_method='percentile', percentiles=(0.2, 0.5, 0.8))
                if self.byte_fn is not None:
                    lo = self.byte_fn(i, **self.kwargs) * 1e-12 / (lo * 1e-3)
                    avg = self.byte_fn(i, **self.kwargs) * 1e-12 / (avg * 1e-3)
                    hi = self.byte_fn(i, **self.kwargs) * 1e-12 / (hi * 1e-3)
                t_min.append(lo)
                t_avg.append(avg)
                t_max.append(hi)
        return BenchData(self.x_vals, self.x_name, self.y_name, self.kwargs, self.bench_data)
