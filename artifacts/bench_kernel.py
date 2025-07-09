import csv
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import tabulate
import torch
import tqdm

import mutis
from hidet.ir.type import data_type
from mutis.kernels.baselines.layers import MatmulLayer
from utils import configure_kernel_cache_dir

# Set the CUDA path to ensure that TVM (used by bitblas) can find the CUDA toolkit.
os.environ['PATH'] = '/usr/local/cuda-12.6/bin:' + os.environ['PATH']

# Setup cache
headers = ['device', 'runner', 'a_dtype', 'b_dtype', 'group_size', 'm', 'k', 'n', 'latency']
cache_dir = Path(os.environ.get('TILUS_ARTIFACT_CACHE_DIR', './cache'))
data_path = cache_dir / 'kernels.csv'
cache_dir.mkdir(parents=True, exist_ok=True)
cache: dict[tuple[Any, ...], float] = {}

def load_cache():
    """Load the cache from the CSV file."""
    if data_path.exists():
        with open(data_path, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    # Skip the header row
                    continue
                device, runner_name, a_dtype, b_dtype, group_size, m, k, n, latency = row
                cache[(device, runner_name, a_dtype, b_dtype, int(group_size), int(m), int(k), int(n))] = float(latency)
            if len(cache) > 0:
                print(f"Cache loaded with {len(cache)} entries.")
    else:
        # write the headers to the file if it does not exist
        with open(data_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print("Cache file does not exist. Starting with an empty cache.")

load_cache()

def bench_configs(configs, warmup: int, repeat: int) -> pd.DataFrame:
    device = torch.cuda.get_device_name().replace(' ', '-')
    rows = []

    for config in tqdm.tqdm(configs, desc='Configs', miniters=1, mininterval=0):
        runner_name, a_dtype, b_dtype, group_size, m, k, n = config
        if (device, runner_name, a_dtype, b_dtype, group_size, m, k, n) in cache:
            latency = cache[(device, runner_name, a_dtype, b_dtype, group_size, m, k, n)]
        else:
            t1 = time.time()
            runner = MatmulLayer.create(runner_name, a_dtype=data_type(a_dtype), b_dtype=data_type(b_dtype), group_size=group_size, m=m, k=k, n=n)

            # first run to check runtime error
            runner.run()
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                print(f"Runtime error in {runner_name} {a_dtype} {b_dtype} {group_size} {m} {k} {n}")
                continue

            # release the loaded shared libraries in this config (otherwise, we might load too many shared libraries)
            mutis.empty_jit_cache()

            # benchmark
            latency = runner.bench(warmup=warmup, repeat=repeat)
            t2 = time.time()
            cache[(runner_name, a_dtype, b_dtype, group_size, m, k, n)] = latency
            with open(data_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([device, runner_name, a_dtype, b_dtype, group_size, m, k, n, latency])
            rows.append([device, runner_name, a_dtype, b_dtype, group_size, m, k, n, latency])
            print()
            print(pd.DataFrame([rows[-1]], columns=headers))
            print('Time elapsed: {:5.3f} seconds'.format(t2 - t1))
        rows.append([device, runner_name, a_dtype, b_dtype, group_size, m, k, n, latency])

    df = pd.DataFrame(rows, columns=headers)
    return df
