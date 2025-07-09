import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame

from aesthetic import colors, executor2color, executor2label, ranked_executors
from bench_kernel import bench_configs
from hidet.ir import data_type
from mutis.kernels.baselines import MatmulLayer
from utils import fill_color

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# use font "Liberation Sans" for all text in the plot
plt.rcParams['font.family'] = 'Liberation Sans'
plt.rcParams['font.size'] = 12

baseline = 'torch-f16'

def get_figure9_configs():
    configs = []
    for m in [
        1,
        16
    ]:
        for k, n in [
            (8192, 8192),
            (8192, 57344),
            (28672, 8192)
        ]:
            for (b_dtype, runners) in [
                ('float16', ['torch-f16']),
                ('uint8', ['triton', 'bitblas', 'mutis']),
                ('float6_e3m2', ['quant-llm', 'mutis']),
                ('uint4b', ['triton', 'bitblas', 'mutis']),
                ('uint2b', ['bitblas', 'mutis']),
                ('uint1b', ['bitblas', 'mutis']),
                ('int4b', ['marlin', 'mutis']),
            ]:
                for runner in runners:
                    for a_dtype in [
                        'float16'
                    ]:
                        if data_type(b_dtype).is_integer():
                            group_size = 128
                        else:
                            # since quant-llm does not support group size, for fair comparison, we set it to -1
                            # for both quant-llm and mutis
                            group_size = -1
                        if not MatmulLayer.supports(runner_name=runner, a_dtype=a_dtype, b_dtype=b_dtype):
                            continue
                        configs.append([runner, a_dtype, b_dtype, group_size, m, k, n])
    return configs

def run_experiments():
    configs = get_figure9_configs()
    df = bench_configs(configs, warmup=10, repeat=50)
    pd.options.display.max_rows = None  # Show all rows
    return df

def process(df: DataFrame, gpu: str, bs: int) -> DataFrame:
    """
    Given data frame looks like:
             device     runner  a_dtype  b_dtype  group_size   m     k      n   latency
    0   NVIDIA-L40S  torch-f16  float16  float16          -1   1  8192  57344  1.368060
    1   NVIDIA-L40S  torch-f16  float16  float16          -1   2  8192  57344  1.383420
    2   NVIDIA-L40S  torch-f16  float16  float16          -1   3  8192  57344  1.384450
    3   NVIDIA-L40S  torch-f16  float16  float16          -1   4  8192  57344  1.385470
    4   NVIDIA-L40S  torch-f16  float16  float16          -1   8  8192  57344  1.386500
    ..          ...        ...      ...      ...         ...  ..   ...    ...       ...
    """
    workloads = [
        '8192-8192',
        '57344-8192',
        '8192-28672',
    ]
    workloads = ['{}-{}'.format(bs, wl) for wl in workloads]
    df['m-n-k'] = df.apply(lambda row: '{}-{}-{}'.format(row['m'], row['n'], row['k']), axis=1)
    df = df[df['m-n-k'].isin(workloads)]
    df = df[~((df['runner'] == 'bitblas') & (df['b_dtype'] == 'int4b'))]

    # filter out the rows with the specified gpu and workloads
    df = df[df['device'] == gpu]

    # calculate speedup
    # create a mapping of baseline latencies for each workload
    baseline_df = df[df['runner'] == baseline].copy()
    baseline_latencies = {}
    for _, row in baseline_df.iterrows():
        key = (row['m'], row['k'], row['n'])
        baseline_latencies[key] = row['latency']

    # calculate speedup for each row by comparing with corresponding baseline
    df['baseline'] = df.apply(lambda row: baseline_latencies[(row['m'], row['k'], row['n'])], axis=1)
    df['speedup'] = df['baseline'] / df['latency']

    runners = ['triton', 'quant-llm', 'bitblas', 'marlin', 'mutis']
    df = df[df['runner'].isin(runners)]


    return df


def plot(df: DataFrame, out_fname):
    df = df.sort_values(by=['m-n-k', 'runner', 'b_dtype'])

    # aesthetics
    bar_width = 3
    bar_sep_width = 0.0
    dtype_sep_width = 1
    workload_sep_width = 2.5

    dtype2tick = {
        'uint8': 'u8',
        'float6_e3m2': 'f6',
        'uint4b': 'u4',
        'int4b': 'i4',
        'uint2b': 'u2',
        'uint1b': 'u1',
    }

    workloads = sorted(list(df['m-n-k'].unique()), key=lambda x: tuple(map(int, list(x.split('-')))))
    executors = sorted(list(df['runner'].unique()), key=lambda x: ranked_executors.index(x))
    dtypes = list(df['b_dtype'].unique())
    dtypes = [dtype for dtype in dtypes if dtype in dtype2tick]
    dtypes = sorted(dtypes, reverse=True, key=lambda d: (data_type(d).nbits, data_type(d).is_integer()))

    fig = plt.figure(figsize=(12, 4.8))
    axes_list = fig.subplots(2, 1)

    for ax_idx, (axes, bs) in enumerate(zip(axes_list, [1, 16])):
        # Filter the dataframe for the current batch size using 'm' column
        df_bs = df[df['m'] == bs]

        # Filter workloads for current batch size
        current_workloads = [w for w in workloads if w.startswith(f'{bs}-')]

        executor_list = []
        x_list = []
        y_list = []
        label_list = []

        tick_pos_list = []
        tick_label_list = []

        workload_ranges = []
        workload_labels = []

        current_x = 0.0
        for workload in current_workloads:  # Use filtered workloads
            workload_start = current_x
            for dtype in dtypes:
                dtype_start = current_x
                for executor in executors:
                    speedup = df_bs[(df_bs['m-n-k'] == workload) & (df_bs['runner'] == executor) & (df_bs['b_dtype'] == dtype)]['speedup']
                    if len(speedup) == 0:
                        continue
                    executor_list.append(executor)
                    x_list.append(current_x)
                    y_list.append(speedup.values[0])
                    if executor == 'mutis':
                        others = df_bs[(df_bs['m-n-k'] == workload) & (df_bs['runner'] != executor) & (df_bs['b_dtype'] == dtype)]
                        if len(others) > 0:
                            label_list.append('{:.2f}x'.format(speedup.values[0] / max(others['speedup'].values)))
                        else:
                            label_list.append('')
                    else:
                        label_list.append('')
                    current_x += bar_width + bar_sep_width
                current_x -= bar_sep_width
                dtype_end = current_x
                current_x += dtype_sep_width

                tick_pos_list.append((dtype_start + dtype_end) / 2)
                tick_label_list.append(dtype2tick[dtype])
            current_x -= dtype_sep_width
            workload_end = current_x
            current_x += workload_sep_width

            workload_ranges.append((workload_start, workload_end))
            workload_labels.append(workload)
        current_x -= workload_sep_width

        executor2bar = {}
        for executor in executors:
            filtered_x_list = [x for x, e in zip(x_list, executor_list) if e == executor]
            filtered_y_list = [y for y, e in zip(y_list, executor_list) if e == executor]
            filtered_label_list = [l for l, e in zip(label_list, executor_list) if e == executor]
            bar = axes.bar(
                filtered_x_list,
                filtered_y_list,
                color=fill_color(colors[executor2color[executor]][0]),
                edgecolor=colors[-1][0],
                width=bar_width,
                align='edge',
                linewidth=1,
                label=executor2label[executor],
            )
            # Add bar text labels
            axes.bar_label(bar, fmt='%.1f', padding=1, fontsize=9)
            executor2bar[executor] = bar

        # minor ticks

        ymax = df_bs['speedup'].max()

        axes.set_ylabel('Speedup')
        axes.text(1.02, 0.5, f'BS={bs}', transform=axes.transAxes,
                  rotation=90, va='center')

        axes.set_xticks(tick_pos_list, tick_label_list, ha='center')
        axes.tick_params(axis='x', which='both', length=0)
        # major ticks
        for t, workload_range in enumerate(workload_ranges):
            for pos in workload_range:
                if ax_idx == 0:
                    ymin = -0.12
                else:
                    ymin = -0.25
                axes.axvline(x=pos, ymin=ymin, ymax=0, color=colors[-1][0], lw=1, clip_on=False)

            axes.hlines(
                y=1,
                lw=1,
                xmin=workload_range[0],
                xmax=workload_range[1],
                color=colors[-1][-1],
                label='cuBLAS (fp16)' if t == 0 else None,
                linestyle='--',
            )
        if ax_idx == len(axes_list) - 1:
            major_ticks = [(start + end) / 2 for start, end in workload_ranges]
            for pos, label in zip(major_ticks, workload_labels):
                label = label.replace('16-', 'BS-')
                axes.text(pos, -0.25 * ymax, label, ha='center')

        if ax_idx == 0:
            axes.set_ylabel('Speedup')
        axes.set_xlim(-bar_width, current_x + bar_width)
        axes.set_ylim(0, ymax * 1.15)
        # if ax_idx == 0:
        #     axes.set_ylim(0, ymax * 1.15)
        # else:
        #     axes.set_ylim(0, ymax * 0.95)

    axes_list[0].legend(
        bbox_to_anchor=(0.5, 1.0), loc='lower center',
        ncol=len(executors) + 1
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
    # fig.show()
    fig.savefig(out_fname, bbox_inches='tight')


def analyze_speedup(df: DataFrame):
    """Analyze speedup of mutis compared to other implementations using geometric mean."""
    competitors = ['triton', 'bitblas', 'quant-llm', 'marlin']

    # Create lists to store detailed results
    all_results = []

    for competitor in competitors:
        mutis_rows = df[df['runner'] == 'mutis']
        comp_rows = df[df['runner'] == competitor]

        common_configs = pd.merge(
            mutis_rows,
            comp_rows,
            on=['m-n-k', 'b_dtype'],
            suffixes=('_mutis', '_comp')
        )

        if len(common_configs) == 0:
            continue

        # Calculate speedups for each configuration
        for (workload, dtype), group in common_configs.groupby(['m-n-k', 'b_dtype']):
            batch_size = int(workload.split('-')[0])
            speedup = group['speedup_mutis'].iloc[0] / group['speedup_comp'].iloc[0]
            all_results.append({
                'competitor': competitor,
                'workload': workload,
                'dtype': dtype,
                'batch_size': batch_size,
                'speedup': speedup
            })

        # Calculate and print summary statistics for each batch size
        for bs in df['m'].unique():
            bs_configs = common_configs[common_configs['m_mutis'] == bs]
            if len(bs_configs) > 0:
                speedups = bs_configs['speedup_mutis'] / bs_configs['speedup_comp']
                geomean = np.exp(np.mean(np.log(speedups)))
                print(f"\nMutis vs {competitor} (BS={bs}):")
                print(f"Geometric mean speedup: {geomean:.2f}x")
                print(f"Number of configurations compared: {len(speedups)}")

                # Add geometric mean to results
                all_results.append({
                    'competitor': competitor,
                    'workload': 'geomean',
                    'dtype': 'all',
                    'batch_size': bs,
                    'speedup': geomean
                })

        # Calculate overall geometric mean across all batch sizes
        all_speedups = common_configs['speedup_mutis'] / common_configs['speedup_comp']
        overall_geomean = np.exp(np.mean(np.log(all_speedups)))
        print(f"\nMutis vs {competitor} (Overall):")
        print(f"Overall geometric mean speedup: {overall_geomean:.2f}x")
        print(f"Total configurations compared: {len(all_speedups)}")

        # Add overall geometric mean to results
        all_results.append({
            'competitor': competitor,
            'workload': 'overall_geomean',
            'dtype': 'all',
            'batch_size': 'all',
            'speedup': overall_geomean
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    return results_df


def main():
    results_dir = os.environ.get('TILUS_ARTIFACT_RESULTS_DIR', './results')
    df = run_experiments()
    print(df)

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'figure9.txt'), 'w') as f:
        f.write(df.to_string(index=False))

    # GPUs
    gpus = list(df['device'].unique())
    if len(gpus) != 1:
        raise ValueError(f"Expected exactly one GPU, but found: {gpus}")
    gpu = gpus[0]

    df_bs1 = process(df, bs=1, gpu=gpu)
    df_bs16 = process(df, bs=16, gpu=gpu)

    # Combine the processed dataframes for plotting
    combined_df = pd.concat([df_bs1, df_bs16])

    # Plot the combined dataframe
    plot(combined_df, out_fname=os.path.join(results_dir, 'figure9.pdf'))


if __name__ == '__main__':
    main()
