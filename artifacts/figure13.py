import pandas as pd
from hidet.ir import data_type
from bench_kernel import bench_configs
from mutis.kernels.baselines import MatmulLayer
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pandas import DataFrame
from aesthetic import colors, executor2color, executor2label, ranked_executors
from hidet.ir.type import data_type
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from utils import darken_color


def get_figure13_configs():
    configs = []
    for m in [
        1,
        4,
        8,
        16,
        2048,
        4096,
        8192,
        12288,
    ]:
        for k, n in [
            (8192, 57344),
        ]:
            for a_dtype in [
                'float16'
            ]:
                for (b_dtype, runners) in [
                    ('float16', ['torch-f16']),
                    ('float6_e3m2', ['quant-llm', 'mutis']),
                    ('uint4b', ['triton', 'bitblas', 'mutis']),
                ]:
                    for runner in runners:
                        if data_type(b_dtype).is_integer():
                            group_size = 128
                        else:
                            # since quant-llm does not support group size, for fair comparison, we set it to -1
                            # for both quant-llm and mutis
                            group_size = -1
                        configs.append([runner, a_dtype, b_dtype, group_size, m, k, n])
    return configs

def run_experiments():
    configs = get_figure13_configs()
    df = bench_configs(configs, warmup=10, repeat=50)
    pd.options.display.max_rows = None  # Show all rows
    return df

# use font "Liberation Sans" for all text in the plot
plt.rcParams['font.family'] = 'Liberation Sans'
plt.rcParams['font.size'] = 12

baseline = 'torch-f16'


def process(
    df: DataFrame,
    a_dtype: str,
    group_size: int,
    batch_sizes: List[int],
    gpu='NVIDIA-L40S'
) -> DataFrame:
    df['m-n-k'] = df.apply(lambda row: '{}-{}-{}'.format(row['m'], row['n'], row['k']), axis=1)
    df = df[(df['k'] == 8192) & (df['n'] == 57344)]
    df = df[df['device'] == gpu]

    # filter out the rows with the specified gpu and workloads
    df = df[(df['group_size'] == group_size) | (df['runner'] == 'torch-f16') | (df['b_dtype'] == 'float6_e3m2')]
    df = df[df['a_dtype'] == a_dtype]

    # filter out the rows with the specified batch sizes
    df = df[df['m'].isin(batch_sizes)]

    # Add aggregation by mean for duplicate rows
    group_columns = ['device', 'runner', 'a_dtype', 'b_dtype', 'group_size', 'm', 'k', 'n']
    df = df.groupby(group_columns, as_index=False)['latency'].mean()

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

    runners = ['triton', 'quant-llm', 'bitblas', 'mutis']
    df = df[df['runner'].isin(runners)]
    return df


def plot(df: DataFrame, out_fname):
    print(df)
    df = df.sort_values(by=['runner', 'b_dtype'])

    # aesthetics
    width_scale = 1.9
    height = 3.2 * 1.05
    width = width_scale * height
    fig_size = (width, height)

    dtype2tick = {
        'float6_e3m2': 'f6',
        'uint4b': 'u4',
    }

    # workloads = sorted(list(df['m-n-k'].unique()), key=lambda x: tuple(map(int, list(x.split('-')))))
    executors = sorted(list(df['runner'].unique()), key=lambda x: ranked_executors.index(x))
    dtypes = list(df['b_dtype'].unique())
    dtypes = [dtype for dtype in dtypes if dtype in dtype2tick]
    dtypes = sorted(dtypes, reverse=True, key=lambda d: (data_type(d).nbits, data_type(d).is_integer()))

    # Create the main plot as before
    fig = plt.figure(figsize=fig_size)
    axes = fig.subplots(1, 1)

    # Add inset axes for zoomed region (fixed bbox_to_anchor)
    axins = inset_axes(axes, width="30%", height="48%",
                       bbox_to_anchor=(0.33, 0.20, 1, 1),  # (x0, y0, width, height)
                       bbox_transform=axes.transAxes,
                       loc='center')

    # Add baseline to both plots
    axes.axhline(y=1, lw=1, color=colors[-1][-1], linestyle='--', label='cuBLAS (f16)')
    axins.axhline(y=1, lw=1, color=colors[-1][-1], linestyle='--')

    legend_items = 1
    # Plot data on both axes
    for executor in executors:
        dtype2marker = {
            'float6_e3m2': 'o',
            'uint4b': 'o',
        }
        dtype2linestyle = {
            'float6_e3m2': '--',
            'uint4b': '-',
        }
        filtered_dtypes = [dtype for dtype in dtypes if len(df[(df['runner'] == executor) & (df['b_dtype'] == dtype)]) > 0]
        for dtype in filtered_dtypes:
            data = df[(df['runner'] == executor) & (df['b_dtype'] == dtype)]
            print(executor, dtype)
            print(data)
            x_pos = range(len(data['m']))
            speedups = data['speedup'].values
            label = f"{executor2label[executor]} ({dtype2tick[dtype]})"
            label = label.replace(' (Ours)', '')

            # Plot on main axes
            color = darken_color(colors[executor2color[executor]][0], sat_factor=0.88, light_factor=0.75)
            axes.plot(x_pos, speedups, marker=dtype2marker[dtype],
                      label=label, color=color,
                      linewidth=2, linestyle=dtype2linestyle[dtype])

            # Plot on inset axes (without label to avoid duplicate legend)
            axins.plot(x_pos, speedups, marker=dtype2marker[dtype],
                       color=color,
                       linewidth=1, linestyle=dtype2linestyle[dtype])
            legend_items += 1

    # Configure zoom region
    batch_sizes = sorted(df['m'].unique())
    large_tokens_idx = [i for i, size in enumerate(batch_sizes) if size >= 4096]
    axins.set_xlim(large_tokens_idx[0] - 0.5, large_tokens_idx[-1] + 0.5)
    axins.set_ylim(0.45, 1.20)

    # Set ticks for both plots
    axes.set_xticks(range(len(batch_sizes)))
    axes.set_xticklabels(batch_sizes)

    # Modified inset axes ticks
    large_tokens_indices = [i for i in range(len(batch_sizes)) if batch_sizes[i] >= 4096]
    axins.set_xticks(large_tokens_indices)
    axins.set_xticklabels([batch_sizes[i] for i in large_tokens_indices], fontsize=10)
    axins.tick_params(axis='both', which='both', length=2)

    # Add y-axis ticks for inset
    axins.set_yticks([0.5, 1.0])
    axins.set_yticklabels([0.5, 1.0], fontsize=10)

    axes.set_ylabel('Speedup')
    axes.set_xlabel('Batch Size')
    axes.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.00),
        ncol=legend_items // 2,
    )

    # Add grid for better readability
    axes.grid(True, which='both', linestyle='--', alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
    # fig.show()
    fig.savefig(out_fname, bbox_inches='tight')

def main():
    df = run_experiments()
    results_dir = os.environ.get('TILUS_ARTIFACT_RESULTS_DIR', './results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'figure13.txt'), 'w') as f:
        f.write(df.to_string(index=False))
    df = process(df, a_dtype='float16', group_size=128, batch_sizes=[1, 4, 8, 16, 4096, 8192, 12288])
    plot(df, out_fname=os.path.join(results_dir, 'figure13.pdf'))

if __name__ == '__main__':
    main()
