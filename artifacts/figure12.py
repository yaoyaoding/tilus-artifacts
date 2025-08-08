import os
import pandas as pd
import tqdm
from pandas import DataFrame
import torch

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from click.core import batch
from pandas import DataFrame
from aesthetic import colors, executor2color, executor2label, ranked_executors
from hidet.ir.type import data_type
from mutis.kernels.baselines import MatmulLayer
from bench_model import bench
from utils import darken_color, fill_color

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# use font "Liberation Sans" for all text in the plot
plt.rcParams['font.family'] = 'Liberation Sans'
plt.rcParams['font.size'] = 14

baseline = 'torch-f16'
model2label = {
    'google/gemma-2-9b': 'Gemma-2-9B',
    'Qwen/Qwen2.5-Coder-32B-Instruct': 'Qwen2.5-32B',
    'meta-llama/Meta-Llama-3-70B-Instruct': 'Llama-3-70B',
}


def run_experiments():
    device = torch.cuda.get_device_name().replace(' ', '-')
    headers = ['device', 'stage', 'mode', 'model', 'backend', 'a_dtype', 'b_dtype', 'bs', 'tokens', 'group_size', 'latency']
    rows = []
    configs = []
    for model, size in [
        ('Qwen/Qwen2.5-Coder-32B-Instruct', 32),
    ]:
        for (stage, bs, tokens) in [
            ('decode', 1, 2048),
            ('decode', 16, 2048),
            ('prefill', 1, 2048)
        ]:
            for a_dtype in ['float16']:
                for b_dtype, backends in [
                    ('float16', ['torch-f16']),
                    ('uint4b', ['bitblas', 'mutis']),
                ]:
                    for backend in backends:
                        configs.append([model, size, stage, bs, tokens, a_dtype, b_dtype, backend])
    for model, size, stage, bs, tokens, a_dtype, b_dtype, backend in tqdm.tqdm(configs, desc='Running Figure 12 experiments', miniters=1, mininterval=0):
        assert MatmulLayer.supports(backend, a_dtype, b_dtype)

        memory_size = torch.cuda.get_device_properties(None).total_memory * 0.8
        model_size = size * 10 ** 9 * data_type(b_dtype).nbits / 8
        if memory_size < model_size:
            continue

        latency = bench(
            model=model,
            stage=stage,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            backend=backend,
            bs=bs,
            group_size=128 if 'int' in b_dtype else -1,
            gpu_memory_utilization=0.96 if '3090' not in torch.cuda.get_device_name() else 0.84,
            mode='cgraph',
            num_repeat=10,
            mutis_space=2,
            tokens=tokens,
        )
        rows.append(
            [
                device,
                stage,
                'cgraph',
                model,
                backend,
                a_dtype,
                b_dtype,
                bs,
                tokens,
                128 if 'int' in b_dtype else -1,
                latency
            ]
        )

    df = DataFrame(rows, columns=headers)
    return df



def process(df: DataFrame, device: str, backends: List[str], b_dtypes: List[str], models: List[str], stage_bs_tokens: List[Tuple[str, int, int]]) -> DataFrame:
    df = df[df['device'] == device]
    runners = ['torch-f16', 'triton', 'quant-llm', 'bitblas', 'mutis']
    df = df[df['backend'].isin(runners)]
    df = df[df['model'].isin(models)]
    df['stage,bs,tokens'] = df.apply(lambda row: (row['stage'], row['bs'], row['tokens']), axis=1)
    df = df[df['backend'].isin(backends)]
    df = df[df['stage,bs,tokens'].isin(stage_bs_tokens)]
    df = df[df['b_dtype'].isin(b_dtypes)]
    df = df.sort_values(by=['model', 'bs', 'b_dtype', 'backend'])

    return df


def plot(df: DataFrame, models, out_fname: str):
    # aesthetics
    bar_width = 0.8
    bar_sep_width = 0.00
    dtype_sep_width = 0.3

    dtype2tick = {
        'float16': 'f16',
        'uint8': 'u8',
        'float6_e3m2': 'f6',
        'uint4b': 'u4',
        'int4b': 'i4',
        'uint3b': 'u3',
        'uint2b': 'u2',
        'uint1b': 'u1',
    }

    # models = sorted(list(df['model'].unique()))
    batch_sizes = sorted(list(df['bs'].unique()))
    stage_bs_tokens_list = sorted(list(df['stage,bs,tokens'].unique()))
    stage_bs_tokens_list = [(stage, int(bs), int(tokens)) for stage, bs, tokens in stage_bs_tokens_list]
    executors = sorted(list(df['backend'].unique()), key=lambda x: ranked_executors.index(x))
    dtypes = list(df['b_dtype'].unique())
    dtypes = [dtype for dtype in dtypes if dtype in dtype2tick]
    dtypes = sorted(dtypes, reverse=True, key=lambda d: (data_type(d).nbits, data_type(d).is_integer()))

    # Create subplot grid
    n_rows = len(models)
    n_cols = len(stage_bs_tokens_list)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0*n_cols, 1.8*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    executor2bars = {}

    for row, model in enumerate(models):
        for col, (stage, bs, tokens) in enumerate(stage_bs_tokens_list):
            ax = axes[row, col]
            current_x = 0.0
            tick_positions = []
            tick_labels = []

            # Get max latency for this specific subplot
            subplot_df = df[(df['model'] == model) & (df['bs'] == bs) & (df['stage'] == stage)]
            max_latency = subplot_df['latency'].max() if not subplot_df.empty else 0

            for dtype in dtypes:
                dtype_start = current_x
                dtype_data = df[(df['model'] == model) &
                                (df['bs'] == bs) &
                                (df['stage'] == stage) &
                                (df['tokens'] == tokens) &
                                (df['b_dtype'] == dtype)]

                if dtype_data.empty:
                    # Add OOM text for missing entire dtype
                    ax.text((current_x + bar_width/2) + bar_width * 0.05, max_latency * 0.05, 'OOM',
                            ha='center', va='bottom', rotation=90,
                            color=darken_color(colors[0][0]), fontweight='bold')
                    ax.vlines(current_x, 0, max_latency* 0.5, colors='black', linestyles='--', linewidth=1)
                    ax.vlines(current_x + bar_width, 0, max_latency * 0.5, colors='black', linestyles='--', linewidth=1)
                    current_x += bar_width + bar_sep_width  # Reserve space for OOM
                else:
                    for executor in executors:
                        data = dtype_data[dtype_data['backend'] == executor]

                        if not data.empty:
                            latency = data['latency'].values[0]
                            bar = ax.bar(
                                current_x, latency,
                                width=bar_width,
                                color=fill_color(colors[executor2color[executor]][0]),
                                edgecolor=colors[-1][0],
                                linewidth=1,
                                align='edge'
                            )
                            if latency >= 100:
                                ax.bar_label(bar, fmt='%.0f', padding=1, fontsize=9)
                            else:
                                ax.bar_label(bar, fmt='%.1f', padding=1, fontsize=9)
                            if row == 0 and col == 0:
                                executor2bars[executor] = bar
                            current_x += bar_width + bar_sep_width
                dtype_end = current_x - bar_sep_width
                ax.axvline(x=dtype_start, ymin=-0.02, ymax=0, color=colors[-1][0], lw=1, clip_on=False)
                ax.axvline(x=dtype_end, ymin=-0.02, ymax=0, color=colors[-1][0], lw=1, clip_on=False)
                current_x -= bar_sep_width

                tick_positions.append((dtype_start + current_x) / 2)
                tick_labels.append(dtype2tick[dtype])
                current_x += dtype_sep_width
            current_x -= dtype_sep_width

            # Customize subplot with individual max_latency
            ax.set_xticks(tick_positions, tick_labels)
            ax.tick_params(axis='x', which='both', length=0)
            ax.set_ylim(0, max_latency * 1.25)
            ax.set_xlim(-dtype_sep_width, current_x + dtype_sep_width)
            # ax.tick_params(axis='y', labelsize=10)  # Adjust the font size as needed

            # Add titles and labels
            if col == 0:
                ax.set_ylabel('Latency (ms)')
            if row == 0:
                assert isinstance(stage, str)
                title = '{}'.format(stage.capitalize())
                if stage == 'prefill':
                    title += ', Tokens={}'.format(subplot_df['tokens'].values[0])
                else:
                    title += ', Tokens={}'.format(bs)
                ax.set_title(title)
            # if row == len(models)-1:
            #     ax.set_xlabel('Data Type')

            # Add model name as text on the right
            if col == len(stage_bs_tokens_list)-1:
                ax.text(1.02, 0.5, model2label[model],
                        transform=ax.transAxes,
                        rotation=270,
                        va='center')

    # Add legend to the top of the figure
    items = [executor2bars[executor][0] for executor in executors if executor in executor2bars]
    models_executor2label = executor2label.copy()
    models_executor2label['torch-f16'] = 'vLLM'
    labels = [models_executor2label[executor] for executor in executors]
    fig.legend(items, labels,
               bbox_to_anchor=(0.515, 1.00),
               loc='lower center',
               ncol=len(executors)
               )  # Add black edge color to legend items

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, wspace=0.28)

    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
    plt.savefig(out_fname, bbox_inches='tight')
    plt.show()
    plt.close()

def main():
    models = [
        'Qwen/Qwen2.5-Coder-32B-Instruct',
    ]
    stage_bs_tokens = [
        ('decode', 1, 2048),
        ('decode', 16, 2048),
        ('prefill', 1, 2048)
    ]
    backends = [
        'torch-f16',
        'bitblas',
        'mutis',
    ]
    b_dtypes = [
        'float16',
        'uint4b',
    ]
    df = run_experiments()
    results_dir = os.environ.get('TILUS_ARTIFACT_RESULTS_DIR', './results')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'figure12.txt'), 'w') as f:
        f.write(df.to_string(index=False))

    gpus = list(df['device'].unique())
    if len(gpus) != 1:
        raise ValueError('Expected only one GPU in the data, found: {}'.format(gpus))

    df = process(df, device=gpus[0], backends=backends, b_dtypes=b_dtypes, models=models, stage_bs_tokens=stage_bs_tokens)
    print(df)
    out_fname = os.path.join(results_dir, 'figure12.pdf')
    plot(df, models=models, out_fname=out_fname)


if __name__ == '__main__':
    main()

