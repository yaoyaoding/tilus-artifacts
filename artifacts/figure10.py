import pandas as pd
from hidet.ir import data_type
from mutis.kernels.baselines import MatmulLayer
from bench_kernel import bench_configs
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pandas import DataFrame
from aesthetic import colors, executor2color, executor2label, ranked_executors
from hidet.ir.type import data_type
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from utils import darken_color

def get_figure10_configs():
    configs = []
    for a_dtype in [
        'float16',
    ]:
        for b_dtype in [
            'float16',
            'uint8',
            'uint7b',
            'uint6b',
            'uint5b',
            'uint4b',
            'uint3b',
            'uint2b',
            'uint1b',
            'int8',
            'int7b',
            'int6b',
            'int5b',
            'int4b',
            'int3b',
            'int2b',
            'int1b',
            'float8_e4m3',
            'float7_e3m3',
            'float6_e3m2',
            'float5_e2m2',
            'float4_e2m1',
            'float3_e1m1',
        ]:
            if data_type(a_dtype).is_signed_integer() and not data_type(b_dtype).is_signed_integer():
                continue
            for m in [
                16
            ]:
                for k, n in [
                    (8192, 57344),
                ]:
                    for runner in [
                        'torch-f16',
                        'mutis',
                    ]:
                        group_size = -1
                        if not MatmulLayer.supports(runner_name=runner, a_dtype=a_dtype, b_dtype=b_dtype):
                            continue
                        configs.append([runner, a_dtype, b_dtype, group_size, m, k, n])
    return configs

def run_experiments():
    configs = get_figure10_configs()
    df = bench_configs(configs, warmup=10, repeat=50)
    pd.options.display.max_rows = None  # Show all rows
    return df



# use font "Liberation Sans" for all text in the plot
plt.rcParams['font.family'] = 'Liberation Sans'
plt.rcParams['font.size'] = 12

baseline = 'torch-f16'


def process(df: DataFrame, a_dtypes: List[str], b_dtypes: List[str], bs=16, gpu='NVIDIA-L40S') -> DataFrame:
    """
    Given data frame looks like:
             device     runner  a_dtype  b_dtype  group_size   m     k      n   latency
    0   NVIDIA-L40S  torch-f16  float16  float16          -1   1  8192  57344  1.368060
    1   NVIDIA-L40S  torch-f16  float16  float16          -1   2  8192  57344  1.383420
    2   NVIDIA-L40S  torch-f16  float16  float16          -1   3  8192  57344  1.384450
    3   NVIDIA-L40S  torch-f16  float16  float16          -1   4  8192  57344  1.385470
    4   NVIDIA-L40S  torch-f16  float16  float16          -1   8  8192  57344  1.386500
    ..          ...        ...      ...      ...         ...  ..   ...    ...       ...
    19    NVIDIA-L4      mutis  float16    int5b         128  16  8192  57344  1.206270
    20    NVIDIA-L4      mutis  float16    int4b         128  16  8192  57344  0.988160
    21    NVIDIA-L4      mutis  float16    int3b         128  16  8192  57344  0.780800
    22    NVIDIA-L4      mutis  float16    int2b         128  16  8192  57344  0.574464
    23    NVIDIA-L4      mutis  float16    int1b         128  16  8192  57344  0.435200
    """
    # workloads = [
    #     # '10240-8192',
    #     # '8192-8192',
    #     '57344-8192',
    #     # '8192-28672',
    # ]
    # workloads = ['{}-{}'.format(bs, wl) for wl in workloads]
    # df = df[df['m-n-k'].isin(workloads)]
    df['m-n-k'] = df.apply(lambda row: '{}-{}-{}'.format(row['m'], row['n'], row['k']), axis=1)
    df = df[(df['k'] == 8192) & (df['n'] == 57344)]

    # filter out the rows with the specified gpu and workloads
    df = df[df['device'] == gpu]

    # filter out the rows with the specified batch sizes
    df = df[df['m'] == bs]

    # calculate speedup
    # create a mapping of baseline latencies for each workload
    print(df)
    baseline_df = df[df['runner'] == baseline].copy()
    baseline_latencies = {}
    for _, row in baseline_df.iterrows():
        key = (row['m'], row['k'], row['n'])
        baseline_latencies[key] = row['latency']

    # calculate speedup for each row by comparing with corresponding baseline
    df['baseline'] = df.apply(lambda row: baseline_latencies[(row['m'], row['k'], row['n'])], axis=1)
    df['speedup'] = df['baseline'] / df['latency']

    df = df[df['a_dtype'].isin(a_dtypes)]
    df = df[df['b_dtype'].isin(b_dtypes)]

    # Filter out rows where a_dtype is 'int' and b_dtype is 'uint' or 'float'
    df = df[~((df['a_dtype'].str.startswith('int')) &
              (df['b_dtype'].str.startswith(('uint', 'float'))))]

    # runners = ['triton', 'quant-llm', 'bitblas', 'mutis']
    runners = ['mutis']
    df = df[df['runner'].isin(runners)]

    # sort according to (a_dtype, b_dtype)
    df = df.sort_values(by=['a_dtype', 'b_dtype'])

    # aggremate the data by taking the mean of latency and speedup by (a_dtype, b_dtype)
    df = df.groupby(['device', 'runner', 'a_dtype', 'b_dtype', 'group_size', 'm', 'k', 'n', 'm-n-k']).mean().reset_index()

    # add two rows each for float2_e1m0 and float1_e0m0 as b_dtype
    return df


def plot(df: DataFrame, out_fname):
    """
    the df looks like:
             device runner  a_dtype      b_dtype  group_size   m     k      n          m-n-k   latency  baseline   speedup
    0   NVIDIA-L40S  mutis  float16  float3_e1m1          -1  16  8192  57344  16-57344-8192  0.276480   1.39162  5.033348
    1   NVIDIA-L40S  mutis  float16  float4_e2m1          -1  16  8192  57344  16-57344-8192  0.348160   1.39162  3.997070
    2   NVIDIA-L40S  mutis  float16  float5_e2m2          -1  16  8192  57344  16-57344-8192  0.423936   1.39162  3.282618
    3   NVIDIA-L40S  mutis  float16  float6_e3m2          -1  16  8192  57344  16-57344-8192  0.495989   1.39162  2.805748
    4   NVIDIA-L40S  mutis  float16  float7_e3m3          -1  16  8192  57344  16-57344-8192  0.570368   1.39162  2.439863
    5   NVIDIA-L40S  mutis  float16  float8_e4m3          -1  16  8192  57344  16-57344-8192  0.643584   1.39162  2.162297
    6   NVIDIA-L40S  mutis  float16        int1b         128  16  8192  57344  16-57344-8192  0.151488   1.39162  9.186338
    7   NVIDIA-L40S  mutis  float16        int2b         128  16  8192  57344  16-57344-8192  0.222208   1.39162  6.262691
    8   NVIDIA-L40S  mutis  float16        int3b         128  16  8192  57344  16-57344-8192  0.295904   1.39162  4.702944
    9   NVIDIA-L40S  mutis  float16        int4b         128  16  8192  57344  16-57344-8192  0.367616   1.39162  3.785526
    10  NVIDIA-L40S  mutis  float16        int5b         128  16  8192  57344  16-57344-8192  0.443888   1.39162  3.135070
    11  NVIDIA-L40S  mutis  float16        int6b         128  16  8192  57344  16-57344-8192  0.515072   1.39162  2.701797
    12  NVIDIA-L40S  mutis  float16        int7b         128  16  8192  57344  16-57344-8192  0.588864   1.39162  2.363228
    13  NVIDIA-L40S  mutis  float16         int8         128  16  8192  57344  16-57344-8192  0.662528   1.39162  2.100470
    14  NVIDIA-L40S  mutis  float16       uint1b         128  16  8192  57344  16-57344-8192  0.148040   1.39162  9.400380
    15  NVIDIA-L40S  mutis  float16       uint2b         128  16  8192  57344  16-57344-8192  0.222120   1.39162  6.265173
    16  NVIDIA-L40S  mutis  float16       uint3b         128  16  8192  57344  16-57344-8192  0.295936   1.39162  4.702436
    17  NVIDIA-L40S  mutis  float16       uint4b         128  16  8192  57344  16-57344-8192  0.367957   1.39162  3.782021
    18  NVIDIA-L40S  mutis  float16       uint5b         128  16  8192  57344  16-57344-8192  0.443392   1.39162  3.138577
    19  NVIDIA-L40S  mutis  float16       uint6b         128  16  8192  57344  16-57344-8192  0.515072   1.39162  2.701797
    20  NVIDIA-L40S  mutis  float16       uint7b         128  16  8192  57344  16-57344-8192  0.588800   1.39162  2.363485
    21  NVIDIA-L40S  mutis  float16        uint8         128  16  8192  57344  16-57344-8192  0.663109   1.39162  2.098629
    """
    # aesthetics
    fig_size = (8.6 / 1.2, 2.5 / 1.2 * 1.1)

    # Create figure and axis
    plt.figure(figsize=fig_size)

    # Categorize dtypes and get bit widths
    def get_dtype_info(dtype):
        if dtype.startswith('float'):
            kind = 'float'
            bits = int(dtype[5]) if len(dtype) > 5 else 16
        elif dtype.startswith('uint'):
            kind = 'uint'
            bits = int(dtype[4]) if len(dtype) > 4 else 8
        elif dtype.startswith('int'):
            kind = 'int'
            bits = int(dtype[3]) if len(dtype) > 3 else 8
        return kind, bits

    # Create mapping of dtypes to (kind, bits)
    dtype_info = {row['b_dtype']: get_dtype_info(row['b_dtype']) for _, row in df.iterrows()}

    # Modify the sorting of kinds and a_dtypes
    kinds = ['uint', 'int', 'float']  # fixed order
    a_dtypes = ['float16', 'bfloat16', 'int8']  # fixed order

    # Only keep row labels that have corresponding data, but maintain specified order
    row_labels = []
    for a in a_dtypes:
        for k in kinds:
            # Check if this combination exists in the data
            if any((row['a_dtype'] == a and dtype_info[row['b_dtype']][0] == k) for _, row in df.iterrows()):
                # row_labels.append((a, k))
                row_labels.append(k)

    # Get unique bit widths for columns
    bits = sorted(list(set(info[1] for info in dtype_info.values())), reverse=True)

    # Create matrix for heatmap
    data = np.zeros((len(row_labels), len(bits)))

    # Fill matrix with speedup values
    for _, row in df.iterrows():
        a_dtype = row['a_dtype']
        kind, bit_width = dtype_info[row['b_dtype']]
        # row_idx = row_labels.index((a_dtype, kind))
        row_idx = row_labels.index(kind)
        col_idx = bits.index(bit_width)
        data[row_idx][col_idx] = row['speedup']

    # Convert zeros to NaN to make them appear white
    data = np.ma.masked_where(data == 0, data)

    # Create heatmap
    custom_cmap = LinearSegmentedColormap.from_list(
        'custom',
        [
            'white',
            # darken_color(colors[1][0], sat_factor=0.1, light_factor=1.0),
            darken_color(colors[2][0], sat_factor=0.75, light_factor=0.65)
        ]
    )
    im = plt.imshow(data, aspect='auto', cmap=custom_cmap, norm=LogNorm(vmin=0.9))

    # Add colorbar
    cbar = plt.colorbar(im, label='Speedup', ticks=[1, 2, 4, 8])
    cbar.set_ticklabels(['1x', '2×', '4×', '8×'])
    cbar.minorticks_off()

    # Set labels
    # plt.yticks(range(len(row_labels)), [f'({a}, {k})' for a, k in row_labels])
    plt.yticks(range(len(row_labels)), row_labels)
    plt.xticks(range(len(bits)), bits)
    plt.xlabel('Weight Bit Width')
    plt.ylabel('Weight Type Kind')

    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(bits)):
            if data[i][j] > 0:
                plt.text(j, i, f'{data[i][j]:.1f}x',
                         ha='center', va='center')

    plt.tight_layout()
    plt.savefig(out_fname, bbox_inches='tight')
    plt.show()
    plt.close()

def main():
    results_dir = os.environ.get('TILUS_ARTIFACT_RESULTS_DIR', './results')
    df = run_experiments()

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'figure10.txt'), 'w') as f:
        f.write(df.to_string(index=False))

    gpus = list(df['device'].unique())
    if len(gpus) != 1:
        raise ValueError(f"Expected exactly one GPU in the data, found: {gpus}")

    df = process(
        df,
        a_dtypes=[
            'float16',
        ],
        b_dtypes=[
            'float16',
            'float8_e4m3', 'float7_e3m3', 'float6_e3m2', 'float5_e2m2', 'float4_e2m1', 'float3_e1m1',
            'uint8', 'uint7b', 'uint6b', 'uint5b', 'uint4b', 'uint3b', 'uint2b', 'uint1b',
            'int8', 'int7b', 'int6b', 'int5b', 'int4b', 'int3b', 'int2b'
        ],
        gpu=gpus[0],
    )
    plot(df, out_fname=os.path.join(results_dir, 'figure10.pdf'))

if __name__ == '__main__':
    main()
