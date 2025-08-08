from typing import Any
import traceback
import csv
import time
import subprocess
import argparse
import gc
import logging
import os
import sys
import contextlib
from pathlib import Path
from typing import List, Sequence, Optional

import numpy as np
import nvtx
import torch
from tabulate import tabulate

import hidet
import mutis
from hidet.ir.type import data_type
from mutis.utils import benchmark_func, unique_file_name
from vllm import LLM
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceGroupMetadata, SequenceData, SequenceStage, SequenceDataDelta
from vllm.worker.model_runner import GPUModelRunnerBase, ModelRunner
from vllm.worker.worker import Worker
from utils import configure_kernel_cache_dir

# Set the CUDA path to ensure that TVM (used by bitblas) can find the CUDA toolkit.
os.environ['PATH'] = '/usr/local/cuda-12.6/bin:' + os.environ['PATH']

# Setup cache
headers = ['device', 'stage', 'backend', 'model', 'bs', 'tokens', 'a_dtype', 'b_dtype', 'group_size', 'latency (ms)']
cache_dir = Path(os.environ.get('TILUS_ARTIFACT_CACHE_DIR', './cache'))
data_path = cache_dir / 'models.csv'
cache_dir.mkdir(parents=True, exist_ok=True)
cache: dict[tuple[Any, ...], float] = {}

# set the log level of vllm package to ERROR
logging.getLogger('vllm').setLevel(logging.ERROR)

parser = argparse.ArgumentParser()

parser.add_argument('--stage', type=str, choices=['prefill', 'decode'])
parser.add_argument('--model', type=str, default='downloads/facebook/opt-125m')
parser.add_argument('--tokenizer', type=str, default=None)
parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
parser.add_argument('--mode', type=str, choices=['eager', 'cgraph'])
parser.add_argument(
    '--backend',
    type=str,
    required=True,
    choices=['torch-f16', 'mutis', 'triton', 'bitblas', 'quant-llm']
)
parser.add_argument('--num_warmup', type=int)
parser.add_argument('--num_repeat', type=int)
parser.add_argument('--bs', type=int, required=True)
parser.add_argument('--tokens', type=int, default=1024)
parser.add_argument('--a_dtype', type=data_type, default='float16')
parser.add_argument('--b_dtype', type=data_type, required=True)
parser.add_argument('--group_size', type=int, default=128)
parser.add_argument('--mutis_space', type=int, default=2)

args: Optional[argparse.Namespace] = None


def _benchmark_vllm(
    llm: LLM,
    seq_group_metadata_list: List[SequenceGroupMetadata],
    percentiles: Sequence[int] = (10, 50, 90),
):
    engine: LLMEngine = llm.llm_engine
    executor = engine.model_executor
    worker: Worker = executor.driver_worker
    model_runner = worker.model_runner
    assert isinstance(model_runner, ModelRunner)

    model_input = model_runner.prepare_model_input(
        seq_group_metadata_list=seq_group_metadata_list,
    )
    if args.stage == 'decode':
        assert (args.mode == 'cgraph') == model_input.attn_metadata.use_cuda_graph

    def run_func():
        model_runner.execute_model(
            model_input,
            kv_caches=worker.gpu_cache[model_input.virtual_engine]
        )

    latencies = benchmark_func(run_func, warmup=args.num_warmup, repeat=args.num_repeat, median=False, nvtx_scope='')
    latency_percentiles = [float(np.percentile(latencies, p)) for p in percentiles]
    return latency_percentiles


def benchmark_vllm_prefill(
    llm: LLM,
    num_requests: int,
    total_prompt_tokens: int,
    percentiles: Sequence[int] = (10, 50, 90),
):
    block_size: int = llm.llm_engine.cache_config.block_size
    current_block = 0
    seqs = []
    sampling_params = SamplingParams()  # default sampling params
    for i in range(num_requests):
        # num of tokens for the prompt
        prompt_tokens = total_prompt_tokens // num_requests + (i < total_prompt_tokens % num_requests)

        # generate dummy sequence data
        seq_data = SequenceData.from_seqs(
            prompt_token_ids=[0] * prompt_tokens,
            output_token_ids=[],
        )
        seqs_data = {i: seq_data}

        # allocate blocks for the prompt
        prompt_num_blocks = (prompt_tokens + block_size - 1) // block_size
        block_tables = {
            i: list(range(current_block, current_block + prompt_num_blocks))
        }
        current_block += prompt_num_blocks

        seqs.append(SequenceGroupMetadata(
            request_id=str(i),
            is_prompt=True,
            seq_data=seqs_data,
            sampling_params=sampling_params,
            block_tables=block_tables
        ))

    assert current_block <= llm.llm_engine.cache_config.num_gpu_blocks

    with nvtx.annotate('prefill request={} tokens={} of {}'.format(
        num_requests, total_prompt_tokens, llm.llm_engine.model_config.model
    )):
        return _benchmark_vllm(llm, seqs, percentiles=percentiles)


def benchmark_vllm_decode(
    llm: LLM,
    num_requests: int,
    total_context_tokens: int,
    percentiles: Sequence[int] = (10, 50, 90),
):
    block_size: int = llm.llm_engine.cache_config.block_size
    current_block = 0
    seqs = []
    sampling_params = SamplingParams()  # default sampling params
    for i in range(num_requests):
        # num of tokens for the context
        context_tokens = total_context_tokens // num_requests + (i < total_context_tokens % num_requests)

        # generate dummy sequence data
        seq_data = SequenceData.from_seqs(
            prompt_token_ids=[0] * context_tokens,
            output_token_ids=[],
        )
        seq_data.apply_delta(
            SequenceDataDelta(
                new_output_token_ids=[0],
                new_cumulative_logprob=0.0,
                new_num_computed_tokens=context_tokens,
                new_stage=SequenceStage.DECODE
            )
        )
        seqs_data = {i: seq_data}

        # allocate blocks for the prompt
        context_num_blocks = (context_tokens + 1 + block_size - 1) // block_size
        block_tables = {
            i: list(range(current_block, current_block + context_num_blocks))
        }
        current_block += context_num_blocks

        seqs.append(SequenceGroupMetadata(
            request_id=str(i),
            is_prompt=False,
            seq_data=seqs_data,
            sampling_params=sampling_params,
            block_tables=block_tables
        ))

    assert current_block <= llm.llm_engine.cache_config.num_gpu_blocks

    with nvtx.annotate('decode request={} tokens={} of {}'.format(
        num_requests, total_context_tokens, llm.llm_engine.model_config.model
    )):
        return _benchmark_vllm(llm, seqs, percentiles=percentiles)


def demo_generation(llm, tokenizer):
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = llm.generate(formatted_prompt)
    print(output)


def main():
    global args
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    device = torch.cuda.get_device_name().replace(' ', '-')
    log_path = unique_file_name(os.path.join(
        './logs',
        device,
        os.path.basename(args.model),
        '{stage}-{mode}-{backend}-{a_dtype}-{b_dtype}-bs{bs}-{tokens}{group_size}.log'.format(
            stage=args.stage,
            backend=args.backend,
            a_dtype=args.a_dtype.name,
            b_dtype=args.b_dtype.name,
            mode=args.mode,
            bs=args.bs,
            tokens=args.tokens,
            group_size='-g{}'.format(args.group_size) if args.group_size != -1 else ''
        )
    ))

    rows = []

    print('logging to {}'.format(log_path))
    latency = None
    try:
        with open(log_path, 'w') as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                if args.stage == 'decode':
                    m_size = args.bs
                else:
                    m_size = args.bs * args.tokens
                mutis.option.space(args.mutis_space)
                mutis.set_benchmark_mode(True)
                mutis.extension.vllm.quantization.set_mutis_config(
                    backend=args.backend,
                    m_size=m_size,
                    a_dtype=args.a_dtype,
                    b_dtype=args.b_dtype,
                    group_size=args.group_size
                )
                llm = LLM(
                    model=args.model,
                    tokenizer=args.tokenizer,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    enforce_eager=args.mode == 'eager',
                    dtype=mutis.dtype_to_torch(args.a_dtype),
                    quantization='mutis',
                    load_format='dummy',
                    disable_log_stats=True,
                    max_model_len=args.tokens + 10,
                )
                if args.stage == 'prefill':
                    num_requests = args.bs
                    num_tokens = args.tokens
                    latency_percentiles = benchmark_vllm_prefill(
                        llm=llm,
                        num_requests=num_requests,
                        total_prompt_tokens=num_tokens,
                    )
                    rows.append([
                        device,
                        args.stage,
                        args.mode,
                        args.model,
                        args.backend,
                        args.a_dtype.name,
                        args.b_dtype.name,
                        args.bs,
                        args.tokens,
                        args.group_size,
                        latency_percentiles[0],
                        latency_percentiles[1],
                        latency_percentiles[2],
                    ])
                    latency = latency_percentiles[1]
                elif args.stage == 'decode':
                    rows = []
                    num_requests = args.bs
                    num_tokens = args.tokens
                    latency_percentiles = benchmark_vllm_decode(
                        llm=llm,
                        num_requests=num_requests,
                        total_context_tokens=num_tokens,
                    )
                    rows.append([
                        device,
                        args.stage,
                        args.mode,
                        args.model,
                        args.backend,
                        args.a_dtype.name,
                        args.b_dtype.name,
                        args.bs,
                        args.tokens,
                        args.group_size,
                        latency_percentiles[0],
                        latency_percentiles[1],
                        latency_percentiles[2],
                    ])
                    latency = latency_percentiles[1]
                else:
                    help_message = parser.format_help()
                    exception_message = f"Invalid stage: {args.stage}\n{help_message}"
                    exception_message += '\nargs: {}'.format(args)
                    latency = float('NaN')
                    raise ValueError(exception_message)
    except Exception as e:
        print(traceback.format_exc())
        print()
        print(float("NaN"), end='')
    else:
        print()
        print(latency, end='')


def load_cache():
    """Load the cache from the CSV file."""
    if data_path.exists():
        with open(data_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    # skip the header
                    continue
                parts = line.strip().split(',')
                if len(parts) != len(headers):
                    continue
                device, stage, backend, model, bs, tokens, a_dtype, b_dtype, group_size, latency = parts
                key = (
                    device,
                    stage,
                    backend,
                    model,
                    int(bs),
                    int(tokens),
                    a_dtype,
                    b_dtype,
                    int(group_size)
                )
                cache[key] = float(latency)
        if len(cache) > 0:
            print(f"Cache loaded with {len(cache)} entries.")
    else:
        # write the headers to the file if it does not exist
        with open(data_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(headers)


load_cache()


def bench(
    gpu_memory_utilization=0.9,
    mode='cgraph',
    model='meta-llama/Meta-Llama-3-8B-Instruct',
    stage='decode',
    backend='torch-f16',
    a_dtype='float16',
    b_dtype='float16',
    group_size=128,
    num_warmup=1,
    num_repeat=50,
    bs=1,
    tokens=1024,
    mutis_space=2,
):
    device = torch.cuda.get_device_name().replace(' ', '-')
    key = (
        device,
        stage,
        backend,
        model,
        bs,
        tokens,
        a_dtype,
        b_dtype,
        group_size,
    )
    if key in cache:
        return cache[key]
    args_string = '''
        --gpu_memory_utilization {}
        --mode {}
        --model {}
        --stage {}
        --backend {}
        --a_dtype {}
        --b_dtype {}
        --group_size {}
        --num_warmup {}
        --num_repeat {}
        --bs {}
        --tokens {}
        --mutis_space {}
    '''.format(
        gpu_memory_utilization,
        mode,
        model,
        stage,
        backend,
        a_dtype,
        b_dtype,
        group_size,
        num_warmup,
        num_repeat,
        bs,
        tokens,
        mutis_space
    )
    args_string = args_string.replace('\n', ' ')
    while '  ' in args_string:
        args_string = args_string.replace('  ', ' ')

    t1 = time.time()
    # the path to the current script
    script_path = Path(__file__).resolve()
    command = '{} {} {}'.format(sys.executable, script_path, args_string)
    print('Bench with ', command)
    ret = subprocess.run(command.split(), check=False, capture_output=True)
    t2 = time.time()
    print('Spend time: {:7.2f} seconds\n'.format(t2 - t1))

    if ret.returncode == 0:
        print(ret.stderr.decode())
        print(ret.stdout.decode())
        latency = float(ret.stdout.decode().split('\n')[-1])
    else:
        print(ret.stderr.decode())
        print(ret.stdout.decode())
        latency = float('nan')

    print('Latency: {} (ms)'.format(latency))

    with open(data_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([
            device,
            stage,
            backend,
            model,
            bs,
            tokens,
            a_dtype,
            b_dtype,
            group_size,
            latency
        ])

    return latency


if __name__ == '__main__':
    main()
