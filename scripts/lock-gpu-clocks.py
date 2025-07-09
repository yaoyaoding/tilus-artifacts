#!/bin/python3
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Lock GPU clocks')
parser.add_argument('--idx', '-i', type=int, default=0, help='GPU index')
parser.add_argument('--mode', type=str, choices=['max', 'base'], default='max', help='Lock to max or base clock')
parser.add_argument('--reset', action='store_true', help='Reset GPU clocks')


def run_command(command) -> str:
    command = command.split()
    print('> {}'.format(" ".join(command)))
    ret = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    if ret.returncode:
        raise Exception(f'Error running command: {command}\n{ret.stdout}\n{ret.stderr}')
    return ret.stdout


def get_base_sm_clock(idx: int = 0) -> int:
    output = run_command(f'nvidia-smi base-clocks -i {idx}')
    lines = output.split('\n')
    ret = int(lines[1].split()[1][:-3])
    return ret


def get_max_sm_clock(idx: int = 0) -> int:
    output = run_command(f'nvidia-smi --query-gpu=clocks.max.sm -i {idx} --format=csv,noheader,nounits')
    return int(output)


def get_max_mem_clock(idx: int = 0) -> int:
    output = run_command(f'nvidia-smi --query-gpu=clocks.max.mem -i {idx} --format=csv,noheader,nounits')
    return int(output)


def get_current_clock(idx: int = 0) -> int:
    output = run_command(f'nvidia-smi --query-gpu=clocks.current.sm -i {idx} --format=csv,noheader,nounits')
    return int(output)


def get_persistence_mode(idx: int = 0) -> int:
    output = run_command(f'nvidia-smi -i {idx} --query-gpu=persistence_mode --format=csv,noheader,nounits')
    output = output.strip()
    return 1 if output == 'Enabled' else 0


def set_persistence_mode(idx: int, mode: int):
    assert mode in [0, 1]
    run_command(f'sudo nvidia-smi -i {idx} -pm {mode}')
    print(f'[{idx}] Set persistence mode to {mode}', )


def lock_sm_clock(idx: int, clock: int):
    run_command(f'sudo nvidia-smi -i {idx} -lgc {clock},{clock}')
    print(f'[{idx}] Locked gpu sm clock to {clock} MHz')


def lock_mem_clock(idx: int, clock: int):
    run_command(f'sudo nvidia-smi -i {idx} -lmc {clock},{clock}')
    print(f'[{idx}] Locked gpu memory clock to {clock} MHz')


def reset_clock(idx: int):
    run_command(f'sudo nvidia-smi -i {idx} -rgc')
    print(f'[{idx}] Reset gpu clock')


def main():
    args = parser.parse_args()

    idx = args.idx
    if args.reset:
        reset_clock(idx)
    else:
        persistence_mode = get_persistence_mode(idx)
        if persistence_mode != 1:
            set_persistence_mode(idx, 1)
        else:
            print(f'[{idx}] Persistence mode is already enabled')

        if args.mode == 'max':
            lock_sm_clock(idx, get_max_sm_clock(idx))
            lock_mem_clock(idx, get_max_mem_clock(idx))
        elif args.mode == 'base':
            lock_sm_clock(idx, get_base_sm_clock(idx))
            lock_mem_clock(idx, get_max_mem_clock(idx))
        else:
            raise ValueError('Invalid mode', args.mode)


if __name__ == '__main__':
    main()
