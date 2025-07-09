import os
import time
import argparse
import sys
import subprocess

parser = argparse.ArgumentParser(
    "entry.py",
    description="Run experiments for reproducing figures in the paper."
)
parser.add_argument(
    "--figure",
    choices=["9", "10", "11", "13", "all"],
    required=False,
    help="The experiment to reproduce"
)
parser.add_argument(
    '--no-cache',
    action='store_true',
    help="Disable the pre-compiled cache. All kernels in Ladder/Triton/Tilus will be recompiled on the current hardware."
)

def run_experiment(script_path, cache_dir: str, result_dir: str = None):
    command = '{python} {script}'.format(
        python=sys.executable,
        script=script_path
    )
    print(f"Running experiment: {command}")
    t1 = time.time()
    env = dict(os.environ)
    env['TILUS_ARTIFACT_CACHE_DIR'] = cache_dir
    env['TILUS_ARTIFACT_RESULTS_DIR'] = result_dir
    result = subprocess.run(command, shell=True, check=True, env=env)
    t2 = time.time()
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")
    print(f"Completed in {t2 - t1:.2f} seconds")


def main():
    args = parser.parse_args()

    if args.no_cache:
        cache_dir = './cache'
        results_dir = './results'
    else:
        cache_dir = './precompiled-cache'
        results_dir = './precompiled-results'

    if args.figure == 'all' or args.figure is None:
        scripts = [
            './artifacts/figure9.py',
            './artifacts/figure10.py',
            './artifacts/figure11.py',
            './artifacts/figure13.py'
        ]
    else:
        scripts = [f'./artifacts/figure{args.figure}.py']
    for script in scripts:
        script = os.path.abspath(script)
        run_experiment(script, cache_dir, results_dir)


if __name__ == "__main__":
    main()

