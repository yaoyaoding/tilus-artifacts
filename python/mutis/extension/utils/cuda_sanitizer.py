# pylint: disable=subprocess-run-check
import os
import pickle
import argparse
import importlib
import sys
import subprocess

_sanitizer_path: str = '/usr/local/cuda/bin/compute-sanitizer'
_sanitizer_template = """
    {sanitizer_path}
    --show-backtrace=device
    --print-limit 20
    {python_executable} {python_script} {args} > {report_path}
""".replace(
    '\n', ' '
).strip()


def _sanitizer_run_func(script_path, func_name, args_pickled_path):
    with open(args_pickled_path, 'rb') as f:
        args, kwargs = pickle.load(f)

    # remove the dir path of the current script from sys.path to avoid module overriding
    sys.path = [path for path in sys.path if not path.startswith(os.path.dirname(__file__))]

    try:
        sys.path.append(os.path.dirname(script_path))
        module = importlib.import_module(os.path.basename(script_path)[:-3])
    except Exception as e:
        raise RuntimeError('Can not import the python script: {}'.format(script_path)) from e

    if not hasattr(module, func_name):
        raise RuntimeError('Can not find the function "{}" in {}'.format(func_name, script_path))

    func = getattr(module, func_name)

    try:
        func(*args, **kwargs)
    except Exception as e:
        raise RuntimeError('Error when running the function "{}"'.format(func_name)) from e


def sanitizer_set_path(sanitizer_path: str):
    global _sanitizer_path
    _sanitizer_path = sanitizer_path


def sanitizer_run(func, *args, **kwargs):
    import inspect
    import tempfile

    # get the python script path and function name
    script_path: str = inspect.getfile(func)
    func_name: str = func.__name__

    # report path
    report_path_template: str = os.path.join(os.path.dirname(script_path), 'sanitizer-reports/report{}.txt')
    idx = 0
    while os.path.exists(report_path_template.format(idx)):
        idx += 1
    report_path = report_path_template.format(idx)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # dump args
    args_path: str = tempfile.mktemp() + '.pkl'
    with open(args_path, 'wb') as f:
        pickle.dump((args, kwargs), f)

    command = _sanitizer_template.format(
        sanitizer_path=_sanitizer_path,
        report_path=report_path,
        python_executable=sys.executable,
        python_script=__file__,
        args='{} {} {}'.format(script_path, func_name, args_path),
    )
    command = " ".join(command.split())
    print('Running command: ')
    print(command)
    subprocess.run(command, shell=True)
    with open(report_path, 'r') as f:
        print(f.read())
    print('Sanitizer report is saved in: {}'.format(report_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('script_path', type=str)
    parser.add_argument('func', type=str)
    parser.add_argument('args', type=str)
    args = parser.parse_args()
    _sanitizer_run_func(args.script_path, args.func, args.args)


if __name__ == '__main__':
    main()
