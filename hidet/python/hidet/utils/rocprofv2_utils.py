import os
import sys
import glob
import pickle
import subprocess
import inspect
import tempfile
import argparse


_rocprofv2_path = '/usr/bin/rocprofv2'
_rocprofv2_command_template = """
    {rocprofv2_path} -d {trace_dir} --hip-trace --kernel-trace --plugin perfetto {python_executable} {python_script} {args}
"""


class PerfettoTrace:
    def __init__(self, trace_path: str):
        self.trace_path: str = trace_path

    def visualize(self, open_browser=True):
        from hidet.utils.perfetto_utils import open_trace

        open_trace(self.trace_path, open_browser=open_browser, origin='https://ui.perfetto.dev')


def _rocprofv2_run_func(script_path, func_name, args_pickled_path):
    import hidet

    with open(args_pickled_path, 'rb') as f:
        args, kwargs = pickle.load(f)

    # remove the dir path of the current script from sys.path to avoid module overriding
    sys.path = [path for path in sys.path if not path.startswith(os.path.dirname(__file__))]

    try:
        sys.path.append(os.path.dirname(script_path))
        module = __import__(os.path.basename(script_path)[:-3])
    except Exception as e:
        raise RuntimeError('Can not import the python script: {}'.format(script_path)) from e

    if not hasattr(module, func_name):
        raise RuntimeError('Can not find the function "{}" in {}'.format(func_name, script_path))

    func = getattr(module, func_name)

    try:
        hidet.hip.synchronize()
        func(*args, **kwargs)
        hidet.hip.synchronize()
    except Exception as e:
        raise RuntimeError('Error when running the function "{}"'.format(func_name)) from e


def rocprofv2_run(func, *args, **kwargs) -> PerfettoTrace:
    # get the python script path and function name
    script_path: str = inspect.getfile(func)
    func_name: str = func.__name__

    # trace path
    trace_dir_template: str = os.path.join(os.path.dirname(script_path), 'rocprofv2_traces/{}')
    idx = 0
    while os.path.exists(trace_dir_template.format(idx)):
        idx += 1
    trace_dir = trace_dir_template.format(idx)
    os.makedirs(trace_dir)

    # dump args
    args_path: str = tempfile.mktemp() + '.pkl'
    with open(args_path, 'wb') as f:
        pickle.dump((args, kwargs), f)

    status = subprocess.run(
        _rocprofv2_command_template.format(
            rocprofv2_path=_rocprofv2_path,
            trace_dir=trace_dir,
            python_executable=sys.executable,
            python_script=__file__,
            args='{} {} {}'.format(script_path, func_name, args_path),
        ),
        shell=True,
    )

    if status.returncode != 0:
        raise RuntimeError('Error when running rocprofv2.')

    # match the trace file under trace_dir with name pattern *.pftrace
    trace_files = glob.glob(os.path.join(trace_dir, '*.pftrace'))
    assert len(trace_files) == 1
    trace_file = trace_files[0]

    return PerfettoTrace(trace_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('script_path', type=str)
    parser.add_argument('func', type=str)
    parser.add_argument('args', type=str)
    args = parser.parse_args()
    _rocprofv2_run_func(args.script_path, args.func, args.args)


if __name__ == '__main__':
    main()
