from __future__ import annotations
from typing import Dict, Any, List, Optional, Callable, Iterable
import os
from hidet.utils import git_utils


class OptionRegistry:
    registered_options: Dict[str, OptionRegistry] = {}

    def __init__(
        self,
        name: str,
        type_hint: str,
        description: str,
        default_value: Any,
        normalizer: Callable[[Any], Any],
        choices: Optional[Iterable[Any]] = None,
        checker: Optional[Callable[[Any], bool]] = None,
        env_name: Optional[str] = None,
    ):
        self.name = name
        self.type_hint = type_hint
        self.description = description
        self.default_value = default_value
        self.normalizer = normalizer
        self.choices = choices
        self.checker = checker
        self.env_name = env_name


def register_option(
    name: str,
    type_hint: str,
    description: str,
    default_value: Any,
    normalizer: Callable[[Any], Any],
    choices: Optional[Iterable[Any]] = None,
    checker: Optional[Callable[[Any], bool]] = None,
    env_name: Optional[str] = None,
):
    registered_options = OptionRegistry.registered_options
    if name in registered_options:
        raise KeyError(f'Option {name} has already been registered.')
    registered_options[name] = OptionRegistry(
        name=name,
        type_hint=type_hint,
        description=description,
        default_value=default_value,
        normalizer=normalizer,
        choices=choices,
        checker=checker,
        env_name=env_name,
    )


def register_mutis_options():
    register_option(
        name='space',
        type_hint='int',
        description='The kernel search space, candidates: 0, 1, and 2.',
        default_value=2,
        choices=[0, 1, 2],
        normalizer=int,
        env_name='MUTIS_SPACE',
    )
    register_option(
        name='cache_dir',
        type_hint='path',
        description='The directory to store the cache.',
        default_value=os.path.abspath('./cache'),
        normalizer=os.path.abspath,
    )
    register_option(
        name='tuning.warmup',
        type_hint='int',
        description='The number of warmup runs during tuning candidates.',
        default_value=5,
        checker=lambda x: x > 0,
        normalizer=int,
    )
    register_option(
        name='tuning.repeat',
        type_hint='int',
        description='The number of repeat runs during tuning candidates.',
        default_value=50,
        checker=lambda x: x > 0,
        normalizer=int,
    )
    register_option(
        name='tuning.maximum_warmup_time',
        type_hint='float',
        description='The maximum time (in milliseconds) to warmup a single candidate.',
        default_value=10.0,  # ms
        checker=lambda x: x > 0,
        normalizer=float,
    )
    register_option(
        name='tuning.maximum_repeat_time',
        type_hint='float',
        description='The maximum time (in milliseconds) to benchmark a single candidate.',
        default_value=50.0,  # ms
        checker=lambda x: x > 0,
        normalizer=float,
    )
    register_option(
        name='build_workers',
        type_hint='optional[int]',
        description='The number of workers to build the kernel candidates in parallel. None indicates number of cpu cores',
        default_value=None,
        normalizer=lambda x: None if x is None else int(x),
        checker=lambda x: x is None or x > 0,
    )
    register_option(
        name='debug.dump_ir',
        type_hint='bool',
        description='Whether to dump the IR after each pass.',
        default_value=False,
        normalizer=bool,
    )
    register_option(
        name='vllm.quantization.verify',
        type_hint='bool',
        description='Whether to verify the correctness of the quantization when the original weight is available.',
        default_value=True,
        normalizer=bool,
    )


register_mutis_options()


class OptionContext:
    """
    The option context.
    """

    stack: List[OptionContext] = []

    def __init__(self):
        self.options: Dict[str, Any] = {}

    def __str__(self):
        pass

    def __enter__(self):
        """
        Enter the option context.

        Returns
        -------
        ret: OptionContext
            The option context itself.
        """
        OptionContext.stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the option context.
        """
        OptionContext.stack.pop()

    @staticmethod
    def current() -> OptionContext:
        """
        Get the current option context.

        Returns
        -------
        ret: OptionContext
            The current option context.
        """
        return OptionContext.stack[-1]

    def set_option(self, name: str, value: Any):
        """
        Set the value of an option in the self option context.

        Parameters
        ----------
        name: str
            The name of the option.

        value: Any
            The value of the option.
        """
        if name not in OptionRegistry.registered_options:
            raise KeyError(f'Option {name} has not been registered.')
        registry = OptionRegistry.registered_options[name]
        if registry.normalizer is not None:
            value = registry.normalizer(value)
        if registry.checker is not None:
            if not registry.checker(value):
                raise ValueError(f'Invalid value for option {name}: {value}')
        if registry.choices is not None:
            if value not in registry.choices:
                raise ValueError(f'Invalid value for option {name}: {value}, choices {registry.choices}')
        self.options[name] = value

    def get_option(self, name: str) -> Any:
        """
        Get the value of an option in the self option context.

        Parameters
        ----------
        name: str
            The name of the option.

        Returns
        -------
        ret: Any
            The value of the option.
        """
        for ctx in reversed(OptionContext.stack):
            if name in ctx.options:
                return ctx.options[name]
        if name not in OptionRegistry.registered_options:
            raise KeyError(f'Option {name} has not been registered.')
        registry = OptionRegistry.registered_options[name]
        return registry.default_value


OptionContext.stack.append(OptionContext())


def _load_from_env(ctx: OptionContext):
    for name, registry in OptionRegistry.registered_options.items():
        if registry.env_name and registry.env_name in os.environ:
            ctx.set_option(name, os.environ[registry.env_name])


_load_from_env(OptionContext.current())


def current_context() -> OptionContext:
    """
    Get the current option context.

    To get the value of an option in the current context:

    .. code-block:: python

        ctx = hidet.option.current_context()
        cache_dir: str = ctx.get_option('cache_dir')
        cache_operator: bool = ctx.get_option('cache_operator')
        ...

    Returns
    -------
    ctx: OptionContext
        The current option context.
    """
    return OptionContext.current()


def context() -> OptionContext:
    """
    Create a new option context.

    To set options in the new context, use the ``with`` statement:

    .. code-block:: python

        with hidet.option.context() as ctx:
            hidet.option.cache_dir('./new_cache_dir')               # set predefined option
            hidet.option.set_option('other_option', 'other_value')  # set a custom option
            ...

    Returns
    -------
    ctx: OptionContext
        The new option context.
    """
    return OptionContext()


def set_option(name: str, value: Any):
    """
    Set the value of an option in current option context.

    The option must be registered before setting via :py:func:`hidet.option.register_option`.

    Parameters
    ----------
    name: str
        The name of the option.
    value: Any
        The value of the option.
    """
    OptionContext.current().set_option(name, value)


def get_option(name: str) -> Any:
    """
    Get the value of an option in current option context.

    Parameters
    ----------
    name: str
        The name of the option.

    Returns
    -------
    ret: Any
        The value of the option.
    """
    return OptionContext.current().get_option(name)


def space(n: int):
    """
    Set the default kernel search space that will be used for all mutis jit kernels.

    Parameters
    ----------
    n: int
        The kernel search space, candidates: 0, 1, and 2. 0 is the smallest space, 2 is the largest space.
    """
    OptionContext.current().set_option('space', n)


def cache_dir(new_dir: str):
    """
    Set the directory to store the cache.

    The default cache directory:

    - If the hidet code is in a git repo, the cache will be stored in the repo root:
      ``hidet-repo/.hidet_cache``.
    - Otherwise, the cache will be stored in the user home directory: ``~/.hidet/cache``.

    Parameters
    ----------
    new_dir: str
        The new directory to store the cache.
    """
    OptionContext.current().set_option('cache_dir', new_dir)


def build_workers(num_workers: Optional[int] = None):
    """
    Set the number of workers to build the kernel candidates in parallel.

    Parameters
    ----------
    num_workers: int
        The number of workers. None indicates number of cpu cores.
    """
    OptionContext.current().set_option('build_workers', num_workers)


class tuning:
    """
    The options in this option group are used to control the tuning process.

    Mutis uses (warmup, repeat, maximum_time) to determine how many times to run the tuned candidates.

    The default values are (1, 10, 10.0), respectively, which means that the tuning process will run each candidate once
    for warmup, then run each candidate 10 times for benchmarking, if the time of a single run exceeds 1 ms, we will
    reduce the number of repeat runs to ensure that the total time of all runs does not exceed 10 ms.

    In short, we will do the following for each candidate to measure the performance:

    ```
    def measure_latency(candidate):
        # pseudocode of tuning process, we will use cuda events to measure the time instead of time.time()
        warmup, repeat, maximum_time = 1, 10, 10.0

        run candidate once   # warmup

        t1 = time.time()
        run candidate once
        t2 = time.time()
        estimated_single_run = t2 - t1

        estimated_repeat = int(ceil(maximum_time / (t2 - t1)))
        repeat = min(estimated_repeat, repeat)

        results = [t2 - t1]
        results.extend([time for a single run of candidate for _ in range(repeat - 1)])

        return results
    ```


    """

    @staticmethod
    def warmup(warmup: int = 10):
        """
        Set the number of warmup runs during tuning candidates.

        Parameters
        ----------
        warmup: int
            The number of warmup runs.
        """
        OptionContext.current().set_option('tuning.warmup', warmup)

    @staticmethod
    def repeat(repeat: int = 100):
        """
        Set the number of repeat runs during tuning candidates.

        Parameters
        ----------
        repeat: int
            The number of repeat runs.
        """
        OptionContext.current().set_option('tuning.repeat', repeat)

    @staticmethod
    def maximum_warmup_time(time_ms: float = 20.0):
        """
        Set the maximum time (in milliseconds) to warm up a single candidate.

        Parameters
        ----------
        time_ms: float
            The maximum time in milliseconds.
        """
        OptionContext.current().set_option('tuning.maximum_warmup_time', time_ms)

    @staticmethod
    def maximum_repeat_time(time_ms: float = 20.0):
        """
        Set the maximum time (in milliseconds) to benchmark a single candidate.

        Parameters
        ----------
        time_ms: float
            The maximum time in milliseconds.
        """
        OptionContext.current().set_option('tuning.maximum_repeat_time', time_ms)


class vllm:
    class quantization:
        @staticmethod
        def verify(enabled: bool = True):
            """
            Set whether to verify the correctness of the quantization when the original weight is available.
            """
            OptionContext.current().set_option('vllm.quantization.verify', enabled)

class debug:
    @staticmethod
    def dump_ir(enabled = True):
        OptionContext.current().set_option('debug.dump_ir', enabled)
