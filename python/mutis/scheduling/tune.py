# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union, Sequence, TypeVar, Any, Dict, List, Iterable, cast, Optional
import itertools
from mutis.utils import prod

Choice = TypeVar('Choice')


class ScheduleError(Exception):
    pass


class TuningSpace:
    MAX_SPACE_SIZE = 1200000

    def __init__(self):
        self.spaces: Dict[int, Dict[str, Any]] = {}
        self.existing_names: List[str] = []

    def iterate_space(self, level: int):
        # when given level is not defined, down to lower level
        while level > 0 and level not in self.spaces:
            level -= 1
        if level == 0 and level not in self.spaces:
            yield {}
            return

        sub_keys = list(self.spaces[level].keys())
        sub_spaces = list(self.spaces[level].values())
        space_size = prod([len(s) for s in sub_spaces])
        if space_size > self.MAX_SPACE_SIZE:
            raise ValueError(
                f'The search space has {space_size} schedules, '
                f'which is larger than the predefined limit {self.MAX_SPACE_SIZE}. '
                f'Please consider to reduce the search space.'
            )
        for values in itertools.product(*sub_spaces):
            kwargs = {}
            for key, value in zip(sub_keys, values):
                if ',' in key:
                    for name, v in zip(key.split(','), value):
                        kwargs[name] = v
                else:
                    kwargs[key] = value
            yield kwargs

    def add_sub_space(self, level: int, name_choice_dict: Dict[str, Sequence[Union[Choice, Sequence[Choice]]]]):
        if level in self.spaces:
            raise ValueError(f'Level {level} is already defined.')

        self.spaces[level] = {}
        for names, choices in name_choice_dict.items():
            choices = list(choices)
            names = [name.strip() for name in names.split(',')]
            for name in names:
                if name in self.existing_names:
                    raise ValueError(f'Subspace {name} is already added.')
            if len(names) > 1:
                for choice in choices:
                    if not hasattr(choice, '__len__'):
                        raise ValueError(f'When multiple names are given, choices must be iterable, got {type(choice)}')
                    if len(choice) != len(names):
                        raise ValueError(
                            f'Number of choices {len(choice)} does not match number of names {len(names)}.'
                        )
            self.spaces[level][",".join(names)] = choices


def _space(level: int, names2values: Dict[str, Sequence[Union[Choice, Sequence[Choice]]]]):
    assert all(isinstance(value, Sequence) for value in names2values.values())

    def wrapper(func):
        if not hasattr(func, 'tuning_space'):
            # attach tuning space when the first time of this function is called
            setattr(func, 'tuning_space', TuningSpace())
        tuning_space: TuningSpace = getattr(func, 'tuning_space')
        tuning_space.add_sub_space(level, names2values)
        return func

    return wrapper


space = cast(Any, _space)  # a trick to avoid ide show inlay hints for parameters, which makes the line too long


def extract(template_func, space: Optional[int] = None, args=(), kwargs=None) -> List[Any]:
    if space is None:
        from mutis.scheduling.scheduler import get_current_space

        space = get_current_space()
    assert space in [0, 1, 2], 'Only support 0, 1, 2 levels of space.'

    # get ir modules to tune
    if hasattr(template_func, 'tuning_space'):
        tuning_space: TuningSpace = getattr(template_func, 'tuning_space')
        # iterate space and instantiate schedules into tensor programs
        kwargs_list = list(tuning_space.iterate_space(space))
    else:
        raise ValueError(
            'No tuning space is attached to the template function.\n'
            'Please use @tune.space to decorate the template function to define the search space.'
        )

    if kwargs is None:
        kwargs = dict()

    results = []
    for tuning_kwargs in kwargs_list:
        merged_kwargs = dict(kwargs)
        merged_kwargs.update(tuning_kwargs)
        try:
            results.append(template_func(*args, **merged_kwargs))
        except ScheduleError as e:
            # the schedule is invalid, skip it
            continue
    return results


def check(condition: bool, message: str = ""):
    if not condition:
        raise ScheduleError(message)
