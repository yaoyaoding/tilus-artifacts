from __future__ import annotations as _

from typing import List, Any, Dict, Tuple, Callable, Optional, Sequence, Union, Hashable, Set
import collections
import hashlib
import inspect
import json
import math
import logging
import os
import pprint
import cProfile
import shelve
import shutil
import traceback
from weakref import WeakSet
import gc
import types
from xml.sax.handler import property_dom_node

import filelock
import tabulate
from tqdm import tqdm

import torch
import hidet.cuda
import mutis.option
import hidet.backend.build
import hidet
from hidet.drivers import build_ir_module
from hidet.drivers.utils import lazy_initialize_cuda
from hidet.ir.dtypes import int32, float32, boolean
from hidet.ir.expr import Var, var
from hidet.ir.module import IRModule
from hidet.ir.type import BaseType, DataType
from hidet.runtime import CompiledModule, load_compiled_module
from mutis.backends.codegen import generate_virtual_machine_program
from mutis.exceptions import CompilationFailedError
from mutis.extension.transforms import apply_mutis_specific_transforms
from mutis.ir.graph import Graph, GraphContext, ParamAttrs
from mutis.ir.schedule import Schedule
from mutis.jit.types import ConstantType, WeightType, OptionalType, ConstantTypeMaker
from mutis.scheduling import generate_schedules
from mutis.target import get_current_target
from mutis.utils import benchmark_func
from mutis.utils import parallel_imap, serial_imap, normalize_filename
from mutis.vm.backends.codegen import generate_ir_module
from mutis.vm.ir.program import VirtualMachineProgram

logger = logging.getLogger(__name__)
__all_jit_functions: WeakSet[JitFunction] = WeakSet()

pr = cProfile.Profile()
pr.disable()


class JitError(Exception):
    pass


def normalize_const_arg_repr(const_arg) -> str:
    if isinstance(const_arg, torch.dtype):
        const_arg = hidet.torch.utils.dtype_from_torch(const_arg)
    if isinstance(const_arg, DataType):
        return const_arg.short_name
    return str(const_arg)


class JitOptions:
    def __init__(
        self,
        space: Optional[int],
        kind: str,
        use_single_weight_transform: bool,
        tuning_trigger_params: Optional[List[str]],
        tuning_trigger_expression: Optional[Callable[[List[Any]], Hashable]],
        disable_cache: bool,
        disable_parallel_build: bool,
        debug_dump_ir: bool,
        debug_use_schedules: Optional[Sequence[int]],
        debug_print_vm_inst_output: Union[bool, Dict[str, int]],
        debug_enable_devise_assertion: bool,
        verbose: bool,
    ):
        self._space: Optional[int] = space
        self.kind: str = kind
        self.use_single_weight_transform: bool = use_single_weight_transform
        self.tuning_trigger_params: Optional[List[str]] = tuning_trigger_params
        self.tuning_trigger_expression: Optional[Callable[[Sequence[Any]], Hashable]] = tuning_trigger_expression
        self.disable_cache: bool = disable_cache
        self.disable_parallel_build: bool = disable_parallel_build
        self.debug_dump_ir: bool = debug_dump_ir
        self.debug_use_schedules: Optional[Sequence[int]] = debug_use_schedules
        self.debug_print_vm_inst_output: Union[bool, Dict[str, int]] = debug_print_vm_inst_output
        self.debug_enable_devise_assertion: bool = debug_enable_devise_assertion
        self.verbose: bool = verbose

    @property
    def space(self) -> int:
        return self._space if self._space is not None else mutis.option.get_option('space')


class JitBaseTemplate:
    def __init__(self, options: JitOptions, fn: types.FunctionType):
        self.options: JitOptions = options

        # target function
        self.fn: types.FunctionType = fn

        # function analysis results
        self.name: str = fn.__name__
        self.num_params: Optional[int] = None
        self.const_param_indices: List[int] = []
        self.kernel_param_indices: List[int] = []
        self.weight_param_indices: List[int] = []
        self.optional_param_indices: List[int] = []
        self.param_names: List[str] = []
        self.param_types: List[Union[BaseType, Any]] = []
        self.annotations: List[Any] = []

        self.init()

    def init(self):
        sig: inspect.Signature = inspect.signature(self.fn)

        self.num_params = len(sig.parameters)

        for idx, param in enumerate(sig.parameters.values()):
            assert isinstance(param, inspect.Parameter)
            if param.kind != param.POSITIONAL_OR_KEYWORD:
                raise ValueError('Can only support normal parameter kind, got {}'.format(param.kind))
            if param.default != param.empty:
                raise ValueError('Can not set default value')
            self.param_names.append(param.name)

            annotation = param.annotation
            self.annotations.append(annotation)
            type_remap = {int: int32, float: float32, bool: boolean}
            if annotation in type_remap:
                annotation = type_remap[annotation]
            if isinstance(annotation, OptionalType):
                annotation = annotation.base_type
                self.optional_param_indices.append(idx)

            if isinstance(annotation, BaseType):
                self.param_types.append(annotation)
                self.kernel_param_indices.append(idx)
            elif isinstance(annotation, (ConstantType, ConstantTypeMaker)):
                if isinstance(annotation, ConstantType):
                    self.param_types.append(annotation.base_type)
                else:
                    self.param_types.append(None)
                self.const_param_indices.append(idx)
                if idx in self.optional_param_indices:
                    raise TypeError('Constant parameter can not be optional, please use `constant[Optional[...]]`')
            elif isinstance(annotation, WeightType):
                self.param_types.append(annotation.base_pointer_type)
                self.kernel_param_indices.append(idx)
                self.weight_param_indices.append(idx)
            else:
                raise ValueError('Can not recognize type hint: {}'.format(repr(param.annotation)))

    def build_vm_program(self, vm_program: VirtualMachineProgram, vm_cache_dir: str) -> bool:
        os.makedirs(vm_cache_dir, exist_ok=True)

        # optimize the virtual machine program
        vm_program = self.optimize_vm_program(vm_program, vm_cache_dir)

        # lower to ir module
        module_dir: str = os.path.join(vm_cache_dir, 'module')
        try:
            ir_module: IRModule = generate_ir_module(vm_program)
        except CompilationFailedError:
            os.makedirs(module_dir, exist_ok=True)
            with open(os.path.join(module_dir, 'module_generation_error.txt'), 'w') as f:
                f.write(traceback.format_exc())
            return False

        # build ir module to binary
        with hidet.option.context():
            if self.options.debug_dump_ir:
                hidet.option.save_lower_ir()

            # apply mutis-specific ir module transforms
            ir_module = self.apply_ir_module_transforms(ir_module, module_dir)

            try:
                build_ir_module(
                    ir_module, output_dir=module_dir, target='cuda' if get_current_target().is_nvgpu() else 'hip'
                )
            except hidet.backend.build.CompilationFailed:
                return False
        return True

    def optimize_vm_program(self, vm_program: VirtualMachineProgram, cache_dir: str) -> VirtualMachineProgram:
        from mutis.vm.transforms.bound_aware_simplify import bound_aware_simplify_pass
        from mutis.vm.transforms.inject_print_instruction import inject_print_instruction_pass
        from mutis.vm.transforms import optimize_vm_program

        vm_passes = [bound_aware_simplify_pass()]
        if self.options.debug_print_vm_inst_output:
            # inject print instructions to the virtual machine program for debugging
            if isinstance(self.options.debug_print_vm_inst_output, bool):
                block_to_print = {}  # by default, print virtual block with all indices = 0
                instructions_to_print: Optional[List[str]] = None
            else:
                configs = self.options.debug_print_vm_inst_output.copy()
                instructions_to_print = None if 'instructions' not in configs else configs.pop('instructions')
                block_to_print = None if 'block' not in configs else configs.pop('block')
            vm_passes.append(inject_print_instruction_pass(block_to_print, instructions_to_print))
        if self.options.debug_dump_ir == True or self.options.debug_dump_ir is None and mutis.option.get_option('debug.dump_ir'):
            vm_dump_dir = os.path.join(cache_dir, 'vm')
            os.makedirs(vm_dump_dir, exist_ok=True)
        else:
            vm_dump_dir = None

        vm_program = optimize_vm_program(vm_program, transforms=vm_passes, dump_dir=vm_dump_dir)

        with open(os.path.join(cache_dir, 'vm_optimized.txt'), 'w') as f:
            f.write(str(vm_program))

        with open(os.path.join(cache_dir, 'weight_transform.txt'), 'w') as f:
            from mutis.vm.ir.printer import VirtualMachinePrinter

            printer = VirtualMachinePrinter()
            for param, transform in vm_program.weight_transforms.items():
                f.write(f'{printer(param)}: {printer(transform)}\n')

        return vm_program

    def apply_ir_module_transforms(self, ir_module: IRModule, module_dir: str) -> IRModule:
        return apply_mutis_specific_transforms(ir_module, module_dir=module_dir, dump_ir=self.options.debug_dump_ir)

    def process_args(self, const_args, optional_args: Tuple[bool, ...]):
        params = []
        kernel_params: List[Var] = []
        param2attrs: Dict[Var, ParamAttrs] = {}
        for arg_idx, (name, arg_type) in enumerate(zip(self.param_names, self.param_types)):
            if arg_idx in self.const_param_indices:
                params.append(const_args[self.const_param_indices.index(arg_idx)])
            else:
                if (
                    arg_idx in self.optional_param_indices
                    and not optional_args[self.optional_param_indices.index(arg_idx)]
                ):
                    # this optional argument is not provided
                    params.append(None)
                else:
                    kernel_param = Var(name, arg_type)
                    params.append(kernel_param)
                    kernel_params.append(kernel_param)
                    param2attrs[kernel_param] = ParamAttrs()
                    if arg_idx in self.weight_param_indices:
                        param2attrs[kernel_param].is_weight = True
        return params, kernel_params, param2attrs

    def resolve_cache_dir(self, const_args, optional_args, extra_key: str = ''):
        items = []  # the items of the cache dir

        # append const arguments
        for const_arg in const_args:
            items.append(normalize_const_arg_repr(const_arg))

        # append the hash of the key string
        key_options = ['debug_use_schedules', 'tuning_trigger_params', 'tuning_trigger_expression']
        key_options_dict = {k: getattr(self.options, k) for k in key_options}
        string_key: str = '\n'.join([pprint.pformat(key_options_dict), str(optional_args), extra_key])
        items.append(hashlib.sha256(string_key.encode()).hexdigest()[:8])

        # append 'debug' if print debug option is enabled
        if self.options.debug_print_vm_inst_output is not False:
            if isinstance(self.options.debug_print_vm_inst_output, bool):
                items.append('debug')
            else:
                items.append(
                    'debug_{}'.format(
                        hashlib.sha256(str(self.options.debug_print_vm_inst_output).encode()).hexdigest()[:4]
                    )
                )

        cache_key: str = normalize_filename('_'.join([str(s) for s in items]))
        cache_dir: str = os.path.join(
            mutis.option.get_option('cache_dir'), 'space-{}'.format(self.options.space), self.name, cache_key
        )
        if self.options.disable_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)
        os.makedirs(cache_dir, exist_ok=True)
        cache_key_path = os.path.join(cache_dir, 'meta', 'cache_key.txt')
        if not os.path.exists(cache_key_path):
            os.makedirs(os.path.dirname(cache_key_path), exist_ok=True)
            with open(cache_key_path, 'w') as f:
                f.write(string_key)
        else:
            with open(cache_key_path, 'r') as f:
                assert f.read() == string_key, 'Cache key mismatch'
        return cache_dir

    def build_template(self, const_args, optional_args, cache_dir: str, candidates: List[Any]):
        # build each schedule candidate, in parallel or serial
        if self.options.disable_parallel_build or len(candidates) <= 1:
            imap = serial_imap
        else:
            lazy_initialize_cuda()
            imap = parallel_imap

        items = [
            '{}'.format(normalize_const_arg_repr(const_arg))
            for i, const_arg in zip(self.const_param_indices, const_args)
        ]
        desc = '[Mutis] Building {}({})'.format(self.name, ', '.join(items))
        status: List[bool] = list(
            tqdm(
                imap(
                    func=lambda idx: self.build_candidate(
                        candidate_dir=os.path.join(cache_dir, 'candidates', str(idx)), candidate=candidates[idx]
                    ),
                    jobs=list(range(len(candidates))),
                    num_workers=mutis.option.get_option('build_workers'),
                ),
                desc=desc,
                total=len(candidates),
            )
        )
        compiled_modules: List[Optional[CompiledModule]] = [
            load_compiled_module(os.path.join(cache_dir, 'candidates', str(idx), 'module')) if s else None
            for idx, s in enumerate(status)
        ]

        # group these candidates by their weight transform
        transform2candidates: Dict[str, List[int]] = collections.defaultdict(list)
        for i in range(len(candidates)):
            if compiled_modules[i] is None:
                # this candidate is not valid
                continue
            with open(os.path.join(cache_dir, 'candidates', str(i), 'weight_transform.txt'), 'r') as f:
                weight_transform = f.read()
                transform2candidates[weight_transform].append(i)

        os.makedirs(os.path.join(cache_dir, 'groups'), exist_ok=True)
        group_indices = {
            group_idx: list(candidates) for group_idx, candidates in enumerate(transform2candidates.values())
        }
        with open(os.path.join(cache_dir, 'groups', 'candidates.txt'), 'w') as f:
            f.write(tabulate.tabulate(group_indices.items(), headers=['Group Index', 'Candidate Index']))
        with open(os.path.join(cache_dir, 'groups', 'candidates.json'), 'w') as f:
            json.dump(group_indices, f, indent=2)
        for idx, transform_text in enumerate(transform2candidates):
            with open(os.path.join(cache_dir, 'groups', f'transform_{idx}.txt'), 'w') as f:
                f.write(transform_text)

        group_sizes = [len(candidates) for candidates in transform2candidates.values()]
        num_valid_modules = sum(group_sizes)
        if num_valid_modules == 0:
            raise RuntimeError('No valid schedule is found.')
        elif self.options.verbose:
            print('Found {} valid candidates, partitioned into groups: {}.'.format(num_valid_modules, group_sizes))

        # finish building
        with open(os.path.join(cache_dir, 'meta', 'done'), 'w') as f:
            f.write('')

        return JitInstanceTemplate(
            options=self.options,
            jit_template=self,
            const_args=const_args,
            optional_args=optional_args,
            cache_dir=cache_dir,
            groups=[[(i, compiled_modules[i]) for i in candidates] for candidates in transform2candidates.values()],
        )

    def instantiate(self, const_args, optional_args: Tuple[bool, ...]) -> JitInstanceTemplate:
        # 2. resolve cache directory
        cache_dir = self.resolve_cache_dir(const_args, optional_args)

        # 3. load the compiled jit instance template if it has been cached in the disk
        #    or build it and cache to disk
        done_path = os.path.join(cache_dir, 'meta', 'done')
        if not os.path.exists(done_path):
            # prevent multiple processes from building the same graph
            lock_path = os.path.join(cache_dir, 'meta', 'lock')
            with filelock.FileLock(lock_path):
                if not os.path.exists(done_path):
                    return self.build(const_args, optional_args, cache_dir)
                # built by another process, go outside the if-body

        assert os.path.exists(done_path)
        if self.options.verbose:
            logger.info('Load cache for {} with constant args {} from {}.'.format(self.name, const_args, cache_dir))
        return JitInstanceTemplate.load(
            options=self.options,
            jit_template=self,
            const_args=const_args,
            optional_args=optional_args,
            cache_dir=cache_dir,
        )

    def build(self, const_args, optional_args: Tuple[bool, ...], cache_dir: str) -> JitInstanceTemplate:
        pass

    def build_candidate(self, candidate_dir: str, candidate: Any) -> bool:
        raise NotImplementedError()


class JitGraphTemplate(JitBaseTemplate):
    def resolve_cache_dir(self, const_args, optional_args, extra_key: str = ''):
        # 1. get a computation graph given the constant arguments
        graph = self.trace_graph(const_args, optional_args)
        return super().resolve_cache_dir(const_args, optional_args, extra_key=graph.astext())

    def build_candidate(self, candidate_dir: str, candidate: Tuple[Graph, Schedule]) -> bool:
        graph, schedule = candidate
        # dump schedule textual representation
        os.makedirs(candidate_dir, exist_ok=True)
        with open(os.path.join(candidate_dir, 'schedule.txt'), 'w') as f:
            f.write(graph.astext(schedule=schedule))

        os.makedirs(candidate_dir, exist_ok=True)
        # lower to virtual machine
        try:
            vm_program: VirtualMachineProgram = generate_virtual_machine_program(graph, schedule)
        except CompilationFailedError as e:
            with open(os.path.join(candidate_dir, 'vm_generation_error.txt'), 'w') as f:
                f.write(traceback.format_exc())
            return False

        return self.build_vm_program(vm_program, candidate_dir)

    def build(self, const_args, optional_args, cache_dir: str) -> JitInstanceTemplate:
        graph: Graph = self.trace_graph(const_args, optional_args)

        # dump textual representation of the graph
        with open(os.path.join(cache_dir, 'graph.txt'), 'w') as f:
            f.write(graph.astext())

        # perform graph processes
        graph = self.graph_postprocess(graph=graph, cache_dir=cache_dir)

        # generate schedules
        dump_schedule_dir: Optional[str] = os.path.join(cache_dir, 'scheduling') if self.options.debug_dump_ir else None
        schedules = generate_schedules(space=self.options.space, graph=graph, dump_ir_dir=dump_schedule_dir)

        if self.options.debug_use_schedules is not None:
            schedules = [schedules[i] for i in self.options.debug_use_schedules if 0 <= i < len(schedules)]

        graph.dump_schedule_summary(schedules, summary_path=os.path.join(cache_dir, 'schedules.txt'))

        candidates = [(graph, schedule) for schedule in schedules]
        return self.build_template(const_args, optional_args, cache_dir, candidates)

    def trace_graph(self, const_args, optional_args: Tuple[bool, ...]) -> Graph:
        params, kernel_params, param2attrs = self.process_args(const_args, optional_args)
        with GraphContext(name=self.fn.__name__, params=kernel_params, param2attrs=param2attrs) as ctx:
            try:
                self.fn(*params)
            except Exception as e:
                raise JitError(traceback.format_exc(limit=2)) from e
        return ctx.graph()

    def graph_postprocess(self, graph: Graph, cache_dir: str) -> Graph:
        from mutis.jit.graph_postprocess.resolve_weight_size import resolve_weight_size_transform
        from mutis.jit.graph_postprocess.fuse_load_cast import fuse_load_cast_transform

        transforms = [resolve_weight_size_transform(), fuse_load_cast_transform()]

        if self.options.debug_dump_ir:
            shutil.rmtree(os.path.join(cache_dir, 'graph_ir'), ignore_errors=True)
            os.makedirs(os.path.join(cache_dir, 'graph_ir'), exist_ok=True)
            with open(os.path.join(cache_dir, 'graph_ir', '0_Origin.txt'), 'w') as f:
                f.write(str(graph))

        for idx, transform in enumerate(transforms):
            graph = transform.transform(graph)
            if self.options.debug_dump_ir:
                with open(
                    os.path.join(cache_dir, 'graph_ir', f'{idx + 1}_{transform.__class__.__name__}.txt'), 'w'
                ) as f:
                    f.write(str(graph))
        return graph


class JitVMTemplate(JitBaseTemplate):
    _current_space: Optional[int] = None

    def resolve_cache_dir(self, const_args, optional_args, extra_key: str = ''):
        with open(inspect.getfile(self.fn), 'r') as f:
            extra_key += f.read()
        return super().resolve_cache_dir(const_args, optional_args, extra_key=extra_key)

    def build_candidate(self, candidate_dir: str, candidate: VirtualMachineProgram) -> bool:
        if 'schedule' in candidate.annotations:
            schedule = candidate.annotations['schedule']
            os.makedirs(candidate_dir, exist_ok=True)
            with open(os.path.join(candidate_dir, 'schedule.txt'), 'w') as f:
                f.write(schedule)
        if 'schedule_dict' in candidate.annotations:
            schedule_dict = candidate.annotations['schedule_dict']
            os.makedirs(candidate_dir, exist_ok=True)
            with open(os.path.join(candidate_dir, 'schedule.json'), 'w') as f:
                json.dump(schedule_dict, f, indent=2)
        return self.build_vm_program(candidate, candidate_dir)

    def build(self, const_args, optional_args: Tuple[bool, ...], cache_dir: str) -> JitInstanceTemplate:
        # process the arguments
        params, kernel_params, param2attrs = self.process_args(const_args, optional_args)

        # generate virtual machine programs
        assert JitVMTemplate._current_space is None
        JitVMTemplate._current_space = self.options.space
        vm_programs: Union[List[VirtualMachineProgram], VirtualMachineProgram] = self.fn(*params)

        if self.options.debug_use_schedules is not None:
            vm_programs = [vm_programs[i] for i in self.options.debug_use_schedules if 0 <= i < len(vm_programs)]

        JitVMTemplate._current_space = None
        if isinstance(vm_programs, VirtualMachineProgram):
            vm_programs = [vm_programs]
        # for vm_program in vm_programs:
        #     vm_program.param2attrs = param2attrs.copy()

        return self.build_template(const_args, optional_args, cache_dir=cache_dir, candidates=vm_programs)


class TuningRecords:
    def __init__(
        self,
        options: JitOptions,
        cache_dir: str,
        jit_template: JitGraphTemplate,
        jit_instance_template: JitInstanceTemplate,
    ):
        self.options: JitOptions = options
        self.cache_dir: str = cache_dir
        self.param_names: List[str] = jit_template.param_names
        self.param_types: List[BaseType] = jit_template.param_types
        self.jit_instance_template: JitInstanceTemplate = jit_instance_template

        self.trigger_args_record: Dict[Any, Tuple[int, int]] = {}
        self.trigger_key_record: Dict[str, Tuple[int, int]] = {}

        # initialize the args that will be used to trigger tuning
        self.tuning_trigger_arg_indices: List[int] = []
        if self.options.tuning_trigger_params is not None:
            # user provided triggerable args
            name2idx = {name: idx for idx, name in enumerate(self.param_names)}
            self.tuning_trigger_arg_indices = [name2idx[name] for name in self.options.tuning_trigger_params]
        else:
            self.tuning_trigger_arg_indices = []
            for arg_index in jit_template.kernel_param_indices:
                if arg_index in jit_template.weight_param_indices:
                    # ignore weight
                    continue
                tp = self.param_types[arg_index]
                if not (isinstance(tp, DataType) and tp.is_integer()):
                    # ignore non-integer args
                    continue
                self.tuning_trigger_arg_indices.append(arg_index)

        # initialize the expression that will be used to generate trigger key based on above args
        if self.options.tuning_trigger_expression is not None:
            assert self.options.tuning_trigger_params is not None, 'Must provide triggerable args when expr is given'
            self.tuning_trigger_expression: Callable[[Sequence[Any]], Hashable] = self.options.tuning_trigger_expression
        else:

            def default_expression(args: Sequence[Any]) -> Hashable:
                lst = []
                for i, arg in zip(self.tuning_trigger_arg_indices, args):
                    tp = self.param_types[i]
                    if isinstance(tp, DataType) and tp.is_integer():
                        if isinstance(arg, hidet.ir.Expr):
                            from hidet.ir.primitives.math import log2, round

                            lst.append((arg > 32) * round(log2(arg)))
                        else:
                            lst.append((arg > 32) * int(math.log2(arg)))
                        lst.append((arg <= 32) * arg)
                    else:
                        raise NotImplementedError(tp)
                return tuple(lst)

            self.tuning_trigger_expression = default_expression

        self.lock_path = os.path.join(cache_dir, 'meta', 'lock')

        params = [var(name, dtype) for name, dtype in zip(self.param_names, self.param_types)]
        trigger_params = [params[i] for i in self.tuning_trigger_arg_indices]
        self.trigger_key_text = (
            self.tuning_trigger_expression(trigger_params) if self.tuning_trigger_expression else 'default'
        )
        self.trigger_method_string: str = str(
            (trigger_params, self.trigger_key_text, self.options.use_single_weight_transform)
        )
        self.trigger_method_hash: str = hashlib.sha256(self.trigger_method_string.encode()).hexdigest()[:8]
        self.record_dir: str = os.path.join(cache_dir, 'tuning_records', self.trigger_method_hash)
        os.makedirs(self.record_dir, exist_ok=True)
        with open(os.path.join(self.record_dir, 'trigger_method.txt'), 'w') as f:
            f.write(self.trigger_method_string)

    def get_records(self, args: Sequence[Any]) -> Optional[Tuple[int, int]]:
        trigger_args = tuple(args[i] for i in self.tuning_trigger_arg_indices)
        if trigger_args not in self.trigger_args_record:
            trigger_key = str(self.tuning_trigger_expression(trigger_args))
            if trigger_key not in self.trigger_key_record:
                self.sync_with_disk()
                if trigger_key not in self.trigger_key_record:
                    return None
            self.trigger_args_record[trigger_args] = self.trigger_key_record[trigger_key]
        return self.trigger_args_record[trigger_args]

    def put_records(self, args: Sequence[Any], record):
        trigger_args = tuple(args[i] for i in self.tuning_trigger_arg_indices)
        trigger_key = str(self.tuning_trigger_expression(trigger_args))
        self.trigger_args_record[trigger_args] = record
        self.trigger_key_record[trigger_key] = record
        self.sync_with_disk()

    def dump_latency(self, args: Sequence[Any], latency: Dict[Tuple[int, int], float]):
        name = '-'.join('{}-{}'.format(self.param_names[i], args[i]) for i in self.tuning_trigger_arg_indices)
        latency_dir = os.path.join(self.record_dir, 'latency')
        os.makedirs(latency_dir, exist_ok=True)
        sorted_pair = sorted(latency, key=lambda x: latency[x])
        with open(os.path.join(latency_dir, '{}.txt'.format(name)), 'w') as f:
            sch_dicts = {}
            for group_id, module_id in sorted_pair:
                sch_idx = self.jit_instance_template.module_groups[group_id][module_id][0]
                sch_dict_path = os.path.join(self.cache_dir, 'candidates', str(sch_idx), 'schedule.json')
                if os.path.exists(sch_dict_path):
                    with open(sch_dict_path, 'r') as sch_f:
                        sch_dicts[sch_idx] = json.load(sch_f, object_pairs_hook=collections.OrderedDict)
                else:
                    sch_dicts[sch_idx] = {}

            sch_keys = []
            for sch_dict in sch_dicts.values():
                for key in sch_dict:
                    assert isinstance(key, str)
                    if key not in sch_keys:
                        sch_keys.append(key)

            headers = ['index', 'latency (ms)'] + sch_keys
            rows = []
            for group_id, module_id in sorted_pair:
                sch_idx = self.jit_instance_template.module_groups[group_id][module_id][0]
                lat = latency[(group_id, module_id)]
                row = [sch_idx, lat]
                for key in sch_keys:
                    if key in sch_dicts[sch_idx]:
                        row.append(sch_dicts[sch_idx][key])
                    else:
                        row.append('')
                rows.append(row)
            f.write(tabulate.tabulate(rows, headers=headers))

        with open(os.path.join(latency_dir, '{}.json'.format(name)), 'w') as f:
            latency_list = [[a, b, latency[(a, b)]] for a, b in sorted_pair]
            json.dump(latency_list, f)

    def sync_with_disk(self):
        with filelock.FileLock(self.lock_path):
            with shelve.open(os.path.join(self.record_dir, 'records.shelve')) as db:
                for key, value in self.trigger_key_record.items():
                    db[key] = value
                for key, value in db.items():
                    self.trigger_key_record[key] = value
            with open(os.path.join(self.record_dir, 'records.txt'), 'w') as f:
                headers = self.trigger_key_text, 'Group Index', 'Module Index', 'Schedule Index'
                rows = []
                for key, value in self.trigger_key_record.items():
                    schedule_idx = self.jit_instance_template.module_groups[value[0]][value[1]][0]
                    rows.append((key, value[0], value[1], schedule_idx))
                f.write(tabulate.tabulate(rows, headers=headers))


class JitInstanceTemplate:
    def __init__(
        self,
        options: JitOptions,
        jit_template: JitBaseTemplate,
        const_args: Tuple[Any, ...],
        optional_args: Tuple[bool, ...],
        cache_dir: str,
        groups: List[List[Tuple[int, CompiledModule]]],
    ):
        self.options: JitOptions = options
        self.jit_template: JitBaseTemplate = jit_template
        self.const_args: Tuple[Any, ...] = const_args
        self.optional_args: Tuple[bool, ...] = optional_args
        self.cache_dir: str = cache_dir
        self.module_groups: List[List[Tuple[int, CompiledModule]]] = groups
        self.tuning_records: TuningRecords = TuningRecords(
            options=options, cache_dir=cache_dir, jit_template=jit_template, jit_instance_template=self
        )

        optional_args_exists: Dict[int, bool] = {
            idx: optional_args[i] for i, idx in enumerate(jit_template.optional_param_indices)
        }
        self.kernel_param_indices: List[int] = [
            idx for idx in jit_template.kernel_param_indices if optional_args_exists.get(idx, True)
        ]

        assert isinstance(groups[0][0], tuple)

    @staticmethod
    def load(
        options, jit_template, const_args: Tuple[Any, ...], optional_args: Tuple[bool, ...], cache_dir: str
    ) -> JitInstanceTemplate:
        with open(os.path.join(cache_dir, 'groups', 'candidates.json'), 'r') as f:
            group_candidates = json.load(f)
        module_groups: List[List[Tuple[int, CompiledModule]]] = [[] for _ in range(len(group_candidates))]
        for group_idx, candidates in group_candidates.items():
            group_idx = int(group_idx)
            for candidate_idx in candidates:
                module_dir = os.path.join(cache_dir, 'candidates', str(candidate_idx), 'module')
                module = load_compiled_module(module_dir)
                module_groups[group_idx].append((candidate_idx, module))
        return JitInstanceTemplate(
            options=options,
            jit_template=jit_template,
            const_args=const_args,
            optional_args=optional_args,
            cache_dir=cache_dir,
            groups=module_groups,
        )

    def pick_best_candidate(self, args: Sequence[Any]) -> Tuple[int, int]:
        record = self.tuning_records.get_records(args)

        if record is None:
            candidate_latency: Dict[Tuple[int, int], float] = {}

            # get the arguments that will be used to identify the tuning process
            items = []

            # tuning triggering arguments
            for idx in self.tuning_records.tuning_trigger_arg_indices:
                items.append('{}={}'.format(self.jit_template.param_names[idx], args[idx]))

            # constant arguments
            for idx in range(len(self.const_args)):
                name = self.jit_template.param_names[self.jit_template.const_param_indices[idx]]
                # items.append('{}={}'.format(name, normalize_const_arg_repr(self.const_args[idx])))
                items.append('{}'.format(normalize_const_arg_repr(self.const_args[idx])))

            sig = self.jit_template.name + '(' + ', '.join(items) + ')'
            desc = '[Mutis] Tuning {}'.format(sig)
            is_single_candidate = sum(len(group) for group in self.module_groups) <= 1
            with tqdm(total=sum(len(group) for group in self.module_groups), desc=desc) as bar:
                for group_id, group in enumerate(self.module_groups):
                    for module_id, (schedule_idx, module) in enumerate(group):
                        kernel_args = [args[i] for i in self.kernel_param_indices]
                        if self.options.debug_print_vm_inst_output or is_single_candidate:
                            # in this debug mode, we do not measure the latency since it will produce a lot of output
                            latency = 0.0
                        else:
                            latency = benchmark_func(
                                lambda: module(*kernel_args),
                                warmup=mutis.option.get_option('tuning.warmup'),
                                repeat=mutis.option.get_option('tuning.repeat'),
                                maximum_warmup_time=mutis.option.get_option('tuning.maximum_warmup_time'),
                                maximum_repeat_time=mutis.option.get_option('tuning.maximum_repeat_time'),
                                median=True,
                            )
                        candidate_latency[(group_id, module_id)] = latency
                        bar.update(1)
            if self.options.use_single_weight_transform:
                major_group_id = max(range(len(self.module_groups)), key=lambda g: len(self.module_groups[g]))
                all_candidates = list(candidate_latency.keys())
                candidates = [
                    (group_id, module_id) for group_id, module_id in all_candidates if group_id == major_group_id
                ]
                record = min(candidates, key=lambda r: candidate_latency[r])
                best_record = min(all_candidates, key=lambda r: candidate_latency[r])
                if record != best_record:
                    print(
                        'better schedule outside the major group ({:.3f} ms vs {:.3f} ms)'.format(
                            candidate_latency[best_record], candidate_latency[record]
                        )
                    )
            else:
                candidates = list(candidate_latency.keys())
                record = min(candidates, key=lambda r: candidate_latency[r])
            self.tuning_records.put_records(args, record)
            self.tuning_records.dump_latency(args, candidate_latency)
        return record

    def instantiate(self, weight_args) -> JitInstance:
        return JitInstance(self, weight_args)


class JitInstance:
    weights: Set[int] = set()

    def __init__(self, template: JitInstanceTemplate, weight_args):
        self.template: JitInstanceTemplate = template
        self.module_groups: List[List[Tuple[int, CompiledModule]]] = template.module_groups
        self.weight_status: int = -1
        self.weight_args = weight_args

        self.record_unique_weights()

    def __call__(self, *args):
        group_id, module_id = self.template.pick_best_candidate(args)
        module = self.template.module_groups[group_id][module_id][1]

        kernel_args = [args[i] for i in self.template.kernel_param_indices]

        # current_torch_stream = torch._C._cuda_getCurrentStream(torch.cuda.current_device())[0]
        # hidet.ffi.runtime_api.set_current_cuda_stream(current_torch_stream)
        hidet.ffi.runtime_api.set_current_cuda_stream(torch.cuda.current_stream().cuda_stream)

        # transform weight to group_id
        self.apply_transform(group_id, args)

        # launch
        module(*kernel_args)

    def _iterate_weight_pointer(self) -> Sequence[int]:
        import mutis

        for idx, arg in zip(self.template.jit_template.weight_param_indices, self.weight_args):
            if arg is None:
                # this is an optional weight argument
                continue
            if isinstance(arg, torch.Tensor):
                arg = arg.data_ptr()
            elif isinstance(arg, mutis.Tensor):
                arg = arg.storage.data_ptr()
            arg = int(arg)
            yield arg

    def record_unique_weights(self):
        for weight_pointer in self._iterate_weight_pointer():
            if weight_pointer in self.weights:
                raise RuntimeError('The weight argument passes to mutis kernel must be unique')
            self.weights.add(weight_pointer)

    def release_unique_weights(self):
        for weight_pointer in self._iterate_weight_pointer():
            assert weight_pointer in self.weights
            self.weights.remove(weight_pointer)

    def apply_transform(self, idx, args):
        if _benchmark_mode:
            return
        if idx != self.weight_status:
            kernel_args = [args[i] for i in self.template.kernel_param_indices]
            # print('transform from {} to {}'.format(self.weight_status, idx))
            # print(kernel_args[1])
            if self.weight_status != -1:
                # reverse to original weight
                self.module_groups[self.weight_status][0][1]['reverse_weight_transforms'](*kernel_args)
            if idx != -1:
                # transform to id idx
                self.module_groups[idx][0][1]['apply_weight_transforms'](*kernel_args)
            # print(kernel_args[1])
            self.weight_status = idx


class JitCache:
    def __init__(self):
        self.instance_templates: Dict[Any, JitInstanceTemplate] = {}
        self.instances: Dict[Any, JitInstance] = {}

    def insert_instance(self, const_args, optional_args, weight_args, instance: JitInstance):
        key = (const_args, optional_args, weight_args)
        if key in self.instances:
            raise RuntimeError('Instance already exists')
        self.instances[key] = instance

    def lookup_instance(self, const_args, optional_args, weight_args) -> Optional[JitInstance]:
        return self.instances.get((const_args, optional_args, weight_args), None)

    def insert_instance_template(self, const_args, optional_args, instance_template: JitInstanceTemplate):
        key = (const_args, optional_args)
        if key in self.instance_templates:
            raise RuntimeError('Instance template already exists')
        self.instance_templates[key] = instance_template

    def lookup_instance_template(self, const_args, optional_args) -> Optional[JitInstanceTemplate]:
        return self.instance_templates.get((const_args, optional_args), None)


class JitFunction:
    def __init__(self, fn: types.FunctionType, options: JitOptions):
        # jit function template
        template_cls = JitGraphTemplate if options.kind == 'graph' else JitVMTemplate
        self.template: JitBaseTemplate = template_cls(options, fn)

        # jit options
        self.options: JitOptions = options

        # jit cache
        self.cache: JitCache = JitCache()

    def __call__(self, *args):
        # pr.enable()
        if len(args) != self.template.num_params:
            raise TypeError('Expected {} arguments, got {}'.format(self.template.num_params, len(args)))

        const_args = tuple(args[i] for i in self.template.const_param_indices)
        optional_args = tuple(args[i] is not None for i in self.template.optional_param_indices)
        weight_args = tuple(args[i] for i in self.template.weight_param_indices)

        instance: Optional[JitInstance] = self.cache.lookup_instance(const_args, optional_args, weight_args)

        if not instance:
            # pr.disable()
            instance_template: Optional[JitInstanceTemplate] = self.cache.lookup_instance_template(
                const_args, optional_args
            )

            if instance_template is None:
                instance_template: JitInstanceTemplate = self.template.instantiate(const_args, optional_args)
                self.cache.insert_instance_template(const_args, optional_args, instance_template)

            instance: JitInstance = instance_template.instantiate(weight_args)
            self.cache.insert_instance(const_args, optional_args, weight_args, instance)
            # pr.enable()

        ret = instance(*args)
        # pr.disable()
        return ret

    def restore_weights(self, *args):
        for instance in self.cache.instances.values():
            instance.apply_transform(-1, args)


def jit(
    fn=None,
    *,
    space=None,
    use_single_weight_transform: bool = False,
    tuning_trigger_params: Optional[List[str]] = None,
    tuning_trigger_expression: Optional[Callable[[List[Any]], Hashable]] = None,
    disable_cache: bool = False,
    disable_parallel_build: bool = False,
    debug_dump_ir: Optional[bool] = None,
    debug_use_schedules: Optional[Sequence[int]] = None,
    debug_print_vm_inst_output: Union[bool, Dict[str, int]] = False,
    debug_enable_device_assertion: bool = False,
    verbose=False,
) -> Union[JitFunction, Callable[[types.FunctionType], JitFunction]]:
    def decorator(fn: types.FunctionType) -> JitFunction:
        jit_function = JitFunction(
            fn,
            options=JitOptions(
                space=space,
                kind='graph',
                use_single_weight_transform=use_single_weight_transform,
                tuning_trigger_params=tuning_trigger_params,
                tuning_trigger_expression=tuning_trigger_expression,
                disable_cache=disable_cache,
                disable_parallel_build=disable_parallel_build,
                debug_dump_ir=debug_dump_ir,
                debug_use_schedules=debug_use_schedules,
                debug_print_vm_inst_output=debug_print_vm_inst_output,
                debug_enable_devise_assertion=debug_enable_device_assertion,
                verbose=verbose,
            ),
        )
        __all_jit_functions.add(jit_function)
        return jit_function

    if fn is not None:
        return decorator(fn)
    else:
        return decorator


def vm_jit(
    fn: Any = None,
    *,
    space=None,
    use_single_weight_transform: bool = False,
    tuning_trigger_params: Optional[List[str]] = None,
    tuning_trigger_expression: Optional[Callable[[List[Any]], Hashable]] = None,
    disable_cache: bool = False,
    disable_parallel_build: bool = False,
    debug_dump_ir: Optional[bool] = None,
    debug_use_schedules: Optional[Sequence[int]] = None,
    debug_print_vm_inst_output: Union[bool, Dict[str, int]] = False,
    debug_enable_device_assertion: bool = False,
    verbose=False,
):
    def decorator(fn: types.FunctionType) -> JitFunction:
        jit_function = JitFunction(
            fn,
            options=JitOptions(
                space=space,
                kind='vm',
                use_single_weight_transform=use_single_weight_transform,
                tuning_trigger_params=tuning_trigger_params,
                tuning_trigger_expression=tuning_trigger_expression,
                disable_cache=disable_cache,
                disable_parallel_build=disable_parallel_build,
                debug_dump_ir=debug_dump_ir,
                debug_use_schedules=debug_use_schedules,
                debug_print_vm_inst_output=debug_print_vm_inst_output,
                debug_enable_devise_assertion=debug_enable_device_assertion,
                verbose=verbose,
            ),
        )
        __all_jit_functions.add(jit_function)
        return jit_function

    if fn is not None:
        return decorator(fn)
    else:
        return decorator


def get_current_space() -> int:
    if JitVMTemplate._current_space is None:
        raise RuntimeError('No current space')
    return JitVMTemplate._current_space


def empty_jit_cache():
    for jit_function in __all_jit_functions:
        for instance in jit_function.cache.instances.values():
            instance.release_unique_weights()
        jit_function.cache.instances.clear()
        jit_function.cache.instance_templates.clear()
    gc.collect()
    assert len(JitInstance.weights) == 0


_benchmark_mode = False


def set_benchmark_mode(flag=False):
    global _benchmark_mode
    _benchmark_mode = flag
