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
from typing import Sequence
from hidet.ir.module import IRModule

from .base import Pass, FunctionPass, SequencePass, RepeatFunctionPass, PassContext
from .instruments import PassInstrument, SaveIRInstrument, ProfileInstrument

from .check_semantics import check_semantics_pass
from .unify_global_objects import unify_global_objects_pass
from .flatten_tensor_slice import flatten_tensor_slice_pass
from .flatten_tensor_index import flatten_tensor_index_pass
from .generate_launch_func import generate_launch_func_pass
from .explicit_unroll import explicit_unroll_pass
from .import_primitive_functions import import_primitive_functions_pass
from .simplify_stmt import simplify_stmt_pass
from .expand_let_expr import expand_let_expr_pass
from .instantiate_symbols import instantiate_symbols_pass
from .resolve_generic_primitive_function import resolve_primitive_func_pass
from .inline_function import inline_function_pass
from .add_explicit_cast import add_explicit_cast_pass
from .inline_let_stmt import inline_let_stmt_pass
from .rule_based_simplifier import rule_based_simplify_pass
from .seperate_loop_invariants import separate_loop_invariants_pass
from .normalize_const_tensor import normalize_const_tensor_pass
from .lower_task_mapping import lower_task_mapping_pass
from .lower_protect_access import lower_protect_access_pass
from .declare_to_let import declare_to_let_pass
from .propagate_launch_bound import propagate_launch_bound_pass
from .check_launch_configuration import check_launch_configuration_pass
from .lower_special_cast import lower_special_cast_pass
from .annotate_header_and_libs import annotate_header_and_libs_pass
from .lower_integer_subbyte import lower_integer_subbyte_pass
from .resolve_gpgpu_functions import resolve_gpgpu_functions_pass

from .tile.generic.canonicalize_to_ssa import canonicalize_to_ssa_pass
from .tile.generic.inject_explicit_transform_ops import inject_explicit_transform_ops_pass
from .tile.generic.canonicalize_expressions import canonicalize_expressions_pass
from .tile.generic.fold_constant import fold_constant_pass
from .tile.generic.pattern_transform import pattern_transform_pass
from .tile.generic.loop_invariant_code_motion import loop_invariant_code_motion_pass

from .tile.cuda.resolve_dot import resolve_dot_pass
from .tile.cuda.instantiate_layout import instantiate_layout_pass
from .tile.cuda.coalesce_memory_access import coalesce_memory_access_pass
from .tile.cuda.remove_layout_convert import remove_layout_convert_pass
from .tile.cuda.software_pipeline import software_pipeline_pass
from .tile.cuda.split_dot_k import split_dot_k_pass
from .tile.cuda.plan_shared_memory import plan_shared_memory_pass
from .tile.cuda.lower_tile_dialect import lower_tile_dialect_pass


def lower_with(ir_module: IRModule, transforms: Sequence[Pass]) -> IRModule:
    ctx = PassContext.current()
    for instrument in ctx.instruments:
        instrument.before_all_passes(ir_module)
    for transform in transforms:
        ir_module = transform(ir_module)
    for instrument in ctx.instruments:
        instrument.after_all_passes(ir_module)

    return ir_module


def lower(ir_module: IRModule) -> IRModule:
    tile_generic_transforms = [
        inject_explicit_transform_ops_pass(),
        canonicalize_expressions_pass(),
        canonicalize_to_ssa_pass(),
        fold_constant_pass(),
        pattern_transform_pass(),
        loop_invariant_code_motion_pass(),
    ]

    tile_cuda_transforms = [
        resolve_dot_pass(),
        instantiate_layout_pass(),
        coalesce_memory_access_pass(),
        remove_layout_convert_pass(),
        loop_invariant_code_motion_pass(),
        software_pipeline_pass(),
        split_dot_k_pass(),
        plan_shared_memory_pass(),
        lower_tile_dialect_pass(),
    ]

    transforms = [
        check_semantics_pass(),
        # necessary passes
        unify_global_objects_pass(),
        generate_launch_func_pass(),
        flatten_tensor_slice_pass(),
        lower_protect_access_pass(),
        lower_task_mapping_pass(),
        normalize_const_tensor_pass(),
        declare_to_let_pass(),
        rule_based_simplify_pass(),  # make ir more readable
        flatten_tensor_index_pass(),
        lower_integer_subbyte_pass(),
        lower_special_cast_pass(),
        inline_function_pass(),
        resolve_primitive_func_pass(),
        import_primitive_functions_pass(),
        resolve_primitive_func_pass(),
        import_primitive_functions_pass(),
        propagate_launch_bound_pass(),
        resolve_gpgpu_functions_pass(),
        add_explicit_cast_pass(),
        declare_to_let_pass(),
        instantiate_symbols_pass(),
        check_launch_configuration_pass(),
        # simplification
        expand_let_expr_pass(),
        inline_let_stmt_pass(),
        explicit_unroll_pass(),
        rule_based_simplify_pass(),
        separate_loop_invariants_pass(),
        inline_let_stmt_pass(),
        simplify_stmt_pass(),
        annotate_header_and_libs_pass(),
    ]
    ir_module = lower_with(ir_module, tile_generic_transforms + tile_cuda_transforms + transforms)
    return ir_module
