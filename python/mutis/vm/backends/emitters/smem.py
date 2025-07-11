from typing import List, Optional, Union

from hidet.ir.dtypes import int32, uint32, uint16, uint8
from hidet.ir.type import DataType, void_p, void, tensor_pointer_type
from hidet.ir.expr import Expr, Var, tensor_var, tensor_pointer_var, cast, index_vars, if_then_else
from hidet.ir.tools import rewrite
from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared
from hidet.ir.primitives.cuda.smem import dynamic_shared_memory
from hidet.ir.primitives.cuda.ldst import load, store
from hidet.ir.utils.index_transform import index_add
from mutis.ir.layout import Layout
from mutis.vm.backends.codegen import BaseInstEmitter, register_inst_emitter
from mutis.vm.ir.inst import AllocateSharedInst, FreeSharedInst, ViewSharedInst, LoadSharedInst, StoreSharedInst
from mutis.vm.ir.value import SharedValue, RegisterValue
from mutis.ir.analyzers import analyze_info, TensorInfo
from mutis.utils import gcd
from mutis.target import nvgpu_any, gpgpu_any


@register_inst_emitter(AllocateSharedInst, target=gpgpu_any)
class AllocateSharedInstEmitter(BaseInstEmitter):
    def emit(self, inst: AllocateSharedInst):
        value: SharedValue = inst.output.as_shared_value()

        allocator_addr = self.codegen.allocate_shared_value(value, nbytes=value.nbytes())
        self.value2var[value] = self.declare_var(
            name='shared',
            tp=tensor_pointer_type(dtype=value.dtype, shape=[value.size]),
            init=cast(dynamic_shared_memory(byte_offset=allocator_addr), ~value.dtype),
        )
        shared_space_addr = cvta_generic_to_shared(self.value2var[value])
        self.shared_value_shared_space_addr[value] = self.declare_var(
            name='shared_addr', tp=int32, init=shared_space_addr
        )

        if inst.init is not None:
            raise NotImplementedError()


@register_inst_emitter(FreeSharedInst, target=gpgpu_any)
class FreeSharedInstEmitter(BaseInstEmitter):
    def emit(self, inst: FreeSharedInst):
        value: SharedValue = inst.inputs[0].as_shared_value()
        self.codegen.free_shared_value(value)

        del self.value2var[value]
        del self.shared_value_shared_space_addr[value]


@register_inst_emitter(ViewSharedInst, target=gpgpu_any)
class ViewSharedInstEmitter(BaseInstEmitter):
    def emit(self, inst: ViewSharedInst):
        value: SharedValue = inst.output.as_shared_value()
        base_value: SharedValue = inst.inputs[0].as_shared_value()

        view_indices: List[Expr] = inst.indices.copy()
        view_indices.extend([int32.zero for _ in range(len(base_value.shape) - len(view_indices))])

        base_var: Var = self.value2var[base_value]
        self.value2var[value] = self.declare_var(
            name='shared_view',
            tp=tensor_pointer_type(dtype=value.dtype, shape=[value.size]),
            init=cast(~base_var[base_value.layout(*view_indices)], ~value.dtype),
        )

        base_addr = self.shared_value_shared_space_addr[base_value]
        self.shared_value_shared_space_addr[value] = self.declare_var(
            name='shared_view_addr',
            tp=int32,
            init=base_addr + base_value.layout(*view_indices) * base_value.dtype.nbytes,
        )
