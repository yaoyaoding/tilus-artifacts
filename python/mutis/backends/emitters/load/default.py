from __future__ import annotations

import functools
from typing import Union, Dict, Any, List, cast

from hidet.ir.dtypes import boolean, int32
from hidet.ir.expr import Var, Expr, logical_and
from hidet.ir.utils.index_transform import index_multiply, index_sum, index_add
from mutis.backends.codegen import register_emitter, BaseEmitter
from mutis.ir.graph import Operator
from mutis.ops.ldst import Load
from mutis.vm.ir.builder import VirtualMachineBuilder
from mutis.vm.ir.inst import Instruction
from mutis.vm.ir.stmt import Stmt


@register_emitter(Load, priority=0, variant={'stages': 'gmem->regs'})
class LoadEmitter(BaseEmitter):
    def __init__(self, codegen, op: Load, variant):
        super().__init__(codegen, op, variant)
        self.transform_weight: bool = False

    def supports(self, op: Operator, variant: Dict[str, Any]) -> bool:
        return True

    def emit(self) -> Union[Stmt, Instruction]:
        load = cast(Load, self.op)

        def f_offset(axes: List[Var]) -> Expr:
            tile_offsets = self.codegen.tensor2tile[load.output].tile_offsets()
            indices = index_add(tile_offsets, axes)
            return index_sum(index_multiply(indices, load.strides), init=int32.zero)

        def f_mask(axes: List[Var]) -> Expr:
            tile_offsets = self.codegen.tensor2tile[load.output].tile_offsets()
            indices = index_add(tile_offsets, axes)
            return functools.reduce(logical_and, [a < b for a, b in zip(indices, load.shape)], boolean.true)

        vb = VirtualMachineBuilder()

        loaded = vb.load_global(
            dtype=load.dtype,
            layout=self.codegen.tensor2layout[load.output],
            ptr=load.ptr,
            f_offset=f_offset,
            f_mask=f_mask,
        )
        if load.cast_dtype and load.cast_dtype != load.dtype:
            loaded = vb.cast(loaded, dtype=load.cast_dtype)
        self.tensor2value[load.output] = loaded

        return vb.finish()
