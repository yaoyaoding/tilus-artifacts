from __future__ import annotations

from typing import Union, Dict, Any, Optional, Tuple, List

from hidet.ir import Var
from hidet.ir.type import DataType
from hidet.ir.dtypes import uint8
from hidet.ir.dtypes import int8, float16
from mutis.ir.graph import Operator, Tensor
from mutis.ops.ldst import Load
from mutis.vm.ir.inst import Instruction, CastInst, ViewInst
from mutis.vm.ir.weight_transform import WeightTransform, WeightValueTransform
from mutis.vm.ir.stmt import Stmt, SeqStmt
from mutis.ops.transform import Cast
from mutis.backends.codegen import register_emitter, BaseEmitter


@register_emitter(Cast, priority=0)
class CastEmitter(BaseEmitter):
    def __init__(self, codegen, op: Operator, variant: Dict[str, Any]):
        super().__init__(codegen, op, variant)

    def emit(self) -> Union[Stmt, Instruction]:
        x = self.codegen.tensor2value[self.op.inputs[0]]
        seq = []
        cast_inst = CastInst.create(dtype=self.op.output.elem_type, x=x)
        self.tensor2value[self.op.output] = cast_inst.output.as_register_value()
        seq.append(cast_inst)
        return SeqStmt(seq)
