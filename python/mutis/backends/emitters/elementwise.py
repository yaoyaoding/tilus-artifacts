from __future__ import annotations

from typing import Union, Dict, Any, cast

from mutis.ir.graph import Operator
from mutis.vm.ir.inst import Instruction, ElementwiseUnaryInst, ElementwiseBinaryInst, BroadcastElementwiseBinaryInst
from mutis.vm.ir.stmt import Stmt
from mutis.vm.ir.value import RegisterValue
from mutis.ops.arithmatic import ElementwiseUnary, ElementwiseBinary, BroadcastElementwiseBinary
from mutis.backends.codegen import register_emitter, BaseEmitter


@register_emitter(ElementwiseUnary, priority=0)
class ElementwiseUnaryEmitter(BaseEmitter):
    def emit(self) -> Union[Stmt, Instruction]:
        op: ElementwiseUnary = cast(ElementwiseUnary, self.op)
        output_value = RegisterValue(op.output.elem_type, self.tensor2layout[op.output])
        inst = ElementwiseUnaryInst(
            output_value,
            x=self.tensor2value[op.inputs[0]].as_register_value(),
            op=op.op,
            other_kwargs={a: b for a, b in op.attrs.items() if a != 'op'},
        )
        self.tensor2value[op.output] = inst.output.as_register_value()
        return inst


@register_emitter(ElementwiseBinary, priority=0)
class ElementwiseBinaryEmitter(BaseEmitter):
    def emit(self) -> Union[Stmt, Instruction]:
        op: ElementwiseBinary = cast(ElementwiseBinary, self.op)
        output_value = RegisterValue(op.output.elem_type, self.tensor2layout[op.output])
        inst = ElementwiseBinaryInst(
            output_value,
            x=self.tensor2value[op.inputs[0]].as_register_value(),
            y=self.tensor2value[op.inputs[1]].as_register_value(),
            op=op.op,
        )
        self.tensor2value[op.output] = inst.output.as_register_value()
        return inst


@register_emitter(BroadcastElementwiseBinary, priority=0)
class BroadcastElementwiseBinaryEmitter(BaseEmitter):
    def emit(self) -> Union[Stmt, Instruction]:
        op: BroadcastElementwiseBinary = cast(BroadcastElementwiseBinary, self.op)
        output_value = RegisterValue(op.output.elem_type, self.tensor2layout[op.output])
        inst = BroadcastElementwiseBinaryInst(
            output_value,
            r=self.tensor2value[op.inputs[0]].as_register_value(),
            s=op.s,
            op=op.op,
            tensor_left=op.tensor_left,
        )
        self.tensor2value[op.output] = inst.output.as_register_value()
        return inst
