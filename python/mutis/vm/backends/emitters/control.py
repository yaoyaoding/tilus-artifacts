from mutis.extension.primitives.cuda.control import exit
from mutis.vm.backends.codegen import BaseInstEmitter, register_inst_emitter
from mutis.vm.ir.inst import ExitInst
from mutis.target import nvgpu_any


@register_inst_emitter(ExitInst, target=nvgpu_any)
class ExitInstEmitter(BaseInstEmitter):
    def emit(self, inst: ExitInst):
        self.append(exit())
