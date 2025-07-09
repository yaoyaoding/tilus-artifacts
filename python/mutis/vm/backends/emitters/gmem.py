from hidet.ir.expr import Expr, cast
from mutis.vm.backends.codegen import BaseInstEmitter, register_inst_emitter
from mutis.vm.ir.inst import AllocateGlobalInst
from mutis.target import gpgpu_any


@register_inst_emitter(AllocateGlobalInst, target=gpgpu_any)
class AllocateGlobalInstEmitter(BaseInstEmitter):
    def emit(self, inst: AllocateGlobalInst):
        ptr: Expr = self.codegen.allocate_global_memory(nbytes=inst.nbytes, clean=inst.require_clean)
        self.declare(inst.var, init=cast(ptr, inst.var.type))
