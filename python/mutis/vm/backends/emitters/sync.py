from hidet.ir.primitives.cuda.sync import syncthreads
from mutis.extension.primitives.cuda.barrier import barrier_sync
from mutis.vm.backends.codegen import BaseInstEmitter, register_inst_emitter
from mutis.vm.ir.inst import SyncThreadsInst, SyncReduceThreadsInst
from mutis.target import gpgpu_any, nvgpu_any


@register_inst_emitter(SyncThreadsInst, target=gpgpu_any)
class SyncThreadsEmitter(BaseInstEmitter):
    def emit(self, inst: SyncThreadsInst):
        self.sync()


@register_inst_emitter(SyncReduceThreadsInst, target=nvgpu_any)
class SyncReduceThreadsEmitter(BaseInstEmitter):
    def emit(self, inst: SyncReduceThreadsInst):
        self.declare(inst.var, init=self.sync_reduce(inst.reduce_value, op=inst.reduce_op))
