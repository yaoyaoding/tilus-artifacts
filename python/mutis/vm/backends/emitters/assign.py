from hidet.ir.expr import tensor_var
from hidet.ir.tools import rewrite
from mutis.vm.backends.codegen import BaseInstEmitter, register_inst_emitter
from mutis.vm.ir.inst import AssignInst
from mutis.target import gpgpu_any


@register_inst_emitter(AssignInst, target=gpgpu_any)
class AllocateInstEmitter(BaseInstEmitter):
    def emit(self, inst: AssignInst):
        value = inst.output.as_register_value()
        input_value = inst.inputs[0].as_register_value()
        var = self.get_or_allocate_var(value=value, name='regs')
        assert input_value.dtype == value.dtype
        assert input_value.layout.quick_equal(value.layout)
        with self.for_range(value.layout.local_size) as i:
            self.buffer_store(buf=var, indices=[i], value=self.value2var[input_value][i])

        self.value2var[value] = var
