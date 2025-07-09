from hidet.ir import Function, IRModule
from hidet.ir.functors import IRRewriter
from hidet.transforms.base import FunctionPass, Pass


class ResolveGPGPUFunctionsRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.kind_map = {}

        if self.is_cuda():
            self.kind_map.update({'gpgpu_kernel': 'cuda_kernel', 'gpgpu_internal': 'cuda_internal'})
        elif self.is_hip():
            self.kind_map.update({'gpgpu_kernel': 'hip_kernel', 'gpgpu_internal': 'hip_internal'})

    @staticmethod
    def is_cuda():
        from hidet.cuda.avaliability import available
        from hidet.cuda.device import device_count

        return available() and device_count() > 0

    @staticmethod
    def is_hip():
        from hidet.hip.device import available
        from hidet.hip.device import device_count

        return available() and device_count() > 0

    def visit_Function(self, func: Function):
        if 'gpgpu' in func.kind:
            return Function(
                name=func.name,
                params=func.params,
                ret_type=func.ret_type,
                body=func.body,
                kind=self.kind_map[func.kind],
                attrs=func.attrs,
            )
        else:
            return func


class ResolveGPGPUFunctionsPass(Pass):
    def process_module(self, ir_module: IRModule) -> IRModule:
        rewriter = ResolveGPGPUFunctionsRewriter()
        return rewriter(ir_module)


def resolve_gpgpu_functions_pass() -> Pass:
    return ResolveGPGPUFunctionsPass()
