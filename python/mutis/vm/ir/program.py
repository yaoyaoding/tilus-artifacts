from typing import List, Dict, Optional, Any
from hidet.ir.type import DataType
from hidet.ir.expr import Var, Expr
from hidet.runtime import CompiledModule
from mutis.ir.layout import Layout
from mutis.ir.graph import ParamAttrs
from mutis.ir.tile import BlockMapping
from mutis.vm.ir.stmt import Stmt
from mutis.vm.ir.weight_transform import WeightTransform
from mutis.utils import prod


class VirtualMachineProgram:
    def __init__(
        self,
        name: str,
        params: List[Var],
        param2attrs: Dict[Var, ParamAttrs],
        num_warps: int,
        block_axes: List[Var],
        num_blocks: List[Expr],
        body: Stmt,
        block_mapping: BlockMapping,
        weight_transforms: Optional[Dict[Var, List[WeightTransform]]],
        var2divisibility: Optional[Dict[Var, int]],
        annotations: Optional[Dict[str, str]],
    ):
        self.name: str = name
        self.params: List[Var] = params
        self.param2attrs: Dict[Var, ParamAttrs] = param2attrs
        self.num_warps: int = num_warps
        self.block_axes: List[Var] = block_axes
        self.num_blocks: List[Expr] = num_blocks
        self.body: Stmt = body
        self.block_mapping: BlockMapping = block_mapping
        self.weight_transforms: Dict[Var, List[WeightTransform]] = weight_transforms if weight_transforms else {}
        self.var2divisibility: Dict[Var, int] = (
            var2divisibility.copy() if var2divisibility else {}
        )  # todo: make compiler analyze this
        self.annotations: Dict[str, Any] = annotations.copy() if annotations else {}

        assert block_mapping is not None
        assert all(isinstance(v, Expr) for v in num_blocks)

    def __str__(self):
        from mutis.vm.ir.printer import VirtualMachinePrinter

        printer = VirtualMachinePrinter()
        return str(printer(self))

    def __call__(self, *args):
        module = self.build()
        return module(*args)

    def build(self, dump_ir_dir: Optional[str] = None) -> CompiledModule:
        from mutis.vm.backends import generate_ir_module
        from mutis.extension.transforms import apply_mutis_specific_transforms

        ir_module = generate_ir_module(self)
        ir_module = apply_mutis_specific_transforms(ir_module, module_dir=dump_ir_dir, dump_ir=dump_ir_dir is not None)

        return ir_module.build()
