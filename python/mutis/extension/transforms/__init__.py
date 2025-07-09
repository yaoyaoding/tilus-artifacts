from typing import Optional
from hidet.ir.module import IRModule
from .add_explicit_cast import add_explicit_cast
from .filter_assertions import filter_assertion_pass
from .lower_float8_cast import lower_float8_cast_pass
from .lower_subbyte_type import lower_subbyte_type_pass


def apply_mutis_specific_transforms(ir_module, module_dir: Optional[str] = None, dump_ir: bool = False) -> IRModule:
    import os
    from hidet.transforms.resolve_generic_primitive_function import resolve_primitive_func_pass
    from hidet.transforms import lower_with, PassContext
    from hidet.transforms.instruments.save_ir_instrument import SaveIRInstrument

    transforms = [
        resolve_primitive_func_pass(),
        add_explicit_cast(),
        lower_subbyte_type_pass(),
        lower_float8_cast_pass(),
        filter_assertion_pass(),
    ]

    instruments = []
    if dump_ir:
        instruments.append(SaveIRInstrument(out_dir=os.path.join(module_dir, 'mutis_custom_passes')))

    with PassContext(instruments=instruments):
        return lower_with(ir_module, transforms=transforms)
