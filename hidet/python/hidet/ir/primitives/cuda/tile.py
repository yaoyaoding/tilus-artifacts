from hidet.ir.dtypes import int32
from hidet.ir.expr import Call
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.ir.type import FuncType, void_p
from hidet.utils import initialize


@initialize()
def register_functions():
    register_primitive_function('cuda_alloc_shared', func_or_type=FuncType(param_types=[int32], ret_type=void_p))


def alloc_shared(nbytes: int) -> Call:
    return call_primitive_func('cuda_alloc_shared', [int32(nbytes)])
