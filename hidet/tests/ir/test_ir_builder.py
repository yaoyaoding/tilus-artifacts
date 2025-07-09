import pytest
import hidet
from hidet.ir.expr import var
from hidet.ir.module import IRModule
from hidet.ir.builders import FunctionBuilder
from hidet.ir.primitives import printf
from hidet.testing import capture_stdout


def test_function_builder():
    from hidet.ir.stmt import ReturnStmt
    from hidet.ir.dtypes import int32

    with FunctionBuilder(name='launch', kind='public', ret_type=int32) as fb:
        i = var('i', dtype='int32')
        fb.extend_params([i])
        with fb.if_then(i == 0):
            fb.ret(0)
        with fb.else_if(i == 1):
            fb.ret(1)
        with fb.else_if(i == 2):
            fb.ret(2)
        with fb.otherwise():
            fb.ret(3)

    module = IRModule(functions={'launch': fb.get()})

    compiled_module = module.build()

    for i, desired_output in zip([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 3, 3]):
        assert compiled_module(i) == desired_output


if __name__ == '__main__':
    pytest.main([__file__])
