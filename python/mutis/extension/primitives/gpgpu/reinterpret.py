from hidet.ir.dtypes import DataType, registered_dtypes, uint8, f32, float32
from hidet.ir.expr import Expr, Var, cast, deref
from hidet.ir.tools import infer_type
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.utils import initialize


@initialize()
def register_functions():
    from hidet.lang import script, attrs

    for a_dtype in registered_dtypes:
        for b_dtype in registered_dtypes:
            if a_dtype == b_dtype:
                continue
            if a_dtype.is_subbyte() or b_dtype.is_subbyte():
                continue
            if a_dtype.nbytes != b_dtype.nbytes:
                continue

            @script
            def _reinterpret(expr: a_dtype) -> b_dtype:
                attrs.func_kind = 'gpgpu_internal'
                attrs.func_name = 'reinterpret_{}_as_{}'.format(a_dtype.short_name, b_dtype.short_name)

                return deref(cast(~expr, ~b_dtype))

            register_primitive_function(_reinterpret)


def reinterpret(expr: Expr, dtype: DataType) -> Expr:
    expr_type = infer_type(expr)
    if not isinstance(expr_type, DataType):
        raise ValueError('Cannot reinterpret non-data type')
    if expr_type.is_subbyte() or dtype.is_subbyte() or expr_type.nbytes != dtype.nbytes:
        raise ValueError(
            'Cannot reinterpret between sub-byte types or types with different sizes: {} to {}'.format(
                expr_type.name, dtype.name
            )
        )
    return call_primitive_func(
        func_name='reinterpret_{}_as_{}'.format(expr_type.short_name, dtype.short_name), args=[expr]
    )
