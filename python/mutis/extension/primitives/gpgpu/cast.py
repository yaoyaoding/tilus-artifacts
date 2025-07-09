from hidet.ir.dtypes import DataType, uint8, f32, float32
from hidet.ir.dtypes.floats_subbyte import FloatSubbyteType
from hidet.ir.expr import Expr, Var, reinterpret
from hidet.ir.builders import StmtBuilder
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.utils import initialize


def register_float_cast_functions(dtype: FloatSubbyteType):
    from hidet.lang import attrs, script  # pylint: disable=import-outside-toplevel
    from hidet.lang.types import uint32, int32

    nbits = dtype.nbits
    exponent_nbits = dtype.exponent_nbits
    mantissa_nbits = dtype.mantissa_nbits

    assert nbits == 1 + exponent_nbits + mantissa_nbits
    assert mantissa_nbits >= 1 and exponent_nbits >= 1 and nbits <= 8

    def pow2_of_float_as_uint32(p: int) -> int:
        return (p + 127) << 23

    @script
    def cast_from_f32_(src: f32) -> uint8:
        attrs.func_kind = 'gpgpu_internal'
        attrs.func_name = 'cast_f32_to_{}'.format(dtype.short_name)

        src_uint32: uint32 = reinterpret(src, uint32)
        sign: uint32 = (src_uint32 >> (32 - nbits)) & (1 << (nbits - 1))
        exponents: int32 = (src_uint32 >> 23) & ((1 << 8) - 1)
        e_adjust: int32 = 128 - (1 << (exponent_nbits - 1))
        mantissa: uint32 = uint32(0)

        if exponents > e_adjust:
            mantissa = (src_uint32 & uint32((1 << 23) - 1)) >> (23 - mantissa_nbits)
            exponents = ((exponents - e_adjust) & ((1 << exponent_nbits) - 1)) << mantissa_nbits
        elif exponents + mantissa_nbits <= e_adjust:
            mantissa = uint32(0)
            exponents = int32(0)
        else:
            mantissa = ((src_uint32 & uint32((1 << 23) - 1)) | uint32(0x800000)) >> (
                24 - mantissa_nbits + e_adjust - exponents
            )
            exponents = int32(0)

        return uint8(sign | exponents | mantissa)

    @script
    def cast_to_f32_(src: uint8) -> f32:
        attrs.func_kind = 'gpgpu_internal'
        attrs.func_name = 'cast_{}_to_f32'.format(dtype.short_name)

        sign: uint32 = (src & uint8(1 << (nbits - 1))) << (32 - nbits)
        exponent_mantissa: uint32 = (src & uint8((1 << (nbits - 1)) - 1)) << (23 - mantissa_nbits)
        dst_uint32: uint32 = sign | exponent_mantissa
        dst_f32: f32 = reinterpret(dst_uint32, float32)
        e_adjust_pow_uint32: uint32 = uint32(pow2_of_float_as_uint32(128 - (1 << (exponent_nbits - 1))))
        dst_f32 = dst_f32 * reinterpret(e_adjust_pow_uint32, float32)

        return dst_f32

    functions = [cast_from_f32_, cast_to_f32_]

    for func in functions:
        register_primitive_function(func.name, func)


@initialize()
def register_functions():
    from hidet.ir.dtypes import float8_e5m2, float8_e4m3
    from hidet.ir.dtypes import float7_e5m1, float7_e4m2, float7_e3m3, float7_e2m4
    from hidet.ir.dtypes import float6_e4m1, float6_e3m2, float6_e2m3
    from hidet.ir.dtypes import float5_e3m1, float5_e2m2
    from hidet.ir.dtypes import float4_e2m1
    from hidet.ir.dtypes import float3_e1m1

    for dtype in [
        float8_e5m2,
        float8_e4m3,
        float7_e5m1,
        float7_e4m2,
        float7_e3m3,
        float7_e2m4,
        float6_e4m1,
        float6_e3m2,
        float6_e2m3,
        float5_e3m1,
        float5_e2m2,
        float4_e2m1,
        float3_e1m1,
    ]:
        register_float_cast_functions(dtype)


def cast_subbyte_float_from_f32(src: Expr, dst_dtype: DataType):
    """
    Cast f32 to a sub-byte float number (represented in the low bits of uint8).
    """
    func_name = 'cast_f32_to_{}'.format(dst_dtype.short_name)
    return call_primitive_func(func_name, [src])


def cast_subbyte_float_to_f32(src: Expr, src_dtype: DataType):
    """
    Cast a sub-byte float number (represented in the low bits of uint8) to f32.
    """
    func_name = 'cast_{}_to_f32'.format(src_dtype.short_name)
    return call_primitive_func(func_name, [src])
