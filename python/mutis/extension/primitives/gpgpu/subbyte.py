from hidet.ir.dtypes import DataType, uint8, f32, float32, int32, uint32, int64
from hidet.ir.dtypes.floats_subbyte import FloatSubbyteType
from hidet.ir.expr import Expr, Var, reinterpret
from hidet.ir.builders import StmtBuilder
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.utils import initialize


def get_func_name(nbits: int) -> str:
    return 'load_subbyte_uint{}b'.format(nbits)


@initialize()
def register_functions():
    from hidet.lang import attrs, script

    for nbits in range(1, 8):

        @script
        def _load_subbyte(uint8_ptr: ~uint8, input_bit_offset: int32, index: int32) -> uint8:
            attrs.func_kind = 'gpgpu_internal'
            attrs.func_name = get_func_name(nbits)

            bit_offset = input_bit_offset + int64(index) * nbits
            uint8_ptr = uint8_ptr + bit_offset // 8
            bit_offset = int32(bit_offset % 8)
            first_mask: uint32 = (uint32((1 << nbits) - 1) << bit_offset) & uint32(0xFF)
            first_byte: uint8 = uint8_ptr[0]
            value = (first_byte & first_mask) >> bit_offset

            if bit_offset > 8 - nbits:
                second_mask: uint32 = (uint32(1) << (nbits - 8 + bit_offset)) - uint32(1)
                second_byte: uint8 = uint8_ptr[1]
                value = value | ((second_byte & second_mask) << (8 - bit_offset))

            return value

        register_primitive_function(_load_subbyte)


def load_subbyte(uint8_ptr: Expr, bit_offset: Expr, index: Expr, nbits: int) -> Expr:
    """
    Load a subbyte element from memory.

    The start bit of the subbyte element is given by uint8_ptr and bit_offset:

        bit_addr = uint8_ptr * 8 + bit_offset

    where uint8_ptr is the address of the byte in memory and bit_offset is the bit offset within the byte.

    We will load the subbyte where its bits are located at [bit_addr + index * nbits, bit_addr + (index + 1) * nbits).

    The loaded bits will be stored at the low bits of the returned uint8 value.

    Parameters
    ----------
    uint8_ptr: Expr
        The address of the byte in memory.

    bit_offset: Expr
        The bit offset within the byte.

    index: Expr
        The index of the subbyte element.

    nbits: int
        The number of bits of the subbyte element.

    Returns
    -------
    ret: Expr
        An uint8 value where the loaded bits are stored at the low bits of the value.
    """
    return call_primitive_func(get_func_name(nbits), [uint8_ptr, bit_offset, index])
