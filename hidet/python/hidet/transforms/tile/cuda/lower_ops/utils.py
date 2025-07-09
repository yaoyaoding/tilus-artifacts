from hidet.ir.dtypes import uint8, uint16, uint32, uint64
from hidet.ir.type import DataType, sizeof, BaseType


def get_type_erased_dtype(elem_type: BaseType) -> DataType:
    # get the type-erased data type of the loaded element
    nbits: int = sizeof(elem_type) * 8
    nbits2dtype = {8: uint8, 16: uint16, 32: uint32, 64: uint64}
    return nbits2dtype[nbits]


def get_dtype_from_bytes(nbytes: int) -> DataType:
    assert nbytes in [1, 2, 4, 8]
    nbits2dtype = {8: uint8, 16: uint16, 32: uint32, 64: uint64}
    return nbits2dtype[nbytes * 8]
