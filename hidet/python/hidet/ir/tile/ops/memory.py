from typing import Union, Optional, List

from hidet.ir.expr import Expr
from hidet.ir.tile.expr import TileOp
from hidet.ir.tile.type import TileType, PointerType, tile_type
from hidet.ir.type import BaseType, DataType, void
from hidet.ir.dtypes import boolean


class Load(TileOp):
    def __init__(self, ptr: Expr, mask: Optional[Expr] = None, other: Optional[Expr] = None):
        super().__init__()
        self.ptr: Expr = ptr
        self.mask: Optional[Expr] = mask
        self.other: Optional[Expr] = other

        self.args.extend([ptr])
        if mask is not None:
            self.args.append(mask)
        if other is not None:
            assert mask is not None
            self.args.append(other)

    @staticmethod
    def _get_loaded_type(ptr_type: PointerType):
        ret_type = ptr_type.base_type
        if isinstance(ret_type, (PointerType, DataType)):
            return ret_type
        else:
            raise RuntimeError(f"Invalid type of Load: {ret_type}")

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        ptr_type = arg_types[0]
        mask_type = arg_types[1] if len(arg_types) >= 2 else None
        if isinstance(ptr_type, PointerType):
            if mask_type and not (isinstance(mask_type, DataType) and mask_type == boolean):
                raise ValueError('Expect the mask has boolean type, but got {}'.format(mask_type))
            return self._get_loaded_type(ptr_type)
        elif isinstance(ptr_type, TileType):
            assert isinstance(ptr_type.type, PointerType)
            elem_type = self._get_loaded_type(ptr_type.type)
            if mask_type and not (
                mask_type == boolean or (isinstance(mask_type, TileType) and mask_type.type == boolean)
            ):
                raise ValueError('Expect the mask has boolean type, but got {}'.format(mask_type))
            return tile_type(elem_type=elem_type, shape=ptr_type.shape, layout=ptr_type.layout)
        else:
            assert False


class StoreBaseOp(TileOp):
    def __init__(self, ptr: Expr, value: Expr, mask: Optional[Expr] = None):
        super().__init__()
        self.ptr: Expr = ptr
        self.value: Expr = value
        self.mask: Optional[Expr] = mask

        self.args.extend([ptr, value])
        if mask is not None:
            self.args.append(mask)

    def write_memory_op(self) -> bool:
        return True

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        ptr_type = arg_types[0]
        value_type = arg_types[1]
        assert isinstance(ptr_type, TileType) and isinstance(value_type, TileType)
        if not isinstance(ptr_type.type, PointerType):
            raise RuntimeError(f"Invalid type of ptr argument of store operator: {ptr_type.type}")
        if isinstance(ptr_type.type.base_type, DataType):
            if not (isinstance(value_type.type, DataType) and ptr_type.type.base_type == value_type.type):
                raise RuntimeError(
                    'Incompatible types of store operator: ptr_type = {}, value_type = {}'.format(
                        ptr_type.type, value_type.type
                    )
                )
        return void


class Store(StoreBaseOp):
    pass


class AtomicAdd(StoreBaseOp):
    pass


def load(ptr: Expr, mask: Optional[Union[Expr, bool]] = None):
    return Load(ptr, mask).make_call()


def store(ptr: Expr, value: Expr, mask: Optional[Union[Expr, bool]] = None):
    return Store(ptr, value, mask).make_call()


def atomic_add(ptr: Expr, value: Expr, mask: Optional[Union[Expr, bool]] = None):
    return AtomicAdd(ptr, value, mask).make_call()
