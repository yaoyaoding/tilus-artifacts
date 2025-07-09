from typing import List
from hidet.ir.type import BaseType, void
from hidet.ir.expr import Var, Expr
from hidet.ir.tile.type import TileType
from hidet.ir.tile.expr import TileOp
from hidet.utils import same_list


class Assign(TileOp):
    def __init__(self, dst: Var, src: Expr):
        super().__init__(args=[dst, src])
        self.dst: Var = dst
        self.src: Expr = src
        assert False, "deprecated"

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        dst_type = arg_types[0]
        src_type = arg_types[1]
        assert isinstance(src_type, TileType)
        assert isinstance(dst_type, TileType)
        assert same_list(src_type.shape, dst_type.shape)
        if src_type.type.is_pointer() and dst_type.type.is_pointer():
            pass
        elif src_type.type.is_data_type() and dst_type.type.is_data_type():
            assert src_type.type == dst_type.type
        else:
            raise NotImplementedError()
        return void


def assign(dst: Var, src: Expr):
    return Assign(dst, src).make_call()
