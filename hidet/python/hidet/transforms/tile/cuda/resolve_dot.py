from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.dtypes import float32, float16
from hidet.ir.tile.ops.dot import Dot, SimtDot, MmaDot
from hidet.ir.tile.type import TileType
from hidet.ir.tools import TypeInfer
from hidet.transforms.base import TileFunctionPass


class ResolveDotRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()

    def visit_Dot(self, e: Dot):
        if isinstance(e, SimtDot) or isinstance(e, MmaDot):
            # this might happen when the user directly use mma_dot or simt_dot
            return super().visit_Dot(e)

        a = self.visit(e.a)
        b = self.visit(e.b)
        c = self.visit(e.c)
        a_type = self.type_infer(a)
        b_type = self.type_infer(b)
        c_type = self.type_infer(c)

        assert isinstance(a_type, TileType)
        assert isinstance(b_type, TileType)
        assert isinstance(c_type, TileType)

        a_dtype = a_type.type
        b_dtype = b_type.type
        c_dtype = c_type.type
        m, n, k = c_type.shape[0], c_type.shape[1], a_type.shape[1]

        if a_dtype == b_dtype == float16 and c_dtype in [float16, float32] and m % 8 == 0 and n % 8 == 0 and k % 8 == 0:
            return MmaDot(a, b, c)
        else:
            return SimtDot(a, b, c)


class ResolveDotPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        rewriter = ResolveDotRewriter()
        return rewriter.visit(func)


def resolve_dot_pass() -> TileFunctionPass:
    return ResolveDotPass()
