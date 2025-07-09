from typing import Union
from hidet.ir.expr import Expr
from hidet.ir.stmt import BlackBoxStmt


def memset_async(dst: Expr, value: Union[Expr, int], count: Union[Expr, int]):
    from hidet.ir.primitives.runtime import get_cuda_stream

    return BlackBoxStmt('cudaMemsetAsync({}, {}, {}, (cudaStream_t){});', dst, value, count, get_cuda_stream())
