from typing import List, Optional, Union, Tuple

from hidet.ir.expr import Expr
from hidet.ir.tile.expr import TileOp
from hidet.ir.type import BaseType, void
from .smem import ProcedureOp


class DebugPrint(TileOp):
    def __init__(
        self,
        x: Expr,
        header: Optional[str] = None,
        program_id: Union[int, Tuple[int, ...]] = 0,
        fmt: Optional[str] = None,
        verbose=False,
    ):
        super().__init__(args=[x], attrs={'fmt': fmt, 'program_id': program_id, 'header': header, 'verbose': verbose})
        self.x: Expr = x
        self.header: Optional[str] = header
        self.program_id: Tuple[int, int, int] = self._normalize_program_id(program_id)
        self.fmt: Optional[str] = fmt
        self.verbose: bool = verbose

    @staticmethod
    def _normalize_program_id(program_id):
        if isinstance(program_id, int):
            return program_id, 0, 0
        elif isinstance(program_id, (tuple, list)):
            program_id = tuple(program_id)
            if len(program_id) > 3:
                raise ValueError(f'Invalid program_id: {program_id}')
            return tuple(program_id + (0,) * (3 - len(program_id)))
        else:
            raise ValueError(f'Invalid program_id: {program_id}')

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        return void


class DebugSyncThreads(ProcedureOp):
    def __init__(self):
        super().__init__(attrs={})


def debug_print(x: Expr, header: Optional[str] = None, program_id: int = 0, fmt: Optional[str] = None, verbose=False):
    return DebugPrint(x, header, program_id, fmt, verbose).make_call()


def debug_sync():
    return DebugSyncThreads().make_call()
