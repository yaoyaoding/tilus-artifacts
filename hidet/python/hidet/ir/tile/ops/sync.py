from typing import List

from hidet.ir.type import BaseType, void
from .smem import ProcedureOp


class SyncThreads(ProcedureOp):
    def __init__(self):
        super().__init__(attrs={})

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        return void
