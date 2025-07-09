from typing import List, Union, Optional

from hidet.ir.type import DataType
from hidet.ir.expr import Expr, logical_and
from hidet.ir.primitives import printf
from hidet.ir.tile.ops.debug import DebugPrint, DebugSyncThreads
from hidet.ir.primitives.cuda.vars import threadIdx, blockIdx
from .registry import TileOpImpl, Buffer, register_impl, TileLayout


@register_impl(DebugPrint)
class DebugPrintImpl(TileOpImpl):
    def implement(self, op: DebugPrint, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        from hidet.ir.primitives.debug import format_string_from_dtype
        from hidet.ir.dtypes import float32

        self.sync_threads()

        buffer: Buffer = args[0]
        assert buffer.scope.is_register()
        layout: TileLayout = buffer.layout
        shape = buffer.shape

        # only print for the given thread block avoid interference between different thread blocks
        with self.if_then(
            logical_and(blockIdx.x == op.program_id[0], blockIdx.y == op.program_id[1], blockIdx.z == op.program_id[2])
        ):
            # print the header if exists
            if op.header is not None:
                with self.if_then(threadIdx.x == 0):
                    self.append(printf("%s\n", op.header))
                self.sync_threads()

            # print the 1d or 2d tile
            with self.for_grid(buffer.shape) as indices:
                # print the [ and [[ if needed
                with self.if_then(threadIdx.x == 0):
                    if len(shape) == 0:
                        pass
                    else:
                        if len(shape) == 1:
                            with self.if_then(indices[0] == 0):
                                self.append(printf('['))
                        else:
                            with self.if_then(logical_and(indices[-2] == 0, indices[-1] == 0)):
                                self.append(printf('[['))
                            with self.else_if(indices[-1] == 0):
                                self.append(printf(' ['))
                    if op.verbose:
                        self.append(printf('{ '))
                self.sync_threads()

                # print the value at this logical indices
                local_indices, is_valid = layout.logical2local(indices)
                _, not_duplicated = layout.local2logical(local_indices)
                if op.verbose:
                    cond = is_valid
                else:
                    cond = logical_and(is_valid, not_duplicated)
                range_extent = self.num_warps * 32 if op.verbose else 1
                with self.for_range(range_extent) as desired_tid:
                    if op.verbose:
                        cond = logical_and(cond, threadIdx.x == desired_tid)
                    with self.if_then(cond):
                        value = buffer.at_local(local_indices)
                        if isinstance(buffer.dtype, DataType) and buffer.dtype.is_float():
                            value = float32(value)
                            fmt = format_string_from_dtype(float32)
                        else:
                            fmt = format_string_from_dtype(buffer.dtype)

                        # override the default format string if the user provides one
                        if op.fmt is not None:
                            fmt = op.fmt
                        if op.verbose:
                            idx_width = len(str(self.num_warps * 32))
                            idx_fmt = f'%{idx_width}d'
                            self.append(printf(f'{idx_fmt}: {fmt}  ', threadIdx.x, value))
                        else:
                            self.append(printf(f'{fmt}', value))
                    self.sync_threads()

                # print the ] and ]] if needed
                with self.if_then(threadIdx.x == 0):
                    if op.verbose:
                        self.append(printf('\b}'))
                    if len(shape) == 0:
                        pass
                    else:
                        with self.if_then(indices[-1] == shape[-1] - 1):
                            if len(shape) == 1:
                                self.append(printf(']\n'))
                            else:
                                with self.if_then(indices[-2] != shape[-2] - 1):
                                    self.append(printf(']\n'))
                                with self.otherwise():
                                    self.append(printf(']]\n'))
                                    if len(shape) > 2:
                                        self.append(printf('\n'))
                        with self.otherwise():
                            self.append(printf(', '))
                self.sync_threads()


@register_impl(DebugSyncThreads)
class DebugSyncThreadsImpl(TileOpImpl):
    def implement(self, op: DebugSyncThreads, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        self.sync_threads()
