from typing import List, Union, Optional, Tuple

from hidet.ir.dtypes import int32, uint32, float16
from hidet.ir.expr import Expr, if_then_else, cast, is_true, deref
from hidet.ir.mapping import row_spatial, col_spatial
from hidet.ir.primitives.cuda import threadIdx
from hidet.ir.primitives.cuda.cp_async import cp_async_commit_group, cp_async_wait_group
from hidet.ir.primitives.cuda.mma import ldmatrix
from hidet.ir.primitives.cuda.sync import syncthreads
from hidet.ir.tile.layout import BlockLayout, MmaDotOperandLayout, MmaDotOperandSharedLayout, MmaOutputLayout
from hidet.ir.tile.ops.smem import AllocTensor, InsertSliceAsync, AsyncCommitGroup, AsyncWait, ExtractSlice
from hidet.ir.type import DataType
from hidet.ir.type import sizeof
from hidet.transforms.tile import annotations
from hidet.utils import is_power_of_two, prod
from .registry import TileOpImpl, Buffer, register_impl
from .utils import get_type_erased_dtype


@register_impl(AllocTensor)
class AllocTensorImpl(TileOpImpl):
    def request_smem_nbytes(self, op: AllocTensor) -> int:
        local_shape: List[int] = op.layout.local_shape()
        return prod(local_shape) * sizeof(op.dtype)

    def implement(self, op: AllocTensor, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        if annotations.global_offset not in op.annotations:
            op.annotations[annotations.global_offset] = 0
        smem_ptr = cast(self.get_smem_ptr(op, nbytes=prod(output.shape) * sizeof(output.dtype)), ~output.dtype)
        self.assign(output.var, smem_ptr)


@register_impl(InsertSliceAsync)
class InsertSliceAsyncImpl(TileOpImpl):
    def implement_for_block_layout(
        self,
        output: Buffer,
        ptr: Buffer,
        dst: Buffer,
        index: Expr,
        mask: Optional[Buffer],
        other: Optional[Buffer],
        insert_axis: int,
        dtype: DataType,
    ) -> bool:
        from hidet.utils.py import iter_grid
        from hidet.ir.primitives.cuda.cp_async import cp_async

        if not isinstance(ptr.layout, BlockLayout) or other is not None:
            return False

        axis: int = len(ptr.layout.shape) - 1  # the last axis is the innermost axis

        # the number of elements loaded by each thread
        vec_size: int = ptr.layout.size_per_thread[axis]

        # we need to make sure that the ptr and optional mask/other are contiguous in the axis
        vec_size = min(vec_size, ptr.info.continuity[axis], ptr.info.divisibility[axis])
        if mask:
            vec_size = min(vec_size, mask.info.constancy[axis])
        if other:
            vec_size = min(vec_size, other.info.constancy[axis])

        # each thread can load at most 16 bytes (128 bits) at a time
        vec_size = min(vec_size, 16 // dtype.nbytes)

        # we also need to make sure the shared memory write satisfy the alignment requirement
        while vec_size > 0:
            cp_size = vec_size * dtype.nbytes
            logical_shape: List[int] = dst.layout.logical_shape()
            logical_shape[-1] //= vec_size

            for logical_indices in iter_grid(logical_shape):
                logical_indices = list(logical_indices)
                logical_indices[-1] *= vec_size
                local_index, _ = dst.layout.logical2local(logical_indices, worker=int32.zero)
                assert len(local_index) == 1
                if local_index[0] % cp_size != 0:
                    # the alignment requirement does not satisfy
                    vec_size //= 2
                    break
            else:
                break

        if vec_size == 0:
            return False

        # get the cp size per thread in the unit of bytes
        cp_size = vec_size * dtype.nbytes

        assert is_power_of_two(cp_size)
        if cp_size not in [4, 8, 16]:
            return False

        # cp.async requires cp_size in [4, 8, 16] bytes
        local_shape: List[int] = ptr.layout.local_shape()
        assert local_shape[axis] % vec_size == 0
        local_shape = [e if i != axis else e // vec_size for i, e in enumerate(local_shape)]

        # if dst.dtype.name == 'int8':
        #     from hidet.utils.py import iter_grid
        #     num_threads = self.num_warps * 32
        #     for i in range(num_threads):
        #         for local_indices in iter_grid(local_shape):
        #             local_indices = list(local_indices)
        #             local_indices[axis] = local_indices[axis] * vec_size
        #             logical_indices, not_duplicated = ptr.layout.local2logical(local_indices, worker=i)
        #             logical_indices = logical_indices[:insert_axis] + [0] + logical_indices[insert_axis:]
        #             local = dst.layout.logical2local(logical_indices=logical_indices, worker=0)
        #             print('thread {} logical {} local {} ({})'.format(i, logical_indices, local[0][0], local[0][0]%16))
        #     exit()

        with self.for_grid(local_shape) as local_indices:
            local_indices[axis] = local_indices[axis] * vec_size
            logical_indices, not_duplicated = ptr.layout.local2logical(local_indices)
            logical_indices = logical_indices[:insert_axis] + [index] + logical_indices[insert_axis:]
            with self.if_then(not_duplicated):
                self.append(
                    cp_async(
                        dst=~dst.at_logical(logical_indices),
                        src=ptr.at_local(local_indices),
                        cp_size=cp_size,
                        src_size=if_then_else(mask.at_local(local_indices), cp_size, 0) if mask else None,
                    )
                )
        self.assign(output.var, dst.var)
        return True

    def implement(self, op: InsertSliceAsync, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        from hidet.ir.primitives.cuda.cp_async import cp_async
        from hidet.ir.primitives.cuda.ldst import load

        ptr: Buffer = args[0]
        dst: Buffer = args[1]
        index: Expr = args[2]
        mask: Optional[Buffer] = args[3] if len(args) > 3 else None
        other: Optional[Buffer] = args[4] if len(args) > 4 else None
        insert_axis: int = op.axis
        dtype: DataType = get_type_erased_dtype(ptr.dtype.base_type)

        if mask:
            assert mask.layout == ptr.layout
        if other:
            assert other.layout == ptr.layout

        if self.implement_for_block_layout(output, ptr, dst, index, mask, other, insert_axis, dtype):
            return

        def f_apply(local_indices, global_indices, not_duplicated):
            global_indices = global_indices[:insert_axis] + [index] + global_indices[insert_axis:]
            with self.if_then(not_duplicated):
                if dtype.nbytes < 4 or (mask is not None and other is not None):
                    if mask is None:
                        self.append(
                            load(dtype, addr=ptr.at_local(local_indices), dst_addrs=[~dst.at_logical(global_indices)])
                        )
                    else:
                        if other is None:
                            other_value: Expr = dtype.zero
                        else:
                            other_value: Expr = other.at_local(local_indices)
                        with self.if_then(mask.at_local(local_indices)):
                            self.append(
                                load(
                                    dtype, addr=ptr.at_local(local_indices), dst_addrs=[~dst.at_logical(global_indices)]
                                )
                            )
                        with self.otherwise():
                            self.logical_store(dst, global_indices, other_value)
                else:
                    if mask is not None:
                        src_size = if_then_else(mask.at_local(local_indices), dtype.nbytes, 0)
                    else:
                        src_size = None
                    self.append(
                        cp_async(
                            dst=~dst.at_logical(global_indices),
                            src=ptr.at_local(local_indices),
                            cp_size=dtype.nbytes,
                            src_size=src_size,
                        )
                    )

        self.iterate_dist_buffer_and_apply(ptr, f_apply)
        self.assign(output.var, dst.var)


@register_impl(AsyncCommitGroup)
class AsyncCommitGroupImpl(TileOpImpl):
    def implement(self, op: AsyncCommitGroup, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        self.append(cp_async_commit_group())


@register_impl(AsyncWait)
class AsyncWaitImpl(TileOpImpl):
    def implement(self, op: AsyncWait, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        self.append(cp_async_wait_group(op.n))
        self.append(syncthreads())


@register_impl(ExtractSlice)
class ExtractSliceImpl(TileOpImpl):
    def get_task_mappings(
        self, m_size: int, n_size: int, load_m: int, load_n: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        assert m_size % load_m == 0 and n_size % load_n == 0
        assert [is_power_of_two(v) for v in [m_size, n_size, load_m, load_n]]
        count_m: int = m_size // load_m
        count_n: int = n_size // load_n

        if count_n == 1:
            inst_m = min(count_m, 4) * load_m
            inst_n = load_n
        elif count_n == 2:
            inst_m = min(count_m, 2) * load_m
            inst_n = load_n
        else:
            assert count_n % 4 == 0
            inst_m = 1 * load_m
            inst_n = 4 * load_n

        return (m_size // inst_m, n_size // inst_n), (inst_m // load_m, inst_n // load_n)

    def implement(self, op: ExtractSlice, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        src: Buffer = args[0]
        index: Expr = args[1]
        assert src.scope.is_shared(), "ExtractSlice operator only supports shared memory buffer as source"

        if output.scope.is_shared():
            if op.extent == 1:
                indices: List[Expr] = [int32(0) for _ in range(len(output.shape))]
                indices = indices[: op.axis] + [index] + indices[op.axis :]
            else:
                indices: List[Expr] = [int32(0) for _ in range(len(output.shape))]
                indices[op.axis] = index
            self.assign(output.var, ~src.at_logical(indices))
        elif output.scope.is_register():
            indices: List[Expr] = [int32(0) if i != op.axis else index for i in range(len(output.shape))]
            if (
                isinstance(output.layout, MmaDotOperandLayout)
                and isinstance(src.layout, MmaDotOperandSharedLayout)  # this guarantees that src is 16-bytes contiguous
                and output.layout.op_idx == src.layout.mma_operand.op_idx
                and is_true(op.start % (16 // output.dtype.nbytes) == 0)  # 16 bytes aligned and contiguous
                and src.dtype.nbytes == 2
            ):
                mma: MmaOutputLayout = output.layout.mma
                op_idx: int = output.layout.op_idx
                if op_idx == 0:  # A
                    load_m = 8
                    load_n = 16 // mma.config.in_dtype.nbytes
                    (outer_m, outer_n), (inner_m, inner_n) = self.get_task_mappings(
                        m_size=mma.repeat_m * mma.inst_m, n_size=mma.repeat_k * mma.inst_k, load_m=load_m, load_n=load_n
                    )
                    warp_offset_m, _ = row_spatial(mma.warps_m, mma.warps_n).map(threadIdx.x // 32)
                    warp_offset_m = warp_offset_m * mma.repeat_m * mma.inst_m
                    warp_offset_n = 0
                    trans = False
                else:  # B
                    load_m = 16 // mma.config.in_dtype.nbytes
                    load_n = 8
                    (outer_m, outer_n), (inner_m, inner_n) = self.get_task_mappings(
                        m_size=mma.repeat_k * mma.inst_k, n_size=mma.repeat_n * mma.inst_n, load_m=load_m, load_n=load_n
                    )
                    _, warp_offset_n = row_spatial(mma.warps_m, mma.warps_n).map(threadIdx.x // 32)
                    warp_offset_m = 0
                    warp_offset_n = warp_offset_n * mma.repeat_n * mma.inst_n
                    trans = True

                inst_m = inner_m * load_m
                inst_n = inner_n * load_n
                offset_i, offset_j = row_spatial(inner_m, inner_n).spatial(load_m, 1).map(threadIdx.x % 32)
                with self.for_grid([outer_m, outer_n]) as (oi, oj):
                    smem_addr: Expr = ~src.at_logical(
                        [
                            indices[0] + warp_offset_m + oi * inst_m + offset_i,
                            indices[1] + warp_offset_n + oj * inst_n + offset_j * load_n,
                        ]
                    )
                    if op_idx == 0:
                        ti, tj = row_spatial(8, 4).map(threadIdx.x % 32)
                        tj = tj * (4 // mma.config.in_dtype.nbytes)
                    else:
                        ti, tj = col_spatial(4, 8).map(threadIdx.x % 32)
                        ti = ti * (4 // mma.config.in_dtype.nbytes)
                    regs = [
                        deref(
                            cast(
                                ~output.at_logical(
                                    [
                                        warp_offset_m + oi * inst_m + ii * load_m + ti,
                                        warp_offset_n + oj * inst_n + ij * load_n + tj,
                                    ]
                                ),
                                ~uint32,
                            )
                        )
                        for ii in range(inner_m)
                        for ij in range(inner_n)
                    ]
                    self.append(ldmatrix(regs=regs, smem_addr=smem_addr, trans=trans))
            else:

                def f_compute(local_indices, global_indices, not_duplicated):
                    global_indices[op.axis] = global_indices[op.axis] + op.start
                    return src.at_logical(global_indices)

                self.iterate_dist_buffer_and_compute(output, f_compute)
        else:
            raise NotImplementedError()
