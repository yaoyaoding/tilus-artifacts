"""
Implementation of reduce op

Given a layout that is composed with atom layouts, we use a rule-based approach to lower it to CUDA code.

1. intra-thread reduction
    reduce all the elements along the reduction dimension in the same thread to the first element in the thread along
    the reduction dimension.

    For example, if we have the following layout `spatial(2, 2).repeat(2, 2)` and want to reduce over axis=0

       | 0 1 2 3
     - + - - - -
     0 | 0 0 1 1
     1 | 0 0 1 1
     2 | 2 2 3 3
     3 | 2 2 3 3

    each thread will contains two elements:
    thread 0: a[0, 0] + a[1, 0], a[0, 1] + a[1, 1]
    thread 1: a[0, 2] + a[1, 2], a[0, 3] + a[1, 3]
    thread 2: a[2, 0] + a[3, 0], a[2, 1] + a[3, 1]
    thread 3: a[2, 2] + a[3, 2], a[2, 3] + a[3, 3]

2. intra-warp reduction
   reduce the reduced element store in different threads in the same warp to the first element of the first thread along
   the reduction dimension

   after intra-thread reduction, we have (assume we have 4 threads in a warp)
   thread 0: a[0, 0] + a[1, 0] + a[2, 0] + a[3, 0], a[0, 1] + a[1, 1] + a[2, 1] + a[3, 1]
   thread 1: a[0, 2] + a[1, 2] + a[2, 2] + a[3, 2], a[0, 3] + a[1, 3] + a[2, 3] + a[3, 3]
   (data in thread 2 and thread 3 do not needed after this step)

3. inter-warp reduction
   if we have a layout `spatial(4, 2).repeat(2, 2)` and each warp has 4 threads.

       | 0 1 2 3
     - + - - - -
     0 | 0 0 1 1
     1 | 0 0 1 1
     2 | 2 2 3 3
     3 | 2 2 3 3
     4 | 4 4 5 5
     5 | 4 4 5 5
     6 | 6 6 7 7
     7 | 6 6 7 7

    before inter-warp reduction, we have
    thread 0: a[0, 0] + a[1, 0] + a[2, 0] + a[3, 0], a[0, 1] + a[1, 1] + a[2, 1] + a[3, 1]
    thread 1: a[0, 2] + a[1, 2] + a[2, 2] + a[3, 2], a[0, 3] + a[1, 3] + a[2, 3] + a[3, 3]
    thread 4: a[4, 0] + a[5, 0] + a[6, 0] + a[7, 0], a[4, 1] + a[5, 1] + a[6, 1] + a[7, 1]
    thread 5: a[4, 2] + a[5, 2] + a[6, 2] + a[7, 2], a[4, 3] + a[5, 3] + a[6, 3] + a[7, 3]

    after inter-warp reduction, we have
    thread 0: sum[i: 0..7]a[i, 0], sum[i: 0..7]a[i, 1]
    thread 1: sum[i: 0..7]a[i, 2], sum[i: 0..7]a[i, 3]

4. broadcast back
    broadcast the reduced result stored in the first element and the first thread along the reduction dimension to all
    threads. Still use the example in step 3, we have

    thread 0: sum[i: 0..7]a[i, 0], sum[i: 0..7]a[i, 1]
    thread 1: sum[i: 0..7]a[i, 2], sum[i: 0..7]a[i, 3]
    thread 2: sum[i: 0..7]a[i, 0], sum[i: 0..7]a[i, 1]
    thread 3: sum[i: 0..7]a[i, 2], sum[i: 0..7]a[i, 3]
    thread 4: sum[i: 0..7]a[i, 0], sum[i: 0..7]a[i, 1]
    thread 5: sum[i: 0..7]a[i, 2], sum[i: 0..7]a[i, 3]
    thread 6: sum[i: 0..7]a[i, 0], sum[i: 0..7]a[i, 1]
    thread 7: sum[i: 0..7]a[i, 2], sum[i: 0..7]a[i, 3]


Implementation notes:
1. intra-thread reduction
    simple loop
2. intra-warp reduction
    use warp shuffle
3. inter-warp reduction
    use shared memory
4. broadcast back
    access shared memory
"""
import contextlib
import time
from collections import defaultdict
from typing import List, Union, Optional, Dict, Tuple, Sequence, cast, Iterable, Set

import hidet.option
from hidet.ir.builders import StmtBuilder
from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr
from hidet.ir.primitives.cuda.shfl import shfl_down_sync, shfl_up_sync
from hidet.ir.primitives.cuda.sync import syncthreads
from hidet.ir.primitives.cuda.vars import threadIdx
from hidet.ir.stmt import BufferStoreStmt
from hidet.ir.tile.layout import TileLayout
from hidet.ir.tile.ops.reduce import ReduceOp
from hidet.ir.tile.type import TileType
from hidet.ir.tools import infer_type
from hidet.transforms.tile.cuda.lower_ops.registry import TileOpImpl, Buffer, register_impl
from hidet.transforms.tile.cuda.lower_ops.sa import aggregate
from hidet.utils import iter_grid, is_power_of_two, prod


class Index:
    def __init__(self, indices: Sequence[int]):
        self.indices: Tuple[int, ...] = tuple(indices)

    def __eq__(self, other):
        return isinstance(other, Index) and self.indices == other.indices

    def __hash__(self):
        return hash(self.indices)

    def __str__(self):
        return str(list(self.indices))

    def __repr__(self):
        return str(list(self.indices))

    def __getitem__(self, item):
        return self.indices.__getitem__(item)


class LayoutMap:
    def __init__(self):
        # logical to {(worker, local), ...}
        self.map: Dict[Index, List[Tuple[int, Index]]] = defaultdict(list)
        # (worker, local) to logical
        self.rmap: Dict[Tuple[int, Index], Index] = {}

        self.layout: Optional[TileLayout] = None

    def __str__(self):
        return str(self.layout.visualize())

    @staticmethod
    def from_layout(layout: TileLayout):
        lmap = LayoutMap()

        for worker in range(layout.num_workers()):
            for local_indices in iter_grid(layout.local_shape()):
                logical_indices, _ = layout.local2logical(local_indices, worker=cast(Expr, worker))
                logical_indices = Index([int(v) for v in logical_indices])
                local_indices = Index([int(v) for v in local_indices])
                lmap.map[logical_indices].append((worker, local_indices))
                lmap.rmap[(worker, local_indices)] = logical_indices

        lmap.layout = layout

        return lmap

    def elements(self, worker: int) -> Iterable[Tuple[Index, Index]]:
        """
        returns iterator for (local, logical) pairs for each local indices
        """
        for local_indices in iter_grid(self.layout.local_shape()):
            local_indices = Index(local_indices)
            logical_indices = self.rmap[(worker, local_indices)]
            yield local_indices, logical_indices

    def workers(self, logical: Index):
        return sorted(list(set(worker for worker, _ in self.map[logical])))

    @contextlib.contextmanager
    def for_elements(self, sb: StmtBuilder, worker: int):
        with sb.for_grid(self.layout.local_shape()) as local_indices:
            logical_indices, _ = self.layout.local2logical(local_indices, worker=cast(Expr, worker))
            yield local_indices, logical_indices


@register_impl(ReduceOp)
class ReduceOpImpl(TileOpImpl):
    def implement(self, op: ReduceOp, args: List[Union[Buffer, Expr]], output: Buffer):
        self.implement_via_sa(op, args, output)

    def shuffle_scheme(self, lane_local_pairs: List[Tuple[int, Index]]) -> Optional[Tuple[Index, int, int]]:
        """
        ret: (local, num_lanes, delta)
        """
        lanes = [pair[0] for pair in lane_local_pairs]
        local_indices = [pair[1] for pair in lane_local_pairs]
        assert len(lanes) > 1
        if len(set(lanes)) != len(lanes):
            return None
        if not is_power_of_two(len(lanes)):
            return None
        if any(local_indices[i] != local_indices[0] for i in range(1, len(local_indices))):
            return None
        local = local_indices[0]
        delta = lanes[1] - lanes[0]
        if any(lanes[i] - lanes[i - 1] != delta for i in range(1, len(lanes))):
            return None
        return local, len(lanes), delta

    def get_smem_shape(self, op: ReduceOp) -> List[int]:
        dst_type = infer_type(op.make_call())
        assert isinstance(dst_type, TileType)
        layout: TileLayout = dst_type.layout
        layout_map = LayoutMap.from_layout(layout)

        logical2warps: Dict[Index, Set[int]] = defaultdict(set)
        axis = op.axis
        for logical, worker_local_pairs in layout_map.map.items():
            logical_copy = Index(logical.indices[:axis] + logical.indices[axis + 1 :])
            for worker, _ in worker_local_pairs:
                logical2warps[logical_copy].add(worker // 32)

        num_warps = set(len(warps) for warps in logical2warps.values())
        if len(num_warps) != 1:
            raise RuntimeError("Can not reduce over the layout\n{}".format(layout.visualize()))
        num_warps = num_warps.pop()
        shape: List[int] = list(layout.logical_shape())
        shape[op.axis] = num_warps
        return shape

    def request_smem_nbytes(self, op: ReduceOp) -> int:
        shape = self.get_smem_shape(op)
        dst_type: TileType = infer_type(op.make_call())
        if shape[op.axis] == 1:
            # do not need shared memory to perform inter-warp reduction
            return 0
        return prod(shape) * dst_type.type.nbytes

    def implement_via_sa(self, op: ReduceOp, args: List[Union[Buffer, Expr]], output: Buffer):
        src: Buffer = args[0]
        dst: Buffer = output

        src_layout: TileLayout = src.layout
        dst_layout: TileLayout = dst.layout
        programs = []
        src_map = LayoutMap.from_layout(src_layout)
        dst_map = LayoutMap.from_layout(dst_layout)
        axis: int = op.axis
        # print(src_map)
        # print(dst_map)
        t1 = time.time()
        smem_shape: List[int] = self.get_smem_shape(op)
        if smem_shape[op.axis] > 1:
            smem = self.make_shared_buffer(
                dtype=dst.dtype,
                shape=self.get_smem_shape(op),
                hint='reduce_smem',
                ptr=self.get_smem_ptr(op, prod(smem_shape) * dst.dtype.nbytes),
            )
        else:
            smem = None

        def read_smem(logical, warp_along_red: int):
            if op.keepdims:
                smem_indices = logical[:axis] + (warp_along_red,) + logical[axis + 1 :]
            else:
                smem_indices = logical[:axis] + logical[axis + 1 :]
            return smem.at_logical(smem_indices)

        def store_smem(logical, warp_along_read: int, value: Expr):
            if op.keepdims:
                smem_indices = logical[:axis] + (warp_along_read,) + logical[axis + 1 :]
            else:
                smem_indices = logical[:axis] + logical[axis + 1 :]
            smem_local_indices, _ = smem.layout.logical2local(smem_indices, worker=cast(Expr, threadIdx.x))
            return BufferStoreStmt(buf=smem.var, indices=smem_local_indices, value=value)

        threads = []
        for i in range(48):
            warp_id = i % self.num_warps
            lane_id = (i + i // 32) % 32
            threads.append(warp_id * 32 + lane_id)
        threads = list(range(self.num_warps * 32))

        hidet.option.internal.eager_expr_simplify(False)

        for thread in threads:
            sb = StmtBuilder()
            # 1. intra-thread reduction
            # 1.1 initialize the dst buffer to default value
            with sb.for_grid(dst_layout.local_shape()) as local_indices:
                sb.buffer_store(dst.var, indices=local_indices, value=op.kind.default_value(src.dtype))
            # 1.2 reduce all the elements along the reduction dimension in the same thread
            with src_map.for_elements(sb, worker=thread) as (local, logical):
                if op.keepdims:
                    dst_logical = logical[:axis] + [int32.zero] + logical[axis + 1 :]
                else:
                    dst_logical = logical[:axis] + logical[axis + 1 :]
                dst_local, _ = dst_layout.logical2local(dst_logical, worker=cast(Expr, thread))
                sb.buffer_store(
                    dst.var, indices=dst_local, value=op.kind.combine(dst.at_local(dst_local), src.at_local(local))
                )

            # 2. intra-warp reduction
            # 2.1 group the entries according to the logical indices for all elements in the warp
            warp = thread // 32
            logical2lanes: Dict[Index, List[Tuple[int, Index]]] = defaultdict(list)
            for lane in range(32):
                thread_id = warp * 32 + lane
                for local, logical in dst_map.elements(thread_id):
                    logical2lanes[logical].append((lane, local))
            # 2.2 for each global entry, reduce the elements stored in the same warp using warp shuffle
            use_fallback = False
            schemes: Dict[Index, Tuple[int, int]] = {}
            for logical, lane_local_pairs in logical2lanes.items():
                scheme: Optional[Tuple[Index, int, int]] = self.shuffle_scheme(lane_local_pairs)
                if scheme is None:
                    # can not use warp shuffle reduction to reduce the entries
                    # use fallback method based on shfl_sync
                    use_fallback = True
                else:
                    local, num_lanes, delta = scheme
                    if local in schemes:
                        if schemes[local] != (num_lanes, delta):
                            use_fallback = True
                    else:
                        schemes[local] = (num_lanes, delta)
            if use_fallback:
                raise NotImplementedError()
            for local, (num_lanes, delta) in schemes.items():
                # use warp shuffle reduction
                width = num_lanes * delta
                while num_lanes > 1:
                    sb.buffer_store(
                        buf=dst.var,
                        indices=local.indices,
                        value=op.kind.combine(
                            lhs=dst.at_local(local.indices),
                            rhs=shfl_down_sync(
                                mask=0xFFFFFFFF, var=dst.at_local(local.indices), delta=delta, width=width
                            ),
                        ),
                    )
                    delta <<= 1
                    num_lanes >>= 1

            # 3. inter-warp reduction
            if smem:
                lane_id = thread % 32
                warp_id = thread // 32
                # 3.1 copy the result stored in the first lane for each logical entry
                for local, logical in dst_map.elements(thread):
                    # 3.1.1 check if the current thread is the first thread for the current tile in this logical entry
                    #       and if yes, get the index of the warp among all warps along the reduction dimension
                    threads_in_entry: List[int] = dst_map.workers(logical)
                    lanes: List[int] = [t % 32 for t in threads_in_entry if t // 32 == thread // 32]
                    warps: List[int] = list(set(v // 32 for v in threads_in_entry))
                    with sb.if_then(lane_id == min(lanes)):
                        # get the no. of the current warp among all warps along the reduction dimension
                        warp_no_in_entry: int = warps.index(warp_id)
                        # 3.1.2 store the result to the shared memory
                        sb += store_smem(logical, warp_along_read=warp_no_in_entry, value=dst.at_local(local.indices))
                sb += syncthreads()
                # 3.2 perform reduce
                for local, logical in dst_map.elements(thread):
                    # 3.2.1 check whether the current thread is the first thread of the first warp along the reduction
                    #       dimension
                    threads_in_entry: List[int] = dst_map.workers(logical)
                    warps: List[int] = list(set(v // 32 for v in threads_in_entry))
                    with sb.if_then(warp_id == min(warps)):
                        # 3.2.2 perform the reduction and store the result to the shared memory
                        with sb.for_range(len(warps) - 1) as warp_reduce_idx:
                            sb.buffer_store(
                                dst.var,
                                indices=local.indices,
                                value=op.kind.combine(
                                    lhs=dst.at_local(local.indices), rhs=read_smem(logical, warp_reduce_idx + 1)
                                ),
                            )
                        # 3.2.3 store the result to the shared memory
                        sb += store_smem(logical, warp_along_read=0, value=dst.at_local(local.indices))
                sb += syncthreads()
                # 3.3 if the thread is the first lane of the warp along the reduction dimension, read the result
                #     from shared memory
                for local, logical in dst_map.elements(thread):
                    threads_in_entry: List[int] = dst_map.workers(logical)
                    threads_in_the_same_warp: List[int] = [t for t in threads_in_entry if t // 32 == thread // 32]
                    with sb.if_then(thread == min(threads_in_the_same_warp)):
                        sb.buffer_store(dst.var, indices=local.indices, value=read_smem(logical, warp_along_red=0))

            # 4. broadcast from the first lane in the warp along the reduction dimension to all the lanes in the warp
            #    along the reduction dimension
            for local, (num_lanes, delta) in schemes.items():
                width = num_lanes * delta
                while num_lanes > 1:
                    sb.buffer_store(
                        buf=dst.var,
                        indices=local.indices,
                        value=shfl_up_sync(mask=0xFFFFFFFF, var=dst.at_local(local.indices), delta=delta, width=width),
                    )
                    delta <<= 1
                    num_lanes >>= 1

            # 5. add program to the list
            programs.append(sb.finish())

        hidet.option.internal.eager_expr_simplify(True)

        t2 = time.time()
        print("generate programs time: {:.2f} seconds".format(t2 - t1))

        t1 = time.time()
        program = aggregate(programs=programs, program_ids=threads, worker=threadIdx.x)
        t2 = time.time()
        print("aggregate programs time: {:.2f} seconds".format(t2 - t1))
        self.append(program)


if __name__ == '__main__':
    from hidet.ir.tile.layout import spatial

    a = spatial(4, 2).repeat(2, 2)
    print(a.visualize())
