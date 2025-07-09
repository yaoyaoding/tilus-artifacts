from typing import List, Union

from hidet.ir.expr import Expr, cast
from hidet.ir.tile.type import TileType
from hidet.ir.dtypes import int32
from hidet.ir.tile.ops.creation import Create
from hidet.ir.primitives.cuda import threadIdx
from hidet.ir.utils.index_transform import index_deserialize
from hidet.ir.tools import infer_type
from hidet.utils import prod
from .registry import TileOpImpl, Buffer, register_impl


@register_impl(Create)
class ConstructImpl(TileOpImpl):
    def request_smem_nbytes(self, op: Create) -> int:
        if op.layout.num_workers() == 1:
            # shared memory layout
            ttype: TileType = infer_type(op.make_call())
            return prod(op.layout.local_shape()) * ttype.type.nbytes
        else:
            return 0

    def implement(self, op: Create, args: List[Union[Buffer, Expr]], output: Buffer):
        if output.scope.is_shared():
            num_threads = self.num_warps * 32
            num_tasks = prod(output.layout.logical_shape())
            if num_tasks < num_threads:
                raise NotImplementedError()
            else:
                assert num_tasks % num_threads == 0
                num_tasks_per_thread = num_tasks // num_threads

                self.assign(output.var, cast(self.get_smem_ptr(op, self.request_smem_nbytes(op)), ~output.dtype))
                with self.for_range(num_tasks_per_thread) as i:
                    task_id = i * num_threads + threadIdx.x
                    local_indices = index_deserialize(scalar_index=task_id, shape=output.layout.local_shape())
                    global_indices, _ = output.layout.local2logical(local_indices, worker=int32.zero)
                    value = op[global_indices]
                    self.local_store(output, indices=local_indices, value=value)
        else:
            self.iterate_dist_buffer_and_compute(
                output, lambda local_indices, global_indices, not_duplicated: op[global_indices]
            )
