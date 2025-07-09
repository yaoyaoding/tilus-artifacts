from typing import List, Union, Optional

from hidet.ir.expr import Expr, logical_and, deref, cast
from hidet.ir.dtypes import uint32, uint16, boolean, float16, float16x2
from hidet.ir.tile.layout import BlockLayout
from hidet.ir.tile.ops.memory import Load, Store, StoreBaseOp, AtomicAdd
from hidet.ir.type import PointerType, DataType, void_p
from .registry import TileOpImpl, Buffer, register_impl
from .utils import get_type_erased_dtype, get_dtype_from_bytes


@register_impl(Load)
class LoadImpl(TileOpImpl):
    def implement(self, op: Load, args: List[Union[Buffer, Expr]], output: Buffer):
        from hidet.ir.primitives.cuda.ldst import load
        from hidet.ir.mapping import repeat_map

        ptr: Buffer = args[0]
        mask: Optional[Buffer] = args[1] if len(args) > 1 else None
        other: Optional[Buffer] = args[2] if len(args) > 2 else None
        layout = ptr.layout

        dtype: DataType = get_type_erased_dtype(ptr.dtype.base_type)

        if isinstance(layout, BlockLayout):
            axis = len(layout.shape) - 1  # the last axis is the innermost axis
            vec_size = min(layout.size_per_thread[axis] * dtype.nbytes, 16) // dtype.nbytes
            if vec_size > 1 and mask is None:
                local_shape: List[int] = layout.local_shape()
                mapping_shape: List[int] = [d if i != axis else d // vec_size for i, d in enumerate(local_shape)]

                with self.for_mapping(repeat_map(mapping_shape)) as indices:
                    local_indices = [idx if dim != axis else idx * vec_size for dim, idx in enumerate(indices)]
                    if vec_size > 4:
                        squash_vec_size = vec_size // 4
                        dtype = get_dtype_from_bytes(squash_vec_size * dtype.nbytes)
                    else:
                        squash_vec_size = 1
                    dst_addrs = []
                    local_indices_iter = local_indices.copy()
                    for i in range(vec_size // squash_vec_size):
                        dst_addrs.append(~output.var[local_indices_iter])
                        local_indices_iter[axis] += squash_vec_size
                    self.append(
                        load(
                            dtype, addr=ptr.at_local(local_indices), dst_addrs=dst_addrs, space='global', nc_cache=True
                        )
                    )
                return

        assert ptr.scope.is_register()
        if mask:
            assert mask.layout == ptr.layout
        if other:
            assert other.layout == ptr.layout

        def f_apply(local_indices, global_indices, not_duplicated):
            if mask is None:
                self.append(load(dtype, addr=ptr.at_local(local_indices), dst_addrs=[~output.at_local(local_indices)]))
            else:
                if other is None:
                    assert isinstance(ptr.dtype, PointerType)
                    value_type = ptr.dtype.base_type
                    if isinstance(value_type, PointerType):
                        other_value = void_p(0)
                    elif isinstance(value_type, DataType):
                        other_value = value_type.zero
                    else:
                        raise NotImplementedError()
                else:
                    other_value = other.at_local(local_indices)
                with self.if_then(mask.at_local(local_indices)):
                    self.append(
                        load(dtype, addr=ptr.at_local(local_indices), dst_addrs=[~output.at_local(local_indices)])
                    )
                with self.otherwise():
                    self.buffer_store(output.var, local_indices, other_value)

        self.iterate_dist_buffer_and_apply(output, f_apply)


@register_impl(StoreBaseOp)
class StoreImpl(TileOpImpl):
    def apply(self, op: StoreBaseOp, dtype, addr: Expr, src_addrs: List[Expr]):
        if isinstance(op, Store):
            from hidet.ir.primitives.cuda.ldst import store

            if dtype.nbytes == 1:
                if len(src_addrs) >= 4:
                    dtype = uint32
                    src_addrs = [src_addrs[i] for i in range(0, len(src_addrs), 4)]
                elif len(src_addrs) >= 2:
                    dtype = uint16
                    src_addrs = [src_addrs[i] for i in range(0, len(src_addrs), 2)]
            elif dtype.nbytes == 2:
                if len(src_addrs) >= 2:
                    dtype = uint32
                    src_addrs = [src_addrs[i] for i in range(0, len(src_addrs), 2)]

            dtype = get_type_erased_dtype(dtype)
            self.append(store(dtype, addr=addr, src_addrs=src_addrs, space='global'))
        elif isinstance(op, AtomicAdd):
            from hidet.ir.primitives.cuda.atomic import reduce_add
            from hidet.ir.primitives.math import make_vector

            src_values = [deref(p) for p in src_addrs]
            if len(src_addrs) % 2 == 0 and dtype == float16:
                dtype = float16x2
                src_values = [make_vector(src_values[i], src_values[i + 1]) for i in range(0, len(src_values), 2)]
                addr = cast(addr, ~dtype)
            for idx, val in enumerate(src_values):
                self.append(reduce_add(dtype, addr=cast(addr, ~dtype) + idx, src_values=[val]))
        else:
            raise NotImplementedError()

    def get_vector_size(self, dtype: Union[PointerType, DataType], args: List[Buffer]) -> int:
        ptr: Buffer = args[0]
        mask: Optional[Buffer] = args[2] if len(args) > 2 else None

        layout = ptr.layout

        if isinstance(layout, BlockLayout):
            axis = len(layout.shape) - 1  # the last axis is the innermost axis
            vec_size = min(layout.size_per_thread[axis] * dtype.nbytes, 16) // dtype.nbytes
            vec_size = min(vec_size, ptr.info.continuity[axis], ptr.info.divisibility[axis])
            if mask:
                vec_size = min(vec_size, mask.info.constancy[axis])
            return vec_size
        else:
            return 1

    def implement(self, op: StoreBaseOp, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        from hidet.ir.mapping import repeat_map

        ptr: Buffer = args[0]
        value: Buffer = args[1]
        mask: Optional[Buffer] = args[2] if len(args) > 2 else None
        dtype = value.dtype

        assert ptr.layout == value.layout
        if mask:
            assert mask.layout == ptr.layout
        layout = ptr.layout

        vec_size: int = self.get_vector_size(dtype=dtype, args=args)

        if vec_size > 1:
            axis = len(ptr.shape) - 1
            local_shape: List[int] = layout.local_shape()
            mapping_shape: List[int] = [d if i != axis else d // vec_size for i, d in enumerate(local_shape)]

            with self.for_mapping(repeat_map(mapping_shape)) as indices:
                local_indices = [idx if dim != axis else idx * vec_size for dim, idx in enumerate(indices)]
                local_indices_iter = local_indices.copy()

                src_addrs = []
                for i in range(vec_size):
                    src_addrs.append(~value.at_local(local_indices_iter))
                    local_indices_iter[axis] += 1

                if mask:
                    cond = mask.at_local(local_indices)
                else:
                    cond = boolean.true

                with self.if_then(cond):
                    self.apply(op=op, dtype=dtype, addr=ptr.at_local(local_indices), src_addrs=src_addrs)
        else:
            assert ptr.scope.is_register()

            if mask:
                assert mask.layout == ptr.layout

            def f_apply(local_indices, global_indices, not_duplicated):
                if mask:
                    assert isinstance(mask, Buffer) and ptr.layout == mask.layout
                    mask_value = mask.at_local(local_indices)
                else:
                    mask_value = True
                with self.if_then(logical_and(not_duplicated, mask_value)):
                    # the same element in the tile might be stored in multiple threads, this if statement
                    # ensures that only one thread stores the value
                    src_addrs = [~value.at_local(local_indices)]
                    self.apply(op=op, dtype=dtype, addr=ptr.var[local_indices], src_addrs=src_addrs)

            self.iterate_dist_buffer_and_apply(ptr, f_apply)
