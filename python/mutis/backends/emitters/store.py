from __future__ import annotations

import functools
from typing import Union, Dict, Any, List, cast, Optional

from hidet.ir.expr import Var, Expr, logical_and, index_vars
from hidet.ir.utils.index_transform import index_multiply, index_sum, index_add, index_divide
from hidet.ir.primitives.cuda.vars import threadIdx
from mutis.backends.codegen import register_emitter, BaseEmitter, NotSupportedEmitter
from mutis.ir.graph import Operator, Tensor
from mutis.ir.layout import (
    Layout,
    AtomLayout,
    simplify,
    squeeze,
    spatial,
    repeat,
    divide,
    get_composition_chain,
    compose_chain,
    identity,
)
from mutis.ir.schedule import TensorTile
from mutis.ir.analyzers import analyze_info, TensorInfo
from mutis.vm.ir.builder import VirtualMachineBuilder
from mutis.vm.ir.inst import Instruction, StoreGlobalInst, StoreSharedInst
from mutis.vm.ir.stmt import Stmt
from mutis.vm.ir.value import RegisterValue, SharedLayout
from mutis.ops.ldst import Store
from mutis.utils import gcd, same_list, prod, factorize_decomposition


@register_emitter(Store, priority=0)
class DirectStoreEmitter(BaseEmitter):
    def emit(self) -> StoreGlobalInst:
        store: Store = cast(Store, self.op)
        x = store.get_input(0)

        def f_offset(axes: List[Var]) -> Expr:
            tile_offsets = self.codegen.tensor2tile[x].tile_offsets()
            indices = index_add(tile_offsets, axes)
            return index_sum(index_multiply(indices, store.strides)) + store.offset

        def f_mask(axes: List[Var]) -> Expr:
            tile_offsets = self.codegen.tensor2tile[x].tile_offsets()
            indices = index_add(tile_offsets, axes)
            return functools.reduce(logical_and, [a < b for a, b in zip(indices, x.shape)])

        inst = StoreGlobalInst.create(
            x=self.tensor2value[x].as_register_value(), ptr=store.ptr, f_offset=f_offset, f_mask=f_mask
        )
        return inst


@register_emitter(Store, priority=0)
class DefaultStoreEmitter(DirectStoreEmitter):
    pass


@register_emitter(Store, priority=3)
class DirectVectorizedStoreEmitter(DirectStoreEmitter):
    """
    If the layout is good (e.g., the store can be coalesced), we directly store the data to global memory.
    """

    def __init__(self, codegen, op: Operator, variant: Dict[str, Any]):
        super().__init__(codegen, op, variant)
        self.check()

    def check(self):
        x_tensor: Tensor = self.op.inputs[0]
        op: Store = cast(Store, self.op)
        layout: Layout = self.tensor2layout[x_tensor]
        local_axis, thread_axis = index_vars(2)
        tile: TensorTile = self.tensor2tile[x_tensor]
        global_indices = layout.local2global(local_index=local_axis, worker=thread_axis)
        tile_offsets: List[Expr] = tile.tile_offsets()
        global_indices = index_add(tile_offsets, global_indices)

        offset = index_sum(index_multiply(global_indices, op.strides)) + op.offset
        mask = logical_and(*[a < b for a, b in zip(global_indices, x_tensor.shape)])

        offset_info = analyze_info(
            shape=[layout.local_size, layout.num_workers], axes=[local_axis, thread_axis], var2info={}, expr=offset
        )
        mask_info = analyze_info(
            shape=[layout.local_size, layout.num_workers], axes=[local_axis, thread_axis], var2info={}, expr=mask
        )

        if offset_info[1].continuity < 32:
            raise NotSupportedEmitter()


@register_emitter(Store, priority=1)
class SharedMemoryStagedStoreEmitter(BaseEmitter):

    supports_layouts = [
        # (nbytes, layout)
        (2, spatial(8, 4).repeat(1, 2))
    ]

    def __init__(self, codegen, op: Store, variant: Dict[str, Any]):
        super().__init__(codegen, op, variant)
        self.dtype = op.inputs[0].elem_type
        self.maximum_smem_nbytes: int = 16 * 1024  # allow to use 16 KiB of shared memory

        self.inner: Optional[Layout] = None
        self.outer: Optional[Layout] = None
        self.shared_layout: Optional[SharedLayout] = None
        self.store_layout: Optional[Layout] = None
        self.determine_shared_layout()
        self.determine_store_layout()

    def determine_shared_layout(self):
        # normalize the layout into 2 dims
        layout: Layout = self.tensor2layout[self.op.inputs[0]]

        if len(layout.shape) < 2:
            raise NotSupportedEmitter()
        if len(layout.shape) > 2:
            if any(s != 1 for s in layout.shape[:-2]):
                raise NotSupportedEmitter()
            layout = simplify(squeeze(layout, dims=list(range(len(layout.shape) - 2))))
        if self.dtype.is_subbyte():
            raise NotSupportedEmitter()

        outer: Optional[Layout] = None
        inner: Optional[Layout] = None
        for nbytes, atom_layout in SharedMemoryStagedStoreEmitter.supports_layouts:
            if nbytes != self.dtype.nbytes:
                continue
            outer = divide(layout, atom_layout)
            if outer is not None:
                inner = atom_layout
                break
        else:
            raise NotSupportedEmitter()
        outer_inner = layout

        outer_chain: List[Layout] = get_composition_chain(outer)

        while len(outer_chain) > 0:
            layout = outer_chain[-1]
            if not isinstance(layout, AtomLayout):
                break

            if layout.num_workers == 1 or layout.local_size == 1:
                shape = list(layout.shape)
                if prod(shape) * prod(inner.shape) * self.dtype.nbytes <= self.maximum_smem_nbytes:
                    inner = layout * inner
                    outer_chain.pop()
                    continue
                else:
                    split_shape = [1, 1]
                    ranks = layout.ranks if layout.num_workers == 1 else layout.worker_ranks
                    for dim in sorted(range(len(layout.shape)), key=lambda i: ranks[i], reverse=True):
                        factors = factorize_decomposition(shape[dim])

                        while (
                            factors
                            and factors[0] * prod(split_shape) * prod(inner.shape) * self.dtype.nbytes
                            <= self.maximum_smem_nbytes
                        ):
                            split_shape[dim] *= factors[0]
                            del factors[0]
                        if len(factors) != 0:
                            break
                    lhs_shape = index_divide(layout.shape, split_shape)
                    rhs_shape = split_shape
                    if layout.num_workers == 1:
                        lhs = repeat(*lhs_shape, ranks=layout.ranks)
                        rhs = repeat(*rhs_shape, ranks=layout.ranks)
                    else:
                        lhs = spatial(*lhs_shape, ranks=layout.ranks)
                        rhs = spatial(*rhs_shape, ranks=layout.ranks)
                    outer_chain[-1] = lhs
                    inner = rhs * inner
                    break
            else:
                break
        self.outer = compose_chain(outer_chain) if outer_chain else identity(rank=2)
        self.inner = inner
        assert (self.outer * self.inner).semantics_equal(outer_inner)  # check if the decomposition is correct

        vector_elements = 16 // self.dtype.nbytes
        rows, columns = [inner.shape[0], inner.shape[1] // vector_elements]
        assert rows % 8 == 0

        srepeat = SharedLayout.repeat
        scompose = SharedLayout.compose

        if columns % 8 == 0:
            # most efficient for global memory storing
            self.shared_layout = srepeat(rows, columns).swizzle(dim=1, regards_dim=0, log_step=0)
        elif columns % 4 == 0:
            self.shared_layout = scompose(
                srepeat(1, columns // 4), srepeat(rows, 4).swizzle(dim=1, regards_dim=0, log_step=1)
            )
        elif columns % 2 == 0:
            self.shared_layout = scompose(
                srepeat(1, columns // 2), srepeat(rows, 2).swizzle(dim=1, regards_dim=0, log_step=2)
            )
        else:
            # most not efficient for cp.async
            self.shared_layout = scompose(
                srepeat(1, columns), srepeat(rows, 1).swizzle(dim=1, regards_dim=0, log_step=3)
            )
        self.shared_layout = scompose(self.shared_layout, srepeat(1, vector_elements)).simplify()

    def determine_store_layout(self):
        vector_elements = 16 // self.dtype.nbytes
        rows, columns = [self.inner.shape[0], self.inner.shape[1] // vector_elements]

        if self.inner.num_workers % columns == 0:
            store_layout = spatial(self.inner.num_workers // columns, columns).repeat(1, vector_elements)
            if rows % (self.inner.num_workers // columns) == 0:
                store_layout = repeat(rows // (self.inner.num_workers // columns), 1) * store_layout
            else:
                raise NotSupportedEmitter()
        elif columns % self.inner.num_workers == 0:
            store_layout = (
                repeat(rows, columns // self.inner.num_workers)
                .spatial(1, self.inner.num_workers)
                .repeat(1, vector_elements)
            )
        else:
            raise NotSupportedEmitter()
        self.store_layout = store_layout

    def emit(self) -> Union[Stmt, Instruction]:
        store: Store = cast(Store, self.op)
        x = store.get_input(0)
        x_value = self.tensor2value[x].as_register_value()

        vb = VirtualMachineBuilder()
        buf = vb.allocate_shared(self.dtype, shared_layout=self.shared_layout)

        with vb.for_range(self.outer.local_size, unroll_factor=self.outer.local_size) as outer_local:
            with vb.for_thread_group(num_groups=self.outer.num_workers) as group_iter:
                inner_x = vb.view(x_value, layout=self.inner, local_offset=outer_local * self.inner.local_size)
                vb.store_shared(buf, src=inner_x)

                vb.syncthreads()

                outer_global: List[Expr] = self.outer.local2global(local_index=outer_local, worker=group_iter)

                def f_offset(axes: List[Var]) -> Expr:
                    tile_offsets = self.codegen.tensor2tile[x].tile_offsets()
                    pad = [0 for _ in range(len(tile_offsets) - len(axes))]
                    mini_tile_offsets = index_add(tile_offsets, pad + index_multiply(outer_global, self.inner.shape))
                    indices = index_add(mini_tile_offsets, pad + axes)
                    return index_sum(index_multiply(indices, store.strides)) + store.offset

                def f_mask(axes: List[Var]) -> Expr:
                    tile_offsets = self.codegen.tensor2tile[x].tile_offsets()
                    pad = [0 for _ in range(len(tile_offsets) - len(axes))]
                    mini_tile_offsets = index_add(tile_offsets, pad + index_multiply(outer_global, self.inner.shape))
                    indices = index_add(mini_tile_offsets, pad + axes)
                    return functools.reduce(logical_and, [a < b for a, b in zip(indices, x.shape)])

                loaded_regs = vb.load_shared(src=buf, register_layout=self.store_layout)
                vb.store_global(x=loaded_regs, ptr=store.ptr, f_offset=f_offset, f_mask=f_mask)

        vb.free_shared(buf)

        return vb.finish()
