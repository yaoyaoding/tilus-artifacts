import os.path
from typing import List, Tuple, Optional, Union, Dict, Any
import torch
import shutil
import traceback
import pprint

import hidet.cuda
from hidet.ir.dtypes import uint64x4, uint32x4, uint32x2, uint32, uint16, uint8, int32, boolean, uint4b, int4b, float16, bfloat16
from hidet.ir.primitives.cuda.vars import threadIdx, blockIdx
from hidet.ir import primitives
from hidet.ir.utils.index_transform import index_serialize, index_deserialize, index_add, index_multiply, index_divide
from hidet.ir.utils.index_transform import index_sum, index_inbound
from hidet.ir.expr import Expr, logical_and, logical_or
from hidet.ir.type import DataType
from hidet.ir.expr import Var
from mutis.types import void, constant
from mutis.jit import vm_jit, get_current_space
from mutis.ir.layout import Layout, spatial, repeat, reduce, column_repeat, auto_repeat_spatial
from mutis.ir.layout import greedy_decompose, flatten, simplify
from mutis.vm.ir.program import VirtualMachineProgram
from mutis.vm.ir.builder import VirtualMachineBuilder
from mutis.vm.ir.inst import MmaConfig, SharedLayout
from mutis.vm.ir.shared_layout import shared_repeat, shared_compose
from mutis.vm.ir.value import RegisterValue, SharedValue
from mutis.utils import cdiv, gcd, prod
from mutis.vm.ir.weight_transform import WeightLayoutTransform, WeightValueTransform, ValueSymbolicMapping
from mutis.target import get_current_target


class InvalidScheduleError(Exception):
    pass


class Config:
    def __init__(
        self,
        m_size: Optional[int],
        n_size: int,
        k_size: int,
        group_size: int,
        a_dtype: DataType,
        b_dtype: DataType,
        c_dtype: DataType,
        mma_operand_dtype: DataType,
        mma_accumulate_dtype: DataType,
    ):
        self.m_size: Optional[int] = m_size
        self.n_size = n_size
        self.k_size = k_size
        self.group_size = group_size
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.c_dtype = c_dtype
        self.mma_operand_dtype = mma_operand_dtype
        self.mma_accumulate_dtype = mma_accumulate_dtype


class Schedule:
    @staticmethod
    def generate_schedules(space: int, config: Config):
        raise NotImplementedError()

    def generate_program(self, params: List[Var]) -> VirtualMachineProgram:
        raise NotImplementedError()


class MatmulSchedule(Schedule, VirtualMachineBuilder):
    def __init__(
        self,
        config: Config,
        unroll_k: int,
        warp_spatial: Tuple[int, int, int],
        warp_repeat: Tuple[int, int, int],
        mma_config: MmaConfig,
        num_stages: int,
        split_k_factor: int,
    ):
        super().__init__()
        self.config: Config = config
        self.warp_spatial: Tuple[int, int, int] = warp_spatial
        self.warp_repeat: Tuple[int, int, int] = warp_repeat
        self.unroll_k: int = unroll_k
        self.mma_config: MmaConfig = mma_config
        self.num_stages: int = num_stages
        self.split_k_factor: int = split_k_factor
        self.num_warps: int = prod(warp_spatial)

        assert self.mma_config.operand_type == self.config.mma_operand_dtype
        assert self.mma_config.acc_type == self.config.mma_accumulate_dtype

        # shorthands
        self.a_dtype: DataType = self.config.a_dtype
        self.b_dtype: DataType = self.config.b_dtype
        self.c_dtype: DataType = self.config.c_dtype
        self.operand_dtype: DataType = self.config.mma_operand_dtype
        self.accumulate_dtype: DataType = self.config.mma_accumulate_dtype
        self.n_size: int = self.config.n_size
        self.k_size: int = self.config.k_size
        self.group_size: int = self.config.group_size if self.config.group_size != -1 else self.k_size
        self.with_zeros = self.b_dtype.is_unsigned_integer()
        self.channel_wise = self.config.group_size == -1

        # the innermost mma dimension conducted by thread block
        self.block_m: int = self.warp_spatial[0] * self.warp_repeat[0] * self.mma_config.m
        self.block_n: int = self.warp_spatial[1] * self.warp_repeat[1] * self.mma_config.n
        self.intra_block_k: int = (
            self.warp_spatial[2] * self.warp_repeat[2] * (self.mma_config.k * self.mma_config.vec_k)
        )
        self.block_k: int = self.unroll_k * self.intra_block_k
        if self.group_size % self.block_k != 0:
            raise InvalidScheduleError("Group size must be divisible by block_k")

        # the layouts of the innermost mma operands
        wsm, wsn, wsk = self.warp_spatial
        wrm, wrn, wrk = self.warp_repeat
        self.layout_a = reduce(spatial(wsm, wsk, wsn, ranks=[1, 0, 2]), dims=[2]).repeat(wrm, wrk) * self.mma_config.la
        self.layout_b = reduce(spatial(wsk, wsn, wsm, ranks=[0, 2, 1]), dims=[2]).repeat(wrk, wrn) * self.mma_config.lb
        self.layout_c = reduce(spatial(wsm, wsn, wsk, ranks=[0, 1, 2]), dims=[2]).repeat(wrm, wrn) * self.mma_config.lc
        self.layout_scales = reduce(self.layout_b, dims=[0])
        self.layout_zeros = self.layout_scales
        self.check_register_usage()

        # transformed load b
        self.t_b_dtype: Optional[DataType] = None
        self.o_layout_b_lhs: Optional[Layout] = None
        self.o_layout_b_rhs: Optional[Layout] = None
        self.t_layout_b_lhs: Optional[Layout] = None
        self.t_layout_b_rhs: Optional[Layout] = None
        self.t_layout_b: Optional[Layout] = None
        self.interleave_width: Optional[int] = None
        self.interleave_stride: Optional[int] = None
        self.ignore_int4b_xor: Optional[bool] = None
        self.determine_transformed_layout_b()

        # the memory loading related components
        self.smem_layout_a: SharedLayout = self.determine_shared_layout_a().simplify()
        self.smem_layout_b: SharedLayout = self.determine_shared_layout_b().simplify()
        self.smem_layout_scales: SharedLayout = shared_repeat(
            cdiv(self.block_k, self.group_size), self.block_n
        ).simplify()
        self.smem_layout_zeros: SharedLayout = shared_repeat(
            cdiv(self.block_k, self.group_size), self.block_n
        ).simplify()
        self.smem_layout_c: SharedLayout = self.determine_shared_layout_c().simplify()
        if self.group_size % self.block_k != 0 and self.block_k % self.group_size != 0:
            raise InvalidScheduleError()
        if self.k_size % self.block_k != 0:
            raise InvalidScheduleError("k_size: {}, block_k: {}".format(self.k_size, self.block_k))

        # program variables
        self.a_ptr: Optional[Var] = None
        self.b_ptr: Optional[Var] = None
        self.c_ptr: Optional[Var] = None
        self.scales_ptr: Optional[Var] = None
        self.zeros_ptr: Optional[Var] = None
        self.m_size: Optional[Var] = None

        self.semaphore: Optional[Var] = None
        self.m_blocks: Optional[Expr] = None
        self.n_blocks: Optional[Expr] = None
        self.k_blocks: Optional[Expr] = None
        self.block_k_part: Optional[Var] = None
        self.block_m_idx: Optional[Var] = None
        self.block_n_idx: Optional[Var] = None
        self.offset_m: Optional[Expr] = None
        self.offset_n: Optional[Expr] = None
        self.offset_k: Optional[Expr] = None
        self.offset_k_end: Optional[Expr] = None

        self.smem_a: Optional[SharedValue] = None
        self.smem_b: Optional[SharedValue] = None
        self.smem_scales: Optional[SharedValue] = None
        self.smem_zeros: Optional[SharedValue] = None

        self.regs_acc: Optional[RegisterValue] = None
        self.regs_a: Optional[RegisterValue] = None
        self.regs_t_b: Optional[RegisterValue] = None
        self.regs_scales: Optional[RegisterValue] = None
        self.regs_zeros: Optional[RegisterValue] = None

    def __str__(self):
        import tabulate

        configs = {}
        for k, v in self.config.__dict__.items():
            if isinstance(v, (int, tuple, str)):
                configs[k] = v
        configs["num_warps"] = str(self.num_warps)
        configs["warp_spatial"] = str(self.warp_spatial)
        configs["warp_repeat"] = str(self.warp_repeat)
        configs["unroll_k"] = str(self.unroll_k)
        configs["mma_config"] = str(self.mma_config.name)
        configs["num_stages"] = str(self.num_stages)
        configs["split_k_factor"] = str(self.split_k_factor)
        configs["block_m"] = str(self.block_m)
        configs["block_n"] = str(self.block_n)
        configs["block_k"] = str(self.block_k)
        return tabulate.tabulate(configs.items(), tablefmt="grid", headers=["key", "value"])

    def schedule_dict(self) -> Dict[str, Any]:
        return {
            "block_m": self.block_m,
            "block_n": self.block_n,
            "block_k": self.block_k,
            "k_size": self.k_size,
            "n_size": self.n_size,
            "threads": self.num_warps * 32,
            "split_k_factor": self.split_k_factor,
            "kn_blocks": cdiv(self.n_size, self.block_n) * self.split_k_factor,
            "num_stages": self.num_stages,
            "warp_spatial": self.warp_spatial,
            "warp_repeat": self.warp_repeat,
            "unroll_k": self.unroll_k,
            "mma_config": self.mma_config.name,
        }

    def check_register_usage(self):
        used_regs = (
            self.operand_dtype.nbytes
            * (
                self.layout_a.local_size
                + self.layout_b.local_size
                + self.layout_scales.local_size
                + self.layout_zeros.local_size
            )
            + self.accumulate_dtype.nbytes * self.layout_c.local_size
        ) // 4
        if used_regs >= 250:
            # used too many registers
            raise InvalidScheduleError()

    def determine_transformed_layout_b(self):
        for load_dtype in [uint32x4, uint32x2, uint32, uint16, uint8]:
            rhs_max_local_size = (
                load_dtype.nbits // self.b_dtype.nbits if load_dtype.nbits % self.b_dtype.nbits == 0 else None
            )
            lhs, rhs = greedy_decompose(layout=self.layout_b, rhs_max_local_size=rhs_max_local_size, rhs_max_workers=32)
            if self.b_dtype.nbits * rhs.local_size % load_dtype.nbits == 0:
                num = self.b_dtype.nbits * rhs.local_size // load_dtype.nbits
                self.o_layout_b_lhs = lhs
                self.o_layout_b_rhs = rhs
                self.t_b_dtype = load_dtype
                if lhs.is_simple():
                    self.t_layout_b_lhs = simplify(repeat(lhs.local_size).spatial(lhs.num_workers))
                else:
                    self.t_layout_b_lhs = flatten(lhs)
                self.t_layout_b_rhs = simplify(repeat(num).spatial(rhs.num_workers))
                self.t_layout_b = self.t_layout_b_lhs * self.t_layout_b_rhs
                assert (
                    self.t_layout_b.num_workers == self.layout_b.num_workers
                    and self.t_layout_b.local_size * self.t_b_dtype.nbits
                    == self.layout_b.local_size * self.b_dtype.nbits
                )
                return
        else:
            raise InvalidScheduleError()
        # lhs, rhs = greedy_decompose(layout=self.layout_b, rhs_max_workers=32)
        # self.o_layout_b_lhs = lhs
        # self.o_layout_b_rhs = rhs
        #
        # for load_dtype in [uint32x4, uint32x2, uint32, uint16, uint8]:
        #     if self.b_dtype.nbits * rhs.local_size % load_dtype.nbits == 0:
        #         num = self.b_dtype.nbits * rhs.local_size // load_dtype.nbits
        #         self.t_b_dtype = load_dtype
        #         self.t_layout_b_lhs = flatten(lhs)
        #         self.t_layout_b_rhs = simplify(repeat(num).spatial(rhs.num_workers))
        #         self.t_layout_b = self.t_layout_b_lhs * self.t_layout_b_rhs
        #         assert (
        #                 self.t_layout_b.num_workers == self.layout_b.num_workers
        #                 and self.t_layout_b.local_size * self.t_b_dtype.nbits == self.layout_b.local_size * self.b_dtype.nbits
        #         )
        #         return
        # else:
        #     raise InvalidScheduleError()

    @staticmethod
    def swizzle_shared_layout(dtype: DataType, m: int, n: int):
        assert dtype.nbits == 16
        assert m % 8 == n % 8 == 0
        rows, columns = m, n // 8

        if columns % 8 == 0:
            # most efficient for cp.async
            layout = shared_repeat(rows, columns).swizzle(dim=1, regards_dim=0, log_step=0)
        elif columns % 4 == 0:
            layout = shared_compose(
                shared_repeat(1, columns // 4), shared_repeat(rows, 4).swizzle(dim=1, regards_dim=0, log_step=1)
            )
        elif columns % 2 == 0:
            layout = shared_compose(
                shared_repeat(1, columns // 2), shared_repeat(rows, 2).swizzle(dim=1, regards_dim=0, log_step=2)
            )
        else:
            # most not efficient for cp.async
            layout = shared_compose(
                shared_repeat(1, columns), shared_repeat(rows, 1).swizzle(dim=1, regards_dim=0, log_step=3)
            )
        return shared_compose(layout, shared_repeat(1, 8))

    def determine_shared_layout_a(self):
        return self.swizzle_shared_layout(self.a_dtype, self.block_m, self.block_k)

    def determine_shared_layout_b(self):
        return shared_repeat(self.unroll_k * self.t_layout_b.shape[0])

    def determine_shared_layout_c(self):
        return self.swizzle_shared_layout(self.c_dtype, self.block_m, self.block_n)

    @staticmethod
    def generate_schedules(space: int, config: Config) -> List[Schedule]:
        failed_reasons = []
        schedules = []

        def append_schedule(*, config, unroll_k, warp_spatial, warp_repeat, num_stages, split_k_factor):
            try:
                mma_config = {
                    float16: MmaConfig.m16n8k16_f16_f32(vec_k=1),
                    bfloat16: MmaConfig.m16n8k16_bf16_f32(vec_k=1),
                }[config.a_dtype]
                schedules.append(
                    MatmulSchedule(
                        config=config,
                        unroll_k=unroll_k,
                        warp_spatial=warp_spatial,
                        warp_repeat=warp_repeat,
                        mma_config=mma_config,
                        num_stages=num_stages,
                        split_k_factor=split_k_factor,
                    )
                )
            except InvalidScheduleError:
                sch = {
                    'config': config.__dict__,
                    'unroll_k': unroll_k,
                    'warp_spatial': warp_spatial,
                    'warp_repeat': warp_repeat,
                    'mma_config': MmaConfig,
                    'num_stages': num_stages,
                    'split_k_factor': split_k_factor,
                }
                msg = '\n\n'.join([
                    'Schedule:\n' + pprint.pformat(sch),
                    'Traceback:\n' + traceback.format_exc()
                ])
                failed_reasons.append(msg)

        if space == 0:
            append_schedule(
                config=config,
                unroll_k=1,
                warp_spatial=(1, 4, 1),
                warp_repeat=(1, 1, 4),
                num_stages=3,
                split_k_factor=4
            )
        elif space == 1:
            for warp_spatial in [(2, 4, 1), (1, 4, 1)]:
                for warp_repeat in [(1, 2, 4), (1, 1, 4)]:
                    for unroll_k in [2, 4, 8]:
                        for num_stages in [3]:
                            for split_k_factor in [1, 2, 3, 4, 6, 8, 12, 16, 24]:
                                append_schedule(
                                    config=config,
                                    unroll_k=unroll_k,
                                    warp_spatial=warp_spatial,
                                    warp_repeat=warp_repeat,
                                    num_stages=num_stages,
                                    split_k_factor=split_k_factor
                                )
        elif space == 2:
            for warp_spatial in [
                (1, 4, 1),
                # (2, 4, 1),
                # (2, 2, 1),
            ]:
                for warp_repeat in [
                    (1, 1, 4),
                    (1, 1, 8),
                    (1, 1, 16),
                    (1, 2, 4),
                    (1, 2, 8),
                    (1, 4, 4),
                    # (2, 2, 2),
                    # (2, 1, 4),
                    # (2, 1, 8),
                    # (2, 2, 4),
                    # (2, 4, 2),
                    # (4, 2, 2),
                    # (4, 4, 1),
                ]:
                    for unroll_k in [1, 4]:
                        for num_stages in [2, 3, 4]:
                            for split_k_factor in [1, 2, 3, 4, 8, 12, 16, 32]:
                                append_schedule(
                                    config=config,
                                    unroll_k=unroll_k,
                                    warp_spatial=warp_spatial,
                                    warp_repeat=warp_repeat,
                                    num_stages=num_stages,
                                    split_k_factor=split_k_factor
                                )
        else:
            assert False

        if len(schedules) == 0:
            log_dir = os.path.abspath('./failed-schedules')
            shutil.rmtree(log_dir, ignore_errors=True)
            os.makedirs(log_dir)
            for idx, msg in enumerate(failed_reasons):
                path = os.path.join(log_dir, str(idx) + '.txt')
                with open(path, 'w') as f:
                    f.write(msg)
            raise RuntimeError('Can not find a valid schedule. Failed schedules have dump at:\n  {}'.format(log_dir))


        return schedules

    def define_regs(self):
        self.regs_acc = self.allocate(dtype=self.accumulate_dtype, layout=self.layout_c, f_init=lambda indices: 0.0)
        self.regs_a = self.allocate(dtype=self.a_dtype, layout=repeat(2, 1, 1) * self.layout_a)
        self.regs_t_b = self.allocate(dtype=self.t_b_dtype, layout=repeat(2, 1) * self.t_layout_b)
        if self.channel_wise:
            self.regs_scales = self.load_global(
                dtype=self.operand_dtype,
                layout=self.layout_scales,
                ptr=self.scales_ptr,
                f_offset=lambda axes: self.offset_n + axes[0],
                f_mask=lambda axes: self.offset_n + axes[0] < self.n_size,
            )
            if self.with_zeros:
                self.regs_zeros = self.load_global(
                    dtype=self.operand_dtype,
                    layout=self.layout_zeros,
                    ptr=self.zeros_ptr,
                    f_offset=lambda axes: self.offset_n + axes[0],
                    f_mask=lambda axes: self.offset_n + axes[0] < self.n_size,
                )
        else:
            self.regs_scales = self.allocate(dtype=self.operand_dtype, layout=repeat(2, 1) * self.layout_scales)
            if self.with_zeros:
                self.regs_zeros = self.allocate(dtype=self.operand_dtype, layout=repeat(2, 1) * self.layout_zeros)

    def define_smem(self):
        self.smem_a = self.allocate_shared(
            dtype=self.a_dtype, shared_layout=self.smem_layout_a.prepend_dim(self.num_stages).simplify()
        )
        self.smem_b = self.allocate_shared(
            dtype=self.t_b_dtype, shared_layout=self.smem_layout_b.prepend_dim(self.num_stages).simplify()
        )
        if not self.channel_wise:
            self.smem_scales = self.allocate_shared(
                dtype=self.a_dtype, shared_layout=self.smem_layout_scales.prepend_dim(self.num_stages).simplify()
            )
            if self.with_zeros:
                self.smem_zeros = self.allocate_shared(
                    dtype=self.a_dtype, shared_layout=self.smem_layout_zeros.prepend_dim(self.num_stages).simplify()
                )

    def init_offsets(self):
        self.m_blocks = cdiv(self.m_size, self.block_m)
        self.n_blocks = cdiv(self.n_size, self.block_n)
        total_k_blocks = cdiv(self.k_size, self.block_k)

        self.block_k_part, self.block_m_idx, self.block_n_idx = self.virtual_blocks(
            [self.split_k_factor, self.m_blocks, self.n_blocks]
        )
        k_blocks_per_cta = cdiv(total_k_blocks, self.split_k_factor)
        self.offset_m = self.block_m_idx * self.block_m
        self.offset_n = self.block_n_idx * self.block_n
        self.offset_k = self.block_k_part * k_blocks_per_cta * self.block_k
        self.k_blocks = (
            primitives.min((self.block_k_part + 1) * k_blocks_per_cta, total_k_blocks)
            - self.block_k_part * k_blocks_per_cta
        )
        self.offset_k_end = self.allocate_scalar(
            "offset_k_end", int32, init=primitives.min(self.offset_k + self.k_blocks * self.block_k, self.k_size)
        )
        # self.printf("[%d] offset_k_end=%d\n", blockIdx.x, self.offset_k_end)
        self.semaphore = self.allocate_global(
            "semaphore", scalar_type=~int32, nbytes=self.m_blocks * self.n_blocks * int32.nbytes, require_clean=True
        )

        self.annotate_divisibility({self.offset_k_end: self.block_k})

    def free_smem(self):
        self.free_shared(self.smem_a)
        self.free_shared(self.smem_b)
        if not self.channel_wise:
            self.free_shared(self.smem_scales)
            if self.with_zeros:
                self.free_shared(self.smem_zeros)

    def load_smem_a_from_gmem(self, offset_m: Expr, offset_k: Expr, stage: Expr):
        def f_offset(axes: List[Var]) -> Expr:
            return index_sum(index_multiply(index_add([offset_m, offset_k], axes), [self.k_size, 1]))

        def f_mask(axes: List[Var]) -> Expr:
            return logical_and(offset_m + axes[0] < self.m_size, offset_k + axes[1] < self.offset_k_end)

        smem_a = self.view_shared(x=self.smem_a, indices=[stage], layout=self.smem_layout_a)
        self.copy_async(dst=smem_a, ptr=self.a_ptr, f_offset=f_offset, f_mask=f_mask)

    def load_smem_b_from_gmem(self, offset_k: Expr, offset_n, stage: Expr):
        def f_offset(axes: List[Var]):
            unroll_k = axes[0] // self.t_layout_b.shape[0]
            outer_tile_indices = index_divide(
                [offset_k + unroll_k * self.intra_block_k, offset_n], self.o_layout_b_rhs.shape
            )
            lhs_global = (axes[0] // self.t_layout_b_rhs.shape[0]) % self.t_layout_b_lhs.shape[0]
            rhs_global = axes[0] % self.t_layout_b_rhs.shape[0]
            intra_tile_indices = index_deserialize(lhs_global, shape=self.o_layout_b_lhs.shape)
            tile_indices = index_add(outer_tile_indices, intra_tile_indices)
            tile_index = index_serialize(
                tile_indices, shape=index_divide([self.k_size, self.n_size], self.o_layout_b_rhs.shape)
            )
            tile_offset = tile_index * self.t_layout_b_rhs.shape[0]
            return tile_offset + rhs_global

        def f_mask(axes: List[Var]):
            unroll_k = axes[0] // self.t_layout_b.shape[0]
            outer_tile_offsets = [offset_k + unroll_k * self.intra_block_k, offset_n]
            lhs_global = (axes[0] // self.t_layout_b_rhs.shape[0]) % self.t_layout_b_lhs.shape[0]
            intra_tile_indices = index_deserialize(lhs_global, shape=self.o_layout_b_lhs.shape)
            intra_tile_offsets = index_multiply(intra_tile_indices, self.o_layout_b_rhs.shape)
            tile_offsets = index_add(outer_tile_offsets, intra_tile_offsets)
            return logical_and(tile_offsets[0] < self.offset_k_end, tile_offsets[1] < self.n_size)

        smem_b = self.view_shared(x=self.smem_b, indices=[stage], layout=self.smem_layout_b)
        self.copy_async(dst=smem_b, ptr=self.b_ptr, f_offset=f_offset, f_mask=f_mask, evict="evict_first")

    def load_smem_scales_from_gmem(self, offset_k: Expr, offset_n: Expr, stage: Expr):
        self.copy_async(
            dst=self.view_shared(self.smem_scales, indices=[stage], layout=self.smem_layout_scales),
            ptr=self.scales_ptr,
            f_offset=lambda axes: (offset_k + axes[0] * self.group_size) // self.group_size * self.n_size
            + offset_n
            + axes[1],
            f_mask=lambda axes: logical_and(
                offset_k + axes[0] * self.group_size < self.offset_k_end, offset_n + axes[1] < self.n_size
            ),
        )

    def load_smem_zeros_from_gmem(self, offset_k: Expr, offset_n: Expr, stage: Expr):
        self.copy_async(
            dst=self.view_shared(self.smem_zeros, indices=[stage], layout=self.smem_layout_zeros),
            ptr=self.zeros_ptr,
            f_offset=lambda axes: (offset_k + axes[0] * self.group_size) // self.group_size * self.n_size
            + offset_n
            + axes[1],
            f_mask=lambda axes: logical_and(
                offset_k + axes[0] * self.group_size < self.offset_k_end, offset_n + axes[1] < self.n_size
            ),
        )

    def load_regs_a_from_smem(self, unroll_k: Union[Expr, int], stage: Expr):
        loaded = self.load_matrix(
            src=self.view_shared(self.smem_a, indices=[stage], layout=self.smem_layout_a),
            register_layout=self.layout_a,
            offsets=[0, unroll_k * self.intra_block_k],
        )
        out = self.view(self.regs_a, layout=self.layout_a, local_offset=(unroll_k % 2) * self.layout_a.local_size)
        self.assign(output=out, x=loaded)

    def load_regs_t_b_from_smem(self, unroll_k: Union[Expr, int], stage: Expr):
        loaded = self.load_shared(
            src=self.view_shared(self.smem_b, indices=[stage], layout=self.smem_layout_b),
            register_layout=self.t_layout_b,
            offsets=[unroll_k * self.t_layout_b.shape[0]],
        )
        out = self.view(self.regs_t_b, layout=self.t_layout_b, local_offset=(unroll_k % 2) * self.t_layout_b.local_size)
        self.assign(output=out, x=loaded)

    def load_regs_scales_from_smem(self, unroll_k: Union[Expr, int], stage: Expr):
        loaded = self.load_shared(
            src=self.view_shared(
                self.smem_scales,
                indices=[stage, unroll_k * self.intra_block_k // self.group_size],
                layout=shared_repeat(self.block_n),
            ),
            register_layout=self.layout_scales,
        )
        out = self.view(
            self.regs_scales, layout=self.layout_scales, local_offset=(unroll_k % 2) * self.layout_scales.local_size
        )
        self.assign(output=out, x=loaded)

    def load_regs_zeros_from_smem(self, unroll_k: Union[Expr, int], stage: Expr):
        loaded = self.load_shared(
            src=self.view_shared(
                self.smem_zeros,
                indices=[stage, unroll_k * self.intra_block_k // self.group_size],
                layout=shared_repeat(self.block_n),
            ),
            register_layout=self.layout_zeros,
        )
        out = self.view(
            self.regs_zeros, layout=self.layout_zeros, local_offset=(unroll_k % 2) * self.layout_zeros.local_size
        )
        self.assign(output=out, x=loaded)

    def issue_weight_transform(self):
        self.set_weight_nbytes(self.b_ptr, (self.k_size * self.n_size * self.b_dtype.nbits + 7) // 8)
        self.append_weight_transform(
            self.b_ptr,
            WeightLayoutTransform(
                dtype=self.b_dtype,
                shape=[self.k_size, self.n_size],
                strides=[self.n_size, 1],
                original_layout=self.o_layout_b_rhs,
                transformed_dtype=self.t_b_dtype,
                transformed_layout=self.t_layout_b_rhs,
            ),
        )
        if self.b_dtype in [uint4b, int4b] and self.operand_dtype == float16:
            vec_elements = self.t_b_dtype.nbits // 4
            if vec_elements % 8 == 0:
                assert self.k_size * self.n_size <= 2**31 - 1
                original_vec_layout = repeat(vec_elements)
                interleaved_vec_layout = simplify(repeat(vec_elements // 8) * flatten(column_repeat(2, 4)))
                assert original_vec_layout.local_size == interleaved_vec_layout.local_size == vec_elements
                assert original_vec_layout.num_workers == interleaved_vec_layout.num_workers == 1
                self.append_weight_transform(
                    self.b_ptr,
                    WeightLayoutTransform(
                        dtype=uint4b,
                        shape=[self.k_size * self.n_size],
                        strides=[1],
                        original_layout=self.t_layout_b_rhs * original_vec_layout,
                        transformed_dtype=uint4b,
                        transformed_layout=self.t_layout_b_rhs * interleaved_vec_layout,
                    ),
                )
                self.interleave_width = 8
                self.interleave_stride = 4

                if self.b_dtype == int4b:
                    self.append_weight_transform(
                        self.b_ptr,
                        WeightValueTransform(
                            dtype=uint32,
                            shape=[self.k_size * self.n_size // 8],
                            mapping=ValueSymbolicMapping.create(x_dtype=int32, f_value=lambda x: x ^ 0x88888888),
                            reverse_mapping=ValueSymbolicMapping.create(
                                x_dtype=int4b, f_value=lambda x: x ^ 0x88888888
                            ),
                        ),
                    )
                    self.ignore_int4b_xor = True

    def lock_semaphore(self, value: Union[Expr, int]):
        """
        wait until the semaphore value is the given value
        """
        idx = self.block_m_idx * self.n_blocks + self.block_n_idx

        semaphore_value = self.allocate_scalar("semaphore_value", scalar_type=int32, init=-int32.one)

        with self.while_loop(boolean.true):
            with self.if_then(threadIdx.x == 0):
                self.assign_scalar(semaphore_value, self.load_scalar(ptr=self.semaphore + idx, sync="acquire"))
            cond = self.syncthreads_or(semaphore_value == value)
            with self.if_then(cond):
                self.brk()

    def release_semaphore(self, value: Union[Expr, int]):
        """
        increase the semaphore by setting it to a new value
        """
        idx = self.block_m_idx * self.n_blocks + self.block_n_idx
        with self.if_then(threadIdx.x == 0):
            # set the semaphore to the new value
            self.store_scalar(ptr=self.semaphore + idx, value=value, sync="release")

    def write_back_sequential(self):
        acc = self.regs_acc
        assert self.warp_spatial[2] == 1
        # compute current id for the segment on k dimension and the given (offset_m, offset_n)

        casted_acc = self.cast(acc, dtype=self.c_dtype)

        # load the previous partial results
        with self.if_then(logical_and(self.split_k_factor > 1, self.block_k_part != 0)):
            self.lock_semaphore(self.block_k_part)
            # write back the results

            prev = self.load_global(
                dtype=casted_acc.dtype,
                layout=casted_acc.layout,
                ptr=self.c_ptr,
                f_offset=lambda ij: (self.offset_m + ij[0]) * self.n_size + self.offset_n + ij[1],
                f_mask=lambda ij: logical_and(
                    (self.offset_m + ij[0]) < self.m_size, (self.offset_n + ij[1]) < self.n_size
                ),
            )
            self.add(casted_acc, prev, out=casted_acc)

        # store to global
        self.store_global(
            x=casted_acc,
            ptr=self.c_ptr,
            f_offset=lambda ij: (self.offset_m + ij[0]) * self.n_size + self.offset_n + ij[1],
            f_mask=lambda ij: logical_and((self.offset_m + ij[0]) < self.m_size, (self.offset_n + ij[1]) < self.n_size),
        )

        with self.if_then(self.split_k_factor > 1):
            self.release_semaphore((self.block_k_part + 1) % self.split_k_factor)

    def write_back_parallel(self):
        assert self.warp_spatial[2] == 1
        vec_elements = 16 // self.c_dtype.nbytes
        while vec_elements > 1 and self.block_m * self.block_n // vec_elements < self.num_warps * 32:
            vec_elements //= 2
        assert (
            self.block_n % vec_elements == 0 and self.block_m * self.block_n // vec_elements % self.num_warps * 32 == 0
        )

        acc = self.cast(self.regs_acc, dtype=self.c_dtype)

        smem_c = self.allocate_shared(dtype=self.c_dtype, shared_layout=self.smem_layout_c)
        layout_store_c = auto_repeat_spatial(
            num_threads=self.num_warps * 32, shape=[self.block_m, self.block_n // vec_elements]
        ) * repeat(1, vec_elements)
        self.store_shared(dst=smem_c, src=acc)
        self.syncthreads()
        regs_store_c = self.load_shared(src=smem_c, register_layout=layout_store_c)
        self.free_shared(smem_c)

        def f_offset(block_k_part, axes):
            return index_serialize(
                indices=[block_k_part, self.offset_m + axes[0], self.offset_n + axes[1]],
                shape=[self.split_k_factor, self.m_size, self.n_size],
            )

        def f_mask(axes):
            return logical_and(
                index_inbound(
                    indices=[self.block_m_idx * self.block_m + axes[0], self.block_n_idx * self.block_n + axes[1]],
                    shape=[self.m_size, self.n_size],
                )
            )

        if self.split_k_factor <= 1:
            self.store_global(x=regs_store_c, ptr=self.c_ptr, f_offset=lambda axes: f_offset(0, axes), f_mask=f_mask)
        else:
            parallel_write = True
            if parallel_write:
                assert self.layout_c.local_size * self.layout_c.num_workers == self.block_m * self.block_n
                gmem_c_parts = self.allocate_global(
                    "c_parts",
                    scalar_type=~self.c_dtype,
                    nbytes=prod([self.split_k_factor, self.m_size, self.n_size, self.c_dtype.nbytes]),
                    require_clean=False,
                )
                with self.if_then(self.block_k_part < self.split_k_factor - 1):
                    self.store_global(
                        x=regs_store_c,
                        ptr=gmem_c_parts,
                        f_offset=lambda axes: f_offset(self.block_k_part, axes),
                        f_mask=f_mask,
                    )
                    self.syncthreads()
                    with self.if_then(threadIdx.x == 0):
                        self.atomic_scalar(
                            ptr=self.semaphore + self.block_m_idx * self.n_blocks + self.block_n_idx,
                            op="add",
                            value=int32(1),
                        )
                with self.otherwise():
                    self.lock_semaphore(value=self.split_k_factor - 1)
                    self.release_semaphore(value=0)
                    with self.for_range(self.split_k_factor - 1) as si:
                        prev = self.load_global(
                            dtype=self.c_dtype,
                            layout=layout_store_c,
                            ptr=gmem_c_parts,
                            f_offset=lambda axes: f_offset(si, axes),
                            f_mask=f_mask,
                        )
                        self.add(prev, regs_store_c, out=regs_store_c)
                    self.store_global(
                        x=regs_store_c, ptr=self.c_ptr, f_offset=lambda axes: f_offset(0, axes), f_mask=f_mask
                    )
            else:
                with self.if_then(logical_and(self.split_k_factor > 1, self.block_k_part != 0)):
                    self.lock_semaphore(self.block_k_part)
                    prev = self.load_global(
                        dtype=acc.dtype,
                        layout=acc.layout,
                        ptr=self.c_ptr,
                        f_offset=lambda axes: f_offset(0, axes),
                        f_mask=f_mask,
                    )
                    self.add(acc, prev, out=acc)
                self.store_global(x=acc, ptr=self.c_ptr, f_offset=lambda axes: f_offset(0, axes), f_mask=f_mask)
                with self.if_then(self.split_k_factor > 1):
                    self.release_semaphore((self.block_k_part + 1) % self.split_k_factor)

    def write_back(self):
        # self.write_back_sequential()
        self.write_back_parallel()

    def load_regs_from_smem(self, unroll_k: Union[Expr, int], stage: Expr):
        self.load_regs_a_from_smem(unroll_k=unroll_k, stage=stage)
        self.load_regs_t_b_from_smem(unroll_k=unroll_k, stage=stage)
        if not self.channel_wise:
            self.load_regs_scales_from_smem(unroll_k=unroll_k, stage=stage)
            if self.with_zeros:
                self.load_regs_zeros_from_smem(unroll_k=unroll_k, stage=stage)

    def load_smem_from_gmem(self, offset_m, offset_n, offset_k, stage):
        self.load_smem_a_from_gmem(offset_m=offset_m, offset_k=offset_k, stage=stage)
        self.load_smem_b_from_gmem(offset_k=offset_k, offset_n=offset_n, stage=stage)
        if not self.channel_wise:
            self.load_smem_scales_from_gmem(offset_k=offset_k, offset_n=offset_n, stage=stage)
            if self.with_zeros:
                self.load_smem_zeros_from_gmem(offset_k=offset_k, offset_n=offset_n, stage=stage)

    def pipelined_body(self):
        self.init_offsets()
        self.define_smem()
        self.define_regs()

        with self.for_range(self.num_stages - 1, unroll_factor=-1) as stage:
            self.load_smem_from_gmem(
                offset_m=self.offset_m,
                offset_n=self.offset_n,
                offset_k=self.offset_k + stage * self.block_k,
                stage=stage,
            )
            self.copy_async_commit_group()

        self.copy_async_wait_group(self.num_stages - 2)
        self.syncthreads()

        current_stage = self.allocate_scalar("current_stage", scalar_type=int32, init=int32(0))
        preload_stage = self.allocate_scalar("preload_stage", scalar_type=int32, init=int32(self.num_stages - 1))
        with self.for_range(self.k_blocks, unroll_factor=self.num_stages) as bk:
            with self.for_range(self.unroll_k, "uk", unroll_factor=-1) as uk:
                with self.if_then(uk == 0):
                    self.load_regs_from_smem(unroll_k=0, stage=current_stage)
                with self.if_then(uk + 1 < self.unroll_k):
                    self.load_regs_from_smem(unroll_k=uk + 1, stage=current_stage)
                a = self.view(self.regs_a, layout=self.layout_a, local_offset=self.layout_a.local_size * (uk % 2))
                t_b = self.view(
                    self.regs_t_b, layout=self.t_layout_b, local_offset=self.t_layout_b.local_size * (uk % 2)
                )
                if not self.channel_wise:
                    if self.with_zeros:
                        zeros = self.view(
                            self.regs_zeros,
                            layout=self.layout_zeros,
                            local_offset=self.layout_zeros.local_size * (uk % 2),
                        )
                    scales = self.view(
                        self.regs_scales,
                        layout=self.layout_scales,
                        local_offset=self.layout_scales.local_size * (uk % 2),
                    )
                b = self.view(t_b, dtype=self.b_dtype, layout=self.layout_b)
                b = self.cast(
                    b,
                    dtype=self.operand_dtype,
                    interleave_width=self.interleave_width,
                    interleave_stride=self.interleave_stride,
                    ignore_int4b_xor=True,
                )
                if not self.channel_wise:
                    scales = self.cast(scales, dtype=self.operand_dtype)
                    if self.with_zeros:
                        zeros = self.cast(zeros, dtype=self.operand_dtype)
                        b = self.mul(self.sub(b, zeros), scales)
                    else:
                        b = self.mul(b, scales)
                self.mma_dot(
                    a=a,
                    b=b,
                    c=self.regs_acc,
                    output=self.regs_acc,
                    mma_inst=self.mma_config.name,
                    warp_spatial=self.warp_spatial,
                    warp_repeat=self.warp_repeat,
                )
                # with self.if_then(logical_and(self.block_m_idx == 0, self.block_n_idx == 0, self.block_k_part == 0)):
                #     self.print_value(msg='scales: ', value=scales)
                #     self.print_value(msg='b: ', value=b)
                #     self.print_value(msg='c: ', value=self.regs_acc)

            # preload the next stage
            self.load_smem_from_gmem(
                offset_m=self.offset_m,
                offset_n=self.offset_n,
                offset_k=self.offset_k + (bk + self.num_stages - 1) * self.block_k,
                stage=preload_stage,
            )
            self.assign_scalar(current_stage, (current_stage + 1) % self.num_stages)
            self.assign_scalar(preload_stage, (preload_stage + 1) % self.num_stages)
            self.copy_async_commit_group()
            self.copy_async_wait_group(self.num_stages - 2)
            self.syncthreads()

        if self.channel_wise:
            if self.with_zeros:
                self.sub(self.regs_acc, self.cast(self.regs_zeros, dtype=self.accumulate_dtype), out=self.regs_acc)
            self.mul(self.regs_acc, self.cast(self.regs_scales, dtype=self.accumulate_dtype), out=self.regs_acc)

        self.free_smem()
        self.write_back()

    def generate_program(self, params: List[Var]) -> VirtualMachineProgram:
        if self.config.m_size is not None:
            self.a_ptr, self.b_ptr, self.c_ptr, self.scales_ptr, self.zeros_ptr = params
            self.m_size = self.config.m_size
        else:
            self.a_ptr, self.b_ptr, self.c_ptr, self.scales_ptr, self.zeros_ptr, self.m_size = params
        with self.program("matmul_mma", num_warps=self.num_warps, params=params):
            self.issue_weight_transform()
            self.pipelined_body()

        prog = self.finish()
        prog.annotations["schedule"] = str(self)
        prog.annotations["schedule_dict"] = self.schedule_dict()
        return prog


def matmul_mma_kernel_base(
    a_ptr,
    b_ptr,
    c_ptr,
    scale_ptr,
    zero_ptr,
    m_size,
    n_size,
    k_size,
    group_size,
    a_dtype,
    b_dtype,
    c_dtype,
    mma_operand_dtype,
    mma_accumulate_dtype,
):
    config = Config(
        m_size=m_size if isinstance(m_size, int) else None,
        n_size=n_size,
        k_size=k_size,
        group_size=group_size,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        mma_operand_dtype=mma_operand_dtype,
        mma_accumulate_dtype=mma_accumulate_dtype,
    )
    schedules = MatmulSchedule.generate_schedules(get_current_space(), config)
    params = [a_ptr, b_ptr, c_ptr, scale_ptr, zero_ptr]
    if isinstance(m_size, Var):
        params.append(m_size)
    programs = [schedule.generate_program(params) for schedule in schedules]
    return programs


@vm_jit(
    # debug_use_schedules=[85],
    # debug_dump_ir=True
    # debug_print_vm_inst_output=True
)
def matmul_mma_kernel_static(
    a_ptr: ~void,
    b_ptr: ~void,
    c_ptr: ~void,
    scale_ptr: ~void,
    zero_ptr: ~void,
    m_size: constant[int],
    n_size: constant[int],
    k_size: constant[int],
    group_size: constant[int],
    a_dtype: constant[DataType],
    b_dtype: constant[DataType],
    c_dtype: constant[DataType],
    mma_operand_dtype: constant[DataType],
    mma_accumulate_dtype: constant[DataType],
):
    return matmul_mma_kernel_base(
        a_ptr=a_ptr,
        b_ptr=b_ptr,
        c_ptr=c_ptr,
        scale_ptr=scale_ptr,
        zero_ptr=zero_ptr,
        m_size=m_size,
        n_size=n_size,
        k_size=k_size,
        group_size=group_size,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        mma_operand_dtype=mma_operand_dtype,
        mma_accumulate_dtype=mma_accumulate_dtype,
    )


@vm_jit()
def matmul_mma_kernel_dynamic(
    a_ptr: ~void,
    b_ptr: ~void,
    c_ptr: ~void,
    scale_ptr: ~void,
    zero_ptr: ~void,
    m_size: int,
    n_size: constant[int],
    k_size: constant[int],
    group_size: constant[int],
    a_dtype: constant[DataType],
    b_dtype: constant[DataType],
    c_dtype: constant[DataType],
    mma_operand_dtype: constant[DataType],
    mma_accumulate_dtype: constant[DataType],
):
    return matmul_mma_kernel_base(
        a_ptr=a_ptr,
        b_ptr=b_ptr,
        c_ptr=c_ptr,
        scale_ptr=scale_ptr,
        zero_ptr=zero_ptr,
        m_size=m_size,
        n_size=n_size,
        k_size=k_size,
        group_size=group_size,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        mma_operand_dtype=mma_operand_dtype,
        mma_accumulate_dtype=mma_accumulate_dtype,
    )


def reference_matmul_mma(
    m: int,
    n: int,
    k: int,
    group_size: int,
    a: torch.Tensor,
    b: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    a_dtype: DataType,
    b_dtype: DataType,
    c_dtype: DataType,
):
    """
    Reference implementation of matrix multiplication using MMA (Matrix Multiply-Accumulate).

    Parameters
    ----------
    m : int
        The number of rows in matrix A and the resulting matrix C.
    n : int
        The number of columns in matrix B and the resulting matrix C.
    k : int
        The number of columns in matrix A and the number of rows in matrix B.
    group_size : int
        The size of sub-channel-wise quantization. If group_size is -1, the quantization is channel-wise.
    a : torch.Tensor
        The first input matrix A with shape (m, k).
    b : torch.Tensor
        The second input matrix B with shape (k, n).
    scales : torch.Tensor
        The scaling factors for quantization.
    zeros : torch.Tensor
        The zero points for quantization.

    Returns
    -------
    torch.Tensor
        The resulting matrix C with shape (m, n) after performing the matrix multiplication.
    """
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    if group_size == -1:
        group_size = k
    a = a.to(dtype=torch.float32).reshape(m, k)
    b = b.to(dtype=torch.float32).reshape(k, n)
    scales = scales.to(dtype=torch.float32).reshape(k // group_size, n)
    zeros = zeros.to(dtype=torch.float32).reshape(k // group_size, n)
    b = b.reshape(k // group_size, group_size, n)
    scales = scales.reshape(k // group_size, 1, n)
    zeros = zeros.reshape(k // group_size, 1, n)
    if b_dtype.is_unsigned_integer():
        b = b - zeros
    b = b * scales
    b = b.reshape(k, n)
    c = torch.matmul(a, b)
    return c.half()


def matmul_mma(
    m: int,
    n: int,
    k: int,
    group_size: int,
    a: torch.Tensor,
    b: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    a_dtype: DataType,
    b_dtype: DataType,
    c_dtype: DataType,
    use_dynamic_m: bool,
):
    import mutis

    assert scales.dtype == zeros.dtype == a.dtype == mutis.dtype_to_torch(a_dtype), (scales.dtype, zeros.dtype, a.dtype)

    c = torch.empty((m, n), dtype=mutis.dtype_to_torch(c_dtype), device="cuda")
    if use_dynamic_m:
        matmul_mma_kernel_dynamic(
            a, b, c, scales, zeros, m, n, k, group_size, a_dtype, b_dtype, c_dtype, a_dtype, mutis.float32
        )
    else:
        matmul_mma_kernel_static(
            a, b, c, scales, zeros, m, n, k, group_size, a_dtype, b_dtype, c_dtype, a_dtype, mutis.float32
        )
    return c
