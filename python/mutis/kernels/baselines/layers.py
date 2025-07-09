from typing import Optional, Union
import torch
from torch import nn

import mutis
from hidet import bfloat16
from hidet.graph.frontend.torch.utils import dtype_to_torch
from hidet.ir.type import data_type
from hidet.ir.dtypes import float16, uint8, uint7b, uint6b, uint5b, uint4b, uint3b, uint2b, uint1b, float32
from hidet.ir.dtypes import int8, int7b, int6b, int5b, int4b, int3b, int2b, int1b, int32
from hidet.ir.dtypes import float8_e4m3, float6_e3m2, float6_e2m3, float5_e3m1, float5_e2m2, float4_e2m1, float3_e1m1
from hidet.ir.dtypes import float7_e3m3, float7_e4m2, float7_e2m4, float8_e5m2, float7_e5m1, float6_e4m1
from mutis.kernels.vm.matmul_mma import matmul_mma
from mutis.kernels.vm.matmul_mma_decode import matmul_mma_decode
from mutis.types import DataType
from mutis.utils import benchmark_func


class NotSupportedError(Exception):
    pass


class MatmulLayer(nn.Module):
    matmul_id = 0

    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__()
        self.a_dtype: DataType = a_dtype
        self.b_dtype: DataType = b_dtype
        self.group_size: int = group_size
        self.m: int = m
        self.k: int = k
        self.n: int = n

        if (a_dtype, b_dtype) not in self.supported_pairs():
            raise NotSupportedError(a_dtype, b_dtype)

        self._a: Optional[torch.Tensor] = None

    @staticmethod
    def get_cls(runner_name):
        return {
            'torch-f16': TorchF16Layer,
            'triton': TritonLayer,
            'bitblas': BitblasLayerV1,
            'quant-llm': QuantLLMLayer,
            'marlin': MarlinLayer,
            'mutis': MutisLayer,
        }[runner_name]

    @staticmethod
    def create(runner_name: str, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        show_memory = True and False
        prev = torch.cuda.memory_allocated()
        layer = MatmulLayer.get_cls(runner_name)(a_dtype, b_dtype, group_size, m, k, n)
        after = torch.cuda.memory_allocated()
        if show_memory:
            print(
                f"[{MatmulLayer.matmul_id // 4}][{MatmulLayer.matmul_id % 4}] Allocated {(after - prev) / 1024 / 1024 / 1024:.2f} GiB layer {k}x{n} with g{group_size}"
            )
            print(f'    current used: {after / 1024 / 1024 / 1024:.2f} GiB')
        MatmulLayer.matmul_id += 1
        return layer

    @staticmethod
    def supports(runner_name, a_dtype: Union[DataType, str], b_dtype: Union[DataType, str]) -> bool:
        a_dtype = data_type(a_dtype)
        b_dtype = data_type(b_dtype)
        return (a_dtype, b_dtype) in MatmulLayer.get_cls(runner_name).supported_pairs()

    @property
    def a(self):
        if self._a is None:
            self._a = torch.empty(self.m, self.k, dtype=mutis.dtype_to_torch(self.a_dtype), device='cuda')
        return self._a

    @staticmethod
    def supported_pairs():
        raise NotImplementedError()

    def run(self, a: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError()

    def bench(self, warmup=None, repeat=None) -> float:
        return benchmark_func(
            run_func=lambda: self.run(),
            warmup=10 if warmup is None else warmup,
            repeat=50 if repeat is None else repeat,
            maximum_repeat_time=None,
            clear_l2_cache=True,
        )


class TorchF16Layer(MatmulLayer):
    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__(a_dtype, b_dtype, group_size, m, k, n)
        self.b = torch.randn(k, n, dtype=torch.float16, device='cuda')

    @staticmethod
    def supported_pairs():
        return [(float16, float16)]

    def run(self, a: Optional[torch.Tensor] = None):
        return torch.matmul(self.a if a is None else a, self.b)


class TritonLayer(MatmulLayer):
    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__(a_dtype, b_dtype, group_size, m, k, n)
        if b_dtype.nbits == 4:
            self.b = torch.randint(0, 1, size=[k // 8, n], dtype=torch.int32, device='cuda')
        else:
            self.b = torch.randint(0, max(int(b_dtype.max_value) // 2, 1), size=[k, n], dtype=torch.int8, device='cuda')
        self.zeros = torch.randint(0, 1, [k // group_size, n // 8], dtype=torch.int32, device='cuda')
        self.scales = torch.rand([k // group_size, n], dtype=mutis.dtype_to_torch(a_dtype), device='cuda')

    @staticmethod
    def supported_pairs():
        return [(float16, uint8), (float16, uint4b)]

    def triton_quantized_gemm(self, a, w, scales, zeros, group_size, b_dtype: DataType):
        from mutis.kernels.triton_kernels import triton_matmul_w8a16, triton_gemm_w4a16

        if b_dtype.nbits == 4:
            return triton_gemm_w4a16(groupsize=group_size, a=a, qweight=w, scales=scales, qzeros=zeros)
        elif b_dtype.nbits == 8:
            return triton_matmul_w8a16(a=a, b=w, scale=scales)
        else:
            raise NotImplementedError()

    def run(self, a: Optional[torch.Tensor] = None):
        a = self.a if a is None else a
        if self.m != a.size(0):
            return torch.empty(a.size(0), self.n, dtype=a.dtype, device=a.device)
        else:
            return self.triton_quantized_gemm(
                a, self.b, self.scales, self.zeros, group_size=self.group_size, b_dtype=self.b_dtype
            )


class BitblasBaseLayer(MatmulLayer):
    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__(a_dtype, b_dtype, group_size, m, k, n)
        import bitblas

        bitblas.set_log_level("ERROR")

    @staticmethod
    def supported_pairs():
        return [
            (mutis.float16, mutis.uint8),
            (mutis.float16, mutis.uint4b),
            (mutis.float16, mutis.uint2b),
            (mutis.float16, mutis.uint1b),
            (mutis.float16, mutis.int8),
            (mutis.float16, mutis.int4b),
            (mutis.float16, mutis.int2b),
            (mutis.float16, mutis.int1b),
            # (mutis.float16, mutis.float4_e2m1)
        ]

    def dtype_to_bitblas(self, dtype: DataType) -> str:
        if dtype.is_integer_subbyte():
            return dtype.name.removesuffix('b')
        if dtype == mutis.float4_e2m1:
            return 'fp4_e2m1'
        else:
            return dtype.name


class BitblasLayerV1(BitblasBaseLayer):
    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__(a_dtype, b_dtype, group_size, m, k, n)

        import bitblas.cache

        # bitblas.set_log_level("Debug")

        acc_dtype = 'float32'
        self.layer = bitblas.Linear(
            in_features=k,
            out_features=n,
            bias=False,
            A_dtype=self.dtype_to_bitblas(a_dtype),
            W_dtype=self.dtype_to_bitblas(b_dtype),
            accum_dtype=acc_dtype,
            out_dtype=self.dtype_to_bitblas(a_dtype),
            group_size=group_size,
            with_scaling=True,
            with_zeros=True if b_dtype.is_unsigned_integer() else False,
            zeros_mode='original',
            opt_M=[m],
        ).cuda()

    def run(self, a: Optional[torch.Tensor] = None):
        a = self.a if a is None else a
        if self.m != a.size(0):
            return torch.empty(a.size(0), self.n, dtype=a.dtype, device=a.device)
        else:
            return self.layer(a)


class BitblasLayerV2(BitblasBaseLayer):
    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__(a_dtype, b_dtype, group_size, m, k, n)
        from bitblas import MatmulConfig, Matmul

        config = MatmulConfig(
            M=self.m,
            N=self.n,
            K=self.k,
            A_dtype=self.dtype_to_bitblas(a_dtype),
            W_dtype=self.dtype_to_bitblas(b_dtype),
            out_dtype=self.dtype_to_bitblas(a_dtype),
            group_size=group_size,
            accum_dtype='float32',
            with_scaling=True,
            with_zeros=True,
            zeros_mode='original',
            storage_dtype="uint32" if b_dtype == float4_e2m1 else "int8",
        )
        self.op = Matmul(config)
        self.b = self.op.transform_weight(torch.randint(-2, 2, [n, k], dtype=torch.int8, device='cuda'))
        self.scales = torch.rand([k // group_size, n], dtype=dtype_to_torch(a_dtype), device='cuda')
        self.zeros = torch.rand([k // group_size, n], dtype=dtype_to_torch(a_dtype), device='cuda')

    def run(self, a: Optional[torch.Tensor] = None):
        a = self.a if a is None else a
        if self.m != a.size(0):
            return torch.empty(a.size(0), self.n, dtype=a.dtype, device=a.device)
        else:
            return self.op(a, self.b, self.scales, self.zeros)


class QuantLLMLayer(MatmulLayer):
    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__(a_dtype, b_dtype, group_size, m, k, n)
        import fp6_llm

        self.fp6_packed_weight = torch.empty(n, k // 16 * 3, dtype=torch.int32, device='cuda')
        self.fp16_scale = torch.randn(n, dtype=torch.float16, device='cuda')
        Number_GPU_SMs = torch.cuda.get_device_properties(0).multi_processor_count
        self.splitK = fp6_llm.HeuristicFuntion_SplitK(m, n, Number_GPU_SMs)

        assert a_dtype.is_float() and a_dtype.nbits == 16 and b_dtype.is_float() and b_dtype.nbits == 6

    @staticmethod
    def supported_pairs():
        return [(float16, float6_e3m2)]

    def run(self, a: Optional[torch.Tensor] = None):
        import fp6_llm

        a = self.a if a is None else a
        return fp6_llm.linear_forward_cuda(a, self.fp6_packed_weight, self.fp16_scale, self.splitK)


class MutisLayer(MatmulLayer):
    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__(a_dtype, b_dtype, group_size, m, k, n)
        mutis.set_benchmark_mode(True)
        group_size = k if group_size == -1 else group_size
        self.b = mutis.randn([k, n], dtype=b_dtype).storage
        self.scales = mutis.randn([k // group_size, n], dtype=self.get_scale_dtype()).torch()
        self.zeros = mutis.randn([k // group_size, n], dtype=self.get_scale_dtype()).torch()

    def get_mma_operand_dtype(self):
        if self.a_dtype in [float16, bfloat16]:
            return self.a_dtype
        elif self.a_dtype == int8 and self.b_dtype.is_signed_integer():
            return int8
        else:
            return float16

    def get_accumulate_dtype(self):
        return float32
        # if self.a_dtype in [float16, bfloat16]:
        #     return self.a_dtype
        # elif self.a_dtype == int8 and self.b_dtype.is_signed_integer():
        #     return float32
        # else:
        #     return float16

    def get_scale_dtype(self):
        if self.a_dtype in [float16, bfloat16]:
            return self.a_dtype
        else:
            return float16

    def get_zeros_dtype(self):
        if self.a_dtype in [float16, bfloat16]:
            return self.a_dtype
        else:
            return float16

    def get_c_dtype(self):
        if self.a_dtype in [float16, bfloat16]:
            return self.a_dtype
        else:
            return float16

    @staticmethod
    def supported_pairs():
        pairs = []
        for a_dtype in [float16, bfloat16, int8, uint8]:
            for b_dtype in [
                uint8,
                uint7b,
                uint6b,
                uint5b,
                uint4b,
                uint3b,
                uint2b,
                uint1b,
                int8,
                int7b,
                int6b,
                int5b,
                int4b,
                int3b,
                int2b,
                int1b,
                float8_e5m2,
                float8_e4m3,
                float7_e5m1,
                float7_e4m2,
                float7_e3m3,
                float7_e2m4,
                float6_e4m1,
                float6_e3m2,
                float6_e2m3,
                float5_e3m1,
                float5_e2m2,
                float4_e2m1,
                float3_e1m1,
            ]:
                pairs.append((a_dtype, b_dtype))
        return pairs

    def run(self, a: Optional[torch.Tensor] = None):
        a = self.a if a is None else a
        if a.size(0) != self.m:
            # skip other batch sizes, which is used to calculate the memory consumption by vllm
            return torch.empty(a.size(0), self.n, dtype=a.dtype, device=a.device)
        else:
            if self.m > 256:
                b = matmul_mma_decode(k=self.k, n=self.n, dtype=self.b_dtype, output_dtype=self.a_dtype, x=self.b)
                return torch.matmul(a, b)
            else:
                return matmul_mma(
                    m=self.m,
                    n=self.n,
                    k=self.k,
                    group_size=self.group_size,
                    a=a,
                    b=self.b,
                    scales=self.scales,
                    zeros=self.zeros,
                    a_dtype=self.a_dtype,
                    b_dtype=self.b_dtype,
                    c_dtype=self.a_dtype,
                    use_dynamic_m=False,
                )


class MarlinLayer(MatmulLayer):
    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__(a_dtype, b_dtype, group_size, m, k, n)
        import marlin

        self.linear = nn.Linear(in_features=k, out_features=n, bias=False).cuda().half()
        self.marlin_layer = marlin.Layer(infeatures=k, outfeatures=n, groupsize=group_size).cuda()

        assert a_dtype == mutis.float16
        assert b_dtype in [mutis.int4b]

    @staticmethod
    def supported_pairs():
        return [(float16, int4b)]

    def run(self, a: Optional[torch.Tensor] = None):
        return self.marlin_layer(self.a if a is None else a)
