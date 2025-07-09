from hidet.ir.type import DataType, data_type
from hidet.ir.primitives.cuda.mma import MmaConfig
from hidet.ir.tile.layout import TileLayout, repeat, spatial
from hidet.utils import same_list


class TileMmaConfig:
    def __init__(self, config: MmaConfig, a_layout, b_layout, c_layout):
        self.config: MmaConfig = config
        self.a_layout: TileLayout = a_layout
        self.b_layout: TileLayout = b_layout
        self.c_layout: TileLayout = c_layout
        self.in_dtype: DataType = data_type(self.config.input_dtype)
        self.out_dtype: DataType = data_type(self.config.output_dtype)
        self.m: int = self.config.m
        self.n: int = self.config.n
        self.k: int = self.config.k

        # check consistency
        assert same_list(self.a_layout.logical_shape(), [self.m, self.k])
        assert same_list(self.b_layout.logical_shape(), [self.k, self.n])
        assert same_list(self.c_layout.logical_shape(), [self.m, self.n])

    def __eq__(self, other):
        assert isinstance(other, TileMmaConfig)
        return (
            self.m == other.m
            and self.n == other.n
            and self.k == other.k
            and self.in_dtype == other.in_dtype
            and self.out_dtype == other.out_dtype
        )

    @staticmethod
    def all():
        return [
            TileMmaConfig.m16n8k8_f16_f16(),
            TileMmaConfig.m16n8k8_f16_f32(),
            TileMmaConfig.m16n8k16_f16_f16(),
            TileMmaConfig.m16n8k16_f16_f32(),
            TileMmaConfig.m16n8k8_bf16_f32(),
            TileMmaConfig.m16n8k16_bf16_f32(),
            TileMmaConfig.m16n8k4_tf32_f32(),
            TileMmaConfig.m16n8k8_tf32_f32(),
            TileMmaConfig.m8n8k16_i8_i32(),
            TileMmaConfig.m16n8k16_i8_i32(),
            TileMmaConfig.m16n8k32_i8_i32(),
        ]

    @staticmethod
    def m16n8k8_f16_f16():
        return TileMmaConfig(
            config=MmaConfig.m16n8k8_f16_f16(),
            a_layout=repeat(2, 1).spatial(8, 4).repeat(1, 2),
            b_layout=spatial(4, 8, ranks=[1, 0]).repeat(2, 1),
            c_layout=repeat(2, 1).spatial(8, 4).repeat(1, 2),
        )

    @staticmethod
    def m16n8k8_f16_f32():
        return TileMmaConfig(
            config=MmaConfig.m16n8k8_f16_f32(),
            a_layout=repeat(2, 1).spatial(8, 4).repeat(1, 2),
            b_layout=spatial(4, 8, ranks=[1, 0]).repeat(2, 1),
            c_layout=repeat(2, 1).spatial(8, 4).repeat(1, 2),
        )

    @staticmethod
    def m16n8k16_f16_f16():
        return TileMmaConfig(
            config=MmaConfig.m16n8k16_f16_f16(),
            a_layout=repeat(2, 2, ranks=[1, 0]).spatial(8, 4).repeat(1, 2),
            b_layout=repeat(2, 1).spatial(4, 8, ranks=[1, 0]).repeat(2, 1),
            c_layout=repeat(2, 1).spatial(8, 4).repeat(1, 2),
        )

    @staticmethod
    def m16n8k16_f16_f32():
        return TileMmaConfig(
            config=MmaConfig.m16n8k16_f16_f32(),
            a_layout=repeat(2, 2, ranks=[1, 0]).spatial(8, 4).repeat(1, 2),
            b_layout=repeat(2, 1).spatial(4, 8, ranks=[1, 0]).repeat(2, 1),
            c_layout=repeat(2, 1).spatial(8, 4).repeat(1, 2),
        )

    @staticmethod
    def m16n8k8_bf16_f32():
        return TileMmaConfig(
            config=MmaConfig.m16n8k8_bf16_f32(),
            a_layout=repeat(2, 1).spatial(8, 4).repeat(1, 2),
            b_layout=spatial(4, 8, ranks=[1, 0]).repeat(2, 1),
            c_layout=repeat(2, 1).spatial(8, 4).repeat(1, 2),
        )

    @staticmethod
    def m16n8k16_bf16_f32():
        return TileMmaConfig(
            config=MmaConfig.m16n8k16_bf16_f32(),
            a_layout=repeat(2, 2, ranks=[1, 0]).spatial(8, 4).repeat(1, 2),
            b_layout=repeat(2, 1).spatial(4, 8, ranks=[1, 0]).repeat(2, 1),
            c_layout=repeat(2, 1).spatial(8, 4).repeat(1, 2),
        )

    @staticmethod
    def m16n8k4_tf32_f32():
        return TileMmaConfig(
            config=MmaConfig.m16n8k4_tf32_f32(),
            a_layout=repeat(2, 1).spatial(8, 4),
            b_layout=spatial(4, 8, ranks=[1, 0]),
            c_layout=repeat(2, 1).spatial(8, 4).repeat(1, 2),
        )

    @staticmethod
    def m16n8k8_tf32_f32():
        return TileMmaConfig(
            config=MmaConfig.m16n8k8_tf32_f32(),
            a_layout=repeat(2, 2, ranks=[1, 0]).spatial(8, 4),
            b_layout=repeat(2, 1).spatial(4, 8, ranks=[1, 0]),
            c_layout=repeat(2, 1).spatial(8, 4).repeat(1, 2),
        )

    @staticmethod
    def m8n8k16_i8_i32():
        return TileMmaConfig(
            config=MmaConfig.m8n8k16_i8_i32(),
            a_layout=spatial(8, 4).repeat(1, 4),
            b_layout=spatial(4, 8, ranks=[1, 0]).repeat(4, 1),
            c_layout=spatial(8, 4).repeat(1, 2),
        )

    @staticmethod
    def m16n8k16_i8_i32():
        return TileMmaConfig(
            config=MmaConfig.m16n8k16_i8_i32(),
            a_layout=repeat(2, 1).spatial(8, 4).repeat(1, 4),
            b_layout=spatial(4, 8, ranks=[1, 0]).repeat(4, 1),
            c_layout=repeat(2, 1).spatial(8, 4).repeat(1, 2),
        )

    @staticmethod
    def m16n8k32_i8_i32():
        return TileMmaConfig(
            config=MmaConfig.m16n8k32_i8_i32(),
            a_layout=repeat(2, 2).spatial(8, 4).repeat(1, 4),
            b_layout=repeat(2, 1).spatial(4, 8, ranks=[1, 0]).repeat(4, 1),
            c_layout=repeat(2, 1).spatial(8, 4).repeat(1, 2),
        )
