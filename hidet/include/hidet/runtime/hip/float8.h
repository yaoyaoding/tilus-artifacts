#pragma once
#include <cstdio>
#include <cstdint>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

namespace hidet {

typedef union interpret_half {
    uint16_t bits;
    __half value;
} interpret_half;

typedef union interpret_float {
    uint32_t bits;
    float value;
} interpret_float;

typedef union interpret_bfloat16 {
    uint16_t bits;
    __hip_bfloat16 value;
} interpret_bfloat16;

struct alignas(1) float8_e4m3 {
    uint8_t v;

    __host__ __device__
    float8_e4m3() {}

    // define the conversion between float and float8_e4m3
    __host__ __device__
    float8_e4m3(float f) {
        uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
        uint32_t sign = (bits >> 31) & 0x1;
        uint32_t exponent = (bits >> 23) & 0xFF;
        uint32_t mantissa = bits & 0x7FFFFF;

        // special cases
        if (exponent == 0xFF) {
            // NaN or Inf
            v = sign << 7 | 0x7F;
            return;
        }

        if (exponent == 0) {
            // zero or denormalized case
            v = sign << 7;
            return;
        }

        if (exponent > uint32_t(120 + 0x1F)) {
            // overflow to NaN
            v = sign << 7 | 0x7F;
            return;
        }
        if (exponent <= uint32_t(120)) {
            // underflow to subnormal or zero
            v = sign << 7 | ((mantissa | 0x800000) >> (uint32_t(120 + 21) - exponent));
            return;
        }

        mantissa = mantissa >> 20;
        v = sign << 7 | (exponent - uint32_t(120)) << 3 | mantissa;
    }
    __host__ __device__
    operator float() const {
        // 1 4 3
        // 1 8 23
        interpret_float t;

        if ((v & 0x7F) == 0x7F) {
            // NaN
            t.bits = 0x7F888888;
        } else {
            t.bits = (v & 0x80) << 24 | ((v & 0x7F) << 20);
            // for accurate result when t.value is subnormal, we should not turn on the `-ftz=true` option for nvcc.
            t.value = t.value * float(1329227995784915872903807060280344576.0);  // 2^120 where 120 = 2^7 - 2^3
        }
        return t.value;
    }

    __host__ __device__
    operator __half() const {
        // 1 4 3
        // 1 5 10
        interpret_half t;
        if ((v & 0x7F) == 0x7F) {
            // NaN
            t.bits = 0x7F88;
        } else {
            t.bits = (v & 0x80) << 8 | ((v & 0x7F) << 7);
            t.value = t.value * half(1 << 8);
        }
        return t.value;
    }

    __host__ __device__
    operator __hip_bfloat16() const {
        // 1 4 3
        // 1 8 7
        interpret_bfloat16 t;
        if ((v & 0x7F) == 0x7F) {
            // NaN
            t.bits = 0x7F88;
        } else {
            t.bits = (v & 0x80) << 8 | ((v & 0x7F) << 4);
            t.value = t.value * __float2bfloat16(1329227995784915872903807060280344576.0f); // 2^120 where 120 = 2^7 - 2^3
        }
        return t.value;
    }

};
}

typedef hidet::float8_e4m3 float8_e4m3;
