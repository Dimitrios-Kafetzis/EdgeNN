/**
 * @file edgenn_math_fp.h
 * @brief Floating-point math operations (reference / Cortex-A path)
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#ifndef EDGENN_MATH_FP_H
#define EDGENN_MATH_FP_H

#include "edgenn_types.h"
#include "edgenn_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief FP32 matrix multiply: C[M×N] = A[M×K] × B[K×N] + bias[N]
 *
 * @param a      Input matrix A (row-major)
 * @param b      Weight matrix B (row-major)
 * @param bias   Bias vector (length N), may be NULL
 * @param c      Output matrix C (row-major)
 * @param M      Number of rows in A / C
 * @param N      Number of columns in B / C
 * @param K      Inner dimension
 */
void edgenn_fp32_matmul(
    const float *a,
    const float *b,
    const float *bias,
    float       *c,
    int32_t      M,
    int32_t      N,
    int32_t      K
);

/**
 * @brief FP32 element-wise vector addition: c[i] = a[i] + b[i]
 */
void edgenn_fp32_vec_add(const float *a, const float *b, float *c, int32_t n);

/**
 * @brief FP32 element-wise vector multiplication: c[i] = a[i] * b[i]
 */
void edgenn_fp32_vec_mul(const float *a, const float *b, float *c, int32_t n);

/**
 * @brief FP32 dot product: sum(a[i] * b[i])
 */
float edgenn_fp32_dot(const float *a, const float *b, int32_t n);

/**
 * @brief FP32 sigmoid: out[i] = 1 / (1 + exp(-x[i]))
 */
void edgenn_fp32_sigmoid(const float *x, float *out, int32_t n);

/**
 * @brief FP32 tanh: out[i] = tanh(x[i])
 */
void edgenn_fp32_tanh(const float *x, float *out, int32_t n);

/**
 * @brief FP32 ReLU: out[i] = max(0, x[i])
 */
void edgenn_fp32_relu(const float *x, float *out, int32_t n);

/**
 * @brief FP32 ReLU6: out[i] = min(6, max(0, x[i]))
 */
void edgenn_fp32_relu6(const float *x, float *out, int32_t n);

/**
 * @brief FP32 GELU approximation: out[i] ≈ x * 0.5 * (1 + tanh(...))
 */
void edgenn_fp32_gelu(const float *x, float *out, int32_t n);

/**
 * @brief FP32 softmax over a vector of length n
 *
 * Numerically stable: subtracts max before exp.
 *
 * @param x    Input vector
 * @param out  Output probability vector (sums to 1.0)
 * @param n    Vector length
 */
void edgenn_fp32_softmax(const float *x, float *out, int32_t n);

/**
 * @brief FP32 layer norm: out = (x - mean) / sqrt(var + eps) * gamma + beta
 *
 * @param x      Input vector
 * @param gamma  Scale parameter
 * @param beta   Offset parameter
 * @param out    Output vector
 * @param n      Vector length
 * @param eps    Small constant for numerical stability
 */
void edgenn_fp32_layernorm(
    const float *x,
    const float *gamma,
    const float *beta,
    float       *out,
    int32_t      n,
    float        eps
);

#ifdef __cplusplus
}
#endif

#endif /* EDGENN_MATH_FP_H */
