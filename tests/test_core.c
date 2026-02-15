/**
 * @file test_core.c
 * @brief Unit tests for Phase 1 core modules
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn_test.h"
#include "edgenn/edgenn.h"
#include <stdlib.h>

/* ============================================================================
 * Tensor Tests
 * ========================================================================= */

static void test_tensor_init_2d(void)
{
    TEST_CASE("tensor_init 2D [3×4]");
    edgenn_tensor_t t;
    int32_t shape[] = {3, 4};
    ASSERT_OK(edgenn_tensor_init(&t, shape, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC));
    ASSERT_EQ(t.ndim, 2);
    ASSERT_EQ(t.shape[0], 3);
    ASSERT_EQ(t.shape[1], 4);
    ASSERT_EQ(t.strides[0], 4);
    ASSERT_EQ(t.strides[1], 1);
    ASSERT_EQ(edgenn_tensor_numel(&t), 12);
    ASSERT_EQ(edgenn_tensor_byte_size(&t), 48u);
    TEST_PASS();
}

static void test_tensor_init_4d(void)
{
    TEST_CASE("tensor_init 4D [1×28×28×3] NHWC");
    edgenn_tensor_t t;
    int32_t shape[] = {1, 28, 28, 3};
    ASSERT_OK(edgenn_tensor_init(&t, shape, 4, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NHWC));
    ASSERT_EQ(t.ndim, 4);
    ASSERT_EQ(edgenn_tensor_numel(&t), 2352);
    ASSERT_EQ(edgenn_tensor_byte_size(&t), 2352u);
    ASSERT_EQ(t.strides[0], 2352);
    ASSERT_EQ(t.strides[1], 84);
    ASSERT_EQ(t.strides[2], 3);
    ASSERT_EQ(t.strides[3], 1);
    TEST_PASS();
}

static void test_tensor_null_checks(void)
{
    TEST_CASE("tensor_init NULL checks");
    int32_t shape[] = {2, 3};
    ASSERT_ERR(edgenn_tensor_init(NULL, shape, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC),
               EDGENN_ERR_NULL_PTR);
    edgenn_tensor_t t;
    ASSERT_ERR(edgenn_tensor_init(&t, NULL, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_tensor_init(&t, shape, 0, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC),
               EDGENN_ERR_INVALID_ARG);
    ASSERT_ERR(edgenn_tensor_init(&t, shape, 5, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC),
               EDGENN_ERR_INVALID_ARG);
    TEST_PASS();
}

static void test_tensor_shapes_equal(void)
{
    TEST_CASE("tensor_shapes_equal");
    edgenn_tensor_t a, b;
    int32_t shape[] = {2, 3, 4};
    edgenn_tensor_init(&a, shape, 3, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NTC);
    edgenn_tensor_init(&b, shape, 3, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NTC);
    ASSERT_TRUE(edgenn_tensor_shapes_equal(&a, &b));

    int32_t shape2[] = {2, 4, 4};
    edgenn_tensor_init(&b, shape2, 3, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NTC);
    ASSERT_TRUE(!edgenn_tensor_shapes_equal(&a, &b));
    TEST_PASS();
}

/* ============================================================================
 * Arena Tests
 * ========================================================================= */

static void test_arena_basic(void)
{
    TEST_CASE("arena basic alloc/reset");
    uint8_t buffer[1024];
    edgenn_arena_t arena;
    ASSERT_OK(edgenn_arena_init(&arena, buffer, sizeof(buffer)));
    ASSERT_EQ(edgenn_arena_remaining(&arena), 1024u);

    void *p1 = NULL;
    ASSERT_OK(edgenn_arena_alloc(&arena, 100, 4, &p1));
    ASSERT_TRUE(p1 != NULL);
    ASSERT_TRUE(edgenn_arena_remaining(&arena) <= 924u);

    void *p2 = NULL;
    ASSERT_OK(edgenn_arena_alloc(&arena, 200, 16, &p2));
    ASSERT_TRUE(p2 != NULL);
    ASSERT_TRUE(((uintptr_t)p2 % 16) == 0);

    edgenn_arena_reset(&arena);
    ASSERT_EQ(edgenn_arena_remaining(&arena), 1024u);
    ASSERT_TRUE(edgenn_arena_peak(&arena) > 0);
    TEST_PASS();
}

static void test_arena_oom(void)
{
    TEST_CASE("arena OOM detection");
    uint8_t buffer[64];
    edgenn_arena_t arena;
    edgenn_arena_init(&arena, buffer, sizeof(buffer));

    void *p = NULL;
    ASSERT_ERR(edgenn_arena_alloc(&arena, 128, 4, &p),
               EDGENN_ERR_OUT_OF_MEMORY);
    ASSERT_TRUE(p == NULL);
    TEST_PASS();
}

static void test_arena_save_restore(void)
{
    TEST_CASE("arena save/restore");
    uint8_t buffer[1024];
    edgenn_arena_t arena;
    edgenn_arena_init(&arena, buffer, sizeof(buffer));

    void *p1 = NULL;
    edgenn_arena_alloc(&arena, 100, 4, &p1);
    size_t saved = edgenn_arena_save(&arena);

    void *p2 = NULL;
    edgenn_arena_alloc(&arena, 200, 4, &p2);
    ASSERT_TRUE(edgenn_arena_remaining(&arena) <= 724u);

    edgenn_arena_restore(&arena, saved);
    ASSERT_TRUE(edgenn_arena_remaining(&arena) >= 920u);
    TEST_PASS();
}

static void test_arena_tensor_alloc(void)
{
    TEST_CASE("arena tensor allocation");
    uint8_t buffer[4096];
    edgenn_arena_t arena;
    edgenn_arena_init(&arena, buffer, sizeof(buffer));

    edgenn_tensor_t t;
    int32_t shape[] = {1, 32};
    edgenn_tensor_init(&t, shape, 2, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NC);
    ASSERT_TRUE(t.data == NULL);

    ASSERT_OK(edgenn_arena_alloc_tensor(&arena, &t));
    ASSERT_TRUE(t.data != NULL);
    ASSERT_TRUE(((uintptr_t)t.data % EDGENN_TENSOR_ALIGN) == 0);
    TEST_PASS();
}

static void test_pingpong(void)
{
    TEST_CASE("ping-pong buffer swap");
    uint8_t buf_a[512], buf_b[512];
    edgenn_pingpong_t pp;
    ASSERT_OK(edgenn_pingpong_init(&pp, buf_a, 512, buf_b, 512));

    edgenn_arena_t *in  = edgenn_pingpong_input(&pp);
    edgenn_arena_t *out = edgenn_pingpong_output(&pp);
    ASSERT_TRUE(in != out);
    ASSERT_EQ(in->base, buf_a);
    ASSERT_EQ(out->base, buf_b);

    edgenn_pingpong_swap(&pp);
    edgenn_arena_t *in2  = edgenn_pingpong_input(&pp);
    edgenn_arena_t *out2 = edgenn_pingpong_output(&pp);
    ASSERT_EQ(in2->base, buf_b);
    ASSERT_EQ(out2->base, buf_a);
    TEST_PASS();
}

/* ============================================================================
 * FP32 Math Tests
 * ========================================================================= */

static void test_fp32_matmul(void)
{
    TEST_CASE("fp32 matmul 2x3 × 3x2");
    float A[] = {1, 2, 3, 4, 5, 6};      /* 2×3 */
    float B[] = {7, 8, 9, 10, 11, 12};   /* 3×2 */
    float C[4];
    edgenn_fp32_matmul(A, B, NULL, C, 2, 2, 3);
    ASSERT_NEAR(C[0], 58.0f,  0.001f);   /* 1*7+2*9+3*11 = 58 */
    ASSERT_NEAR(C[1], 64.0f,  0.001f);   /* 1*8+2*10+3*12 = 64 */
    ASSERT_NEAR(C[2], 139.0f, 0.001f);   /* 4*7+5*9+6*11 = 139 */
    ASSERT_NEAR(C[3], 154.0f, 0.001f);   /* 4*8+5*10+6*12 = 154 */
    TEST_PASS();
}

static void test_fp32_matmul_bias(void)
{
    TEST_CASE("fp32 matmul with bias");
    float A[] = {1, 1};
    float B[] = {2, 3};
    float bias[] = {10, 20};
    float C[2];
    edgenn_fp32_matmul(A, B, bias, C, 1, 2, 1);
    ASSERT_NEAR(C[0], 12.0f, 0.001f);
    ASSERT_NEAR(C[1], 23.0f, 0.001f);
    TEST_PASS();
}

static void test_fp32_sigmoid(void)
{
    TEST_CASE("fp32 sigmoid");
    float x[] = {0.0f, 2.0f, -2.0f, 10.0f, -10.0f};
    float out[5];
    edgenn_fp32_sigmoid(x, out, 5);
    ASSERT_NEAR(out[0], 0.5f,    0.001f);
    ASSERT_NEAR(out[1], 0.8808f, 0.001f);
    ASSERT_NEAR(out[2], 0.1192f, 0.001f);
    ASSERT_TRUE(out[3] > 0.999f);
    ASSERT_TRUE(out[4] < 0.001f);
    TEST_PASS();
}

static void test_fp32_softmax(void)
{
    TEST_CASE("fp32 softmax sums to 1.0");
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4];
    edgenn_fp32_softmax(x, out, 4);
    float sum = out[0] + out[1] + out[2] + out[3];
    ASSERT_NEAR(sum, 1.0f, 0.001f);
    ASSERT_TRUE(out[3] > out[2]);
    ASSERT_TRUE(out[2] > out[1]);
    ASSERT_TRUE(out[1] > out[0]);
    TEST_PASS();
}

static void test_fp32_relu(void)
{
    TEST_CASE("fp32 relu");
    float x[] = {-3.0f, -1.0f, 0.0f, 1.0f, 5.0f};
    float out[5];
    edgenn_fp32_relu(x, out, 5);
    ASSERT_NEAR(out[0], 0.0f, 0.001f);
    ASSERT_NEAR(out[1], 0.0f, 0.001f);
    ASSERT_NEAR(out[2], 0.0f, 0.001f);
    ASSERT_NEAR(out[3], 1.0f, 0.001f);
    ASSERT_NEAR(out[4], 5.0f, 0.001f);
    TEST_PASS();
}

static void test_fp32_gelu(void)
{
    TEST_CASE("fp32 gelu approximation");
    float x[] = {0.0f, 1.0f, -1.0f};
    float out[3];
    edgenn_fp32_gelu(x, out, 3);
    ASSERT_NEAR(out[0], 0.0f,    0.01f);
    ASSERT_NEAR(out[1], 0.8412f, 0.01f);   /* GELU(1) ≈ 0.8412 */
    ASSERT_NEAR(out[2], -0.1588f, 0.01f);  /* GELU(-1) ≈ -0.1588 */
    TEST_PASS();
}

static void test_fp32_layernorm(void)
{
    TEST_CASE("fp32 layernorm");
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float gamma[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float beta[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float out[4];
    edgenn_fp32_layernorm(x, gamma, beta, out, 4, 1e-5f);
    /* Mean = 2.5, std ≈ 1.118 → normalized: [-1.342, -0.447, 0.447, 1.342] */
    ASSERT_NEAR(out[0], -1.3416f, 0.01f);
    ASSERT_NEAR(out[3],  1.3416f, 0.01f);
    /* Sum should be ~0 */
    float sum = out[0] + out[1] + out[2] + out[3];
    ASSERT_NEAR(sum, 0.0f, 0.01f);
    TEST_PASS();
}

/* ============================================================================
 * Quantization Tests
 * ========================================================================= */

static void test_quant_fp32_to_int8_roundtrip(void)
{
    TEST_CASE("quant fp32→int8→fp32 roundtrip");
    float orig[] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    int8_t quantized[5];
    float dequantized[5];

    float scale = 2.0f / 255.0f; /* range [-1, 1] symmetric */
    int32_t zp = 0;

    ASSERT_OK(edgenn_quant_fp32_to_int8(orig, quantized, 5, scale, zp));
    ASSERT_OK(edgenn_quant_int8_to_fp32(quantized, dequantized, 5, scale, zp));

    for (int i = 0; i < 5; i++) {
        ASSERT_NEAR(dequantized[i], orig[i], 0.02f);
    }
    TEST_PASS();
}

static void test_quant_multiplier(void)
{
    TEST_CASE("quant compute_multiplier");
    int32_t mult;
    int8_t shift;
    ASSERT_OK(edgenn_quant_compute_multiplier(0.5f, &mult, &shift));
    ASSERT_TRUE(mult > 0);
    /* 0.5 should decompose to significand=1.0 * 2^-1, but frexp gives 0.5*2^0 */
    TEST_PASS();
}

static void test_quant_symmetric_params(void)
{
    TEST_CASE("quant symmetric params");
    edgenn_qparams_t qp;
    ASSERT_OK(edgenn_quant_params_symmetric(-3.0f, 3.0f, &qp));
    ASSERT_EQ(qp.scheme, EDGENN_QSCHEME_SYMMETRIC);
    ASSERT_EQ(qp.zero_point, 0);
    ASSERT_NEAR(qp.scale, 3.0f / 127.0f, 0.001f);
    TEST_PASS();
}

/* ============================================================================
 * INT8 Math Tests
 * ========================================================================= */

static void test_q8_matmul(void)
{
    TEST_CASE("q8 matmul simple");
    /* Simple 1×2 × 2×1 multiply */
    int8_t A[] = {10, 20};
    int8_t B[] = {3, 4};
    int32_t bias[] = {0};
    int8_t C[1];

    /* Symmetric quantization: zp=0 */
    /* Expected: 10*3 + 20*4 = 110 */
    /* With trivial requant (mult≈1, shift=0): output should be clipped to 110→127 range */
    edgenn_q8_matmul(A, B, bias, C, 1, 1, 2, 0, 0,
                     1073741824, /* ~0.5 in fixed point */ 0, 0);
    /* 110 * 0.5 = 55 */
    ASSERT_EQ(C[0], 55);
    TEST_PASS();
}

static void test_q8_relu(void)
{
    TEST_CASE("q8 relu (zp=0)");
    int8_t x[] = {-10, -1, 0, 1, 50};
    int8_t out[5];
    edgenn_q8_relu(x, out, 5, 0);
    ASSERT_EQ(out[0], 0);
    ASSERT_EQ(out[1], 0);
    ASSERT_EQ(out[2], 0);
    ASSERT_EQ(out[3], 1);
    ASSERT_EQ(out[4], 50);
    TEST_PASS();
}

static void test_requantize(void)
{
    TEST_CASE("requantize basic");
    /* multiplier = 0.5 in Q31 = 1073741824 */
    int32_t result = edgenn_q_requantize(100, 1073741824, 0);
    ASSERT_EQ(result, 50);

    result = edgenn_q_requantize(200, 1073741824, 1);
    ASSERT_EQ(result, 50);
    TEST_PASS();
}

/* ============================================================================
 * LUT Tests
 * ========================================================================= */

static void test_lut_sigmoid(void)
{
    TEST_CASE("LUT sigmoid generation");
    int8_t lut[256];
    ASSERT_OK(edgenn_lut_sigmoid_q8(lut, 0.05f, 0, 1.0f/256.0f, -128));
    /* At input q=0 → real=0.0 → sigmoid=0.5 → q_out = 0.5/scale + zp */
    /* lut[128] corresponds to q_in=0 */
    /* sigmoid(0) = 0.5, so q_out = round(0.5 * 256) - 128 = 0 */
    ASSERT_TRUE(lut[128] >= -5 && lut[128] <= 5); /* roughly 0 */
    /* At extreme negative: lut[0] should be near -128 */
    ASSERT_TRUE(lut[0] < -100);
    /* At extreme positive: lut[255] should be near 127 */
    ASSERT_TRUE(lut[255] > 100);
    TEST_PASS();
}

/* ============================================================================
 * HAL Tests
 * ========================================================================= */

static void test_hal_basics(void)
{
    TEST_CASE("HAL platform name and cycle counter");
    const char *name = edgenn_hal_platform_name();
    ASSERT_TRUE(name != NULL);
    ASSERT_TRUE(strlen(name) > 0);

    edgenn_hal_cycle_init();
    uint32_t c1 = edgenn_hal_cycle_count();
    /* Do some work */
    volatile int x = 0;
    for (int i = 0; i < 10000; i++) x += i;
    (void)x;
    uint32_t c2 = edgenn_hal_cycle_count();
    ASSERT_TRUE(c2 != c1); /* Cycles should have advanced */
    TEST_PASS();
}

/* ============================================================================
 * Main
 * ========================================================================= */

int main(void)
{
    TEST_SUITE_BEGIN("EdgeNN Phase 1 — Core Foundation");

    /* Tensor tests */
    test_tensor_init_2d();
    test_tensor_init_4d();
    test_tensor_null_checks();
    test_tensor_shapes_equal();

    /* Arena tests */
    test_arena_basic();
    test_arena_oom();
    test_arena_save_restore();
    test_arena_tensor_alloc();
    test_pingpong();

    /* FP32 math tests */
    test_fp32_matmul();
    test_fp32_matmul_bias();
    test_fp32_sigmoid();
    test_fp32_softmax();
    test_fp32_relu();
    test_fp32_gelu();
    test_fp32_layernorm();

    /* Quantization tests */
    test_quant_fp32_to_int8_roundtrip();
    test_quant_multiplier();
    test_quant_symmetric_params();

    /* INT8 math tests */
    test_q8_matmul();
    test_q8_relu();
    test_requantize();

    /* LUT tests */
    test_lut_sigmoid();

    /* HAL tests */
    test_hal_basics();

    TEST_SUITE_END();
}
