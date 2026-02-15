/**
 * @file test_dnn_ops.c
 * @brief Unit tests for Phase 2 DNN operators (starting with Dense)
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn_test.h"
#include "edgenn/edgenn.h"
#include <string.h>

/* ============================================================================
 * Helper: zero-initialize a dense params struct
 * ========================================================================= */

static void init_dense_params(edgenn_dense_params_t *p)
{
    memset(p, 0, sizeof(*p));
}

/* ============================================================================
 * Dense FP32 Tests
 * ========================================================================= */

static void test_dense_fp32_basic(void)
{
    TEST_CASE("dense FP32 basic 2×3 → 2×4");

    /* Input: [2 × 3] */
    float input_data[] = {1.0f, 2.0f, 3.0f,
                          4.0f, 5.0f, 6.0f};
    /* Weights: [4 × 3] (out_features × in_features) */
    float weight_data[] = {
        1.0f,  0.0f,  0.0f,   /* w[0]: picks input[0] */
        0.0f,  1.0f,  0.0f,   /* w[1]: picks input[1] */
        0.0f,  0.0f,  1.0f,   /* w[2]: picks input[2] */
        1.0f,  1.0f,  1.0f,   /* w[3]: sums all       */
    };
    float output_data[8];

    edgenn_tensor_t input, output;
    int32_t in_shape[] = {2, 3};
    int32_t out_shape[] = {2, 4};
    edgenn_tensor_init(&input,  in_shape,  2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, out_shape, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = input_data;
    output.data = output_data;

    edgenn_dense_params_t params;
    init_dense_params(&params);
    params.in_features  = 3;
    params.out_features = 4;
    params.fused_act    = EDGENN_ACT_NONE;
    edgenn_tensor_init(&params.weights, (int32_t[]){4, 3}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.weights.data = weight_data;

    ASSERT_OK(edgenn_dense_execute(&input, &output, &params, NULL));

    /* Row 0: [1,2,3] × identity-like weights */
    ASSERT_NEAR(output_data[0], 1.0f, 0.001f);  /* 1*1+2*0+3*0 */
    ASSERT_NEAR(output_data[1], 2.0f, 0.001f);  /* 1*0+2*1+3*0 */
    ASSERT_NEAR(output_data[2], 3.0f, 0.001f);  /* 1*0+2*0+3*1 */
    ASSERT_NEAR(output_data[3], 6.0f, 0.001f);  /* 1+2+3 */

    /* Row 1: [4,5,6] */
    ASSERT_NEAR(output_data[4], 4.0f,  0.001f);
    ASSERT_NEAR(output_data[5], 5.0f,  0.001f);
    ASSERT_NEAR(output_data[6], 6.0f,  0.001f);
    ASSERT_NEAR(output_data[7], 15.0f, 0.001f); /* 4+5+6 */

    TEST_PASS();
}

static void test_dense_fp32_with_bias(void)
{
    TEST_CASE("dense FP32 with bias");

    float input_data[]  = {1.0f, 2.0f};
    float weight_data[] = {1.0f, 0.0f,   /* w[0] = [1,0] */
                           0.0f, 1.0f};  /* w[1] = [0,1] */
    float bias_data[]   = {10.0f, 20.0f};
    float output_data[2];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 2}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 2}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = input_data;
    output.data = output_data;

    edgenn_dense_params_t params;
    init_dense_params(&params);
    params.in_features  = 2;
    params.out_features = 2;
    params.fused_act    = EDGENN_ACT_NONE;
    edgenn_tensor_init(&params.weights, (int32_t[]){2, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.weights.data = weight_data;
    edgenn_tensor_init(&params.bias, (int32_t[]){2}, 1,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.bias.data = bias_data;

    ASSERT_OK(edgenn_dense_execute(&input, &output, &params, NULL));

    ASSERT_NEAR(output_data[0], 11.0f, 0.001f);  /* 1*1+2*0+10 */
    ASSERT_NEAR(output_data[1], 22.0f, 0.001f);  /* 1*0+2*1+20 */

    TEST_PASS();
}

static void test_dense_fp32_relu(void)
{
    TEST_CASE("dense FP32 with fused ReLU");

    float input_data[]  = {-2.0f, 3.0f};
    float weight_data[] = {1.0f, 0.0f,   /* w[0] = [1,0] → picks -2 */
                           0.0f, 1.0f};  /* w[1] = [0,1] → picks  3 */
    float output_data[2];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 2}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 2}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = input_data;
    output.data = output_data;

    edgenn_dense_params_t params;
    init_dense_params(&params);
    params.in_features  = 2;
    params.out_features = 2;
    params.fused_act    = EDGENN_ACT_RELU;
    edgenn_tensor_init(&params.weights, (int32_t[]){2, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.weights.data = weight_data;

    ASSERT_OK(edgenn_dense_execute(&input, &output, &params, NULL));

    ASSERT_NEAR(output_data[0], 0.0f, 0.001f);  /* relu(-2) = 0 */
    ASSERT_NEAR(output_data[1], 3.0f, 0.001f);  /* relu(3)  = 3 */

    TEST_PASS();
}

static void test_dense_fp32_batch1_1x1(void)
{
    TEST_CASE("dense FP32 edge case batch=1, 1×1");

    float input_data[]  = {5.0f};
    float weight_data[] = {3.0f};
    float bias_data[]   = {1.0f};
    float output_data[1];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 1}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 1}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = input_data;
    output.data = output_data;

    edgenn_dense_params_t params;
    init_dense_params(&params);
    params.in_features  = 1;
    params.out_features = 1;
    params.fused_act    = EDGENN_ACT_NONE;
    edgenn_tensor_init(&params.weights, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.weights.data = weight_data;
    edgenn_tensor_init(&params.bias, (int32_t[]){1}, 1,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.bias.data = bias_data;

    ASSERT_OK(edgenn_dense_execute(&input, &output, &params, NULL));

    ASSERT_NEAR(output_data[0], 16.0f, 0.001f);  /* 5*3 + 1 */

    TEST_PASS();
}

/* ============================================================================
 * Dense INT8 Tests
 * ========================================================================= */

static void test_dense_int8_basic(void)
{
    TEST_CASE("dense INT8 basic per-tensor");

    /*
     * Simple 1×2 input, 2×2 weights, symmetric quantization.
     * Input:   [10, 20]   (INT8, zp=0, scale=0.1 → real [1.0, 2.0])
     * Weights: [[3, 4],   (INT8, zp=0, scale=0.1)
     *           [1, 2]]
     * Real result: [10*3+20*4, 10*1+20*2] = [110, 50] in INT32 accumulators
     * With multiplier ~0.5 (1073741824), shift=0:
     *   output[0] = requant(110, ~0.5, 0) + zp = 55
     *   output[1] = requant(50,  ~0.5, 0) + zp = 25
     */
    int8_t input_data[]  = {10, 20};
    int8_t weight_data[] = {3, 4,   /* row 0 */
                            1, 2};  /* row 1 */
    int32_t bias_data[]  = {0, 0};
    int8_t output_data[2];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 2}, 2, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 2}, 2, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NC);
    input.data  = input_data;
    output.data = output_data;

    /* Symmetric quantization: zp=0 */
    input.qparams.zero_point = 0;
    output.qparams.zero_point = 0;
    output.qparams.scale = 0.1f;
    output.qparams.multiplier = 1073741824; /* ~0.5 in Q31 */
    output.qparams.shift = 0;

    edgenn_dense_params_t params;
    init_dense_params(&params);
    params.in_features  = 2;
    params.out_features = 2;
    params.fused_act    = EDGENN_ACT_NONE;
    edgenn_tensor_init(&params.weights, (int32_t[]){2, 2}, 2,
                       EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NC);
    params.weights.data = weight_data;
    edgenn_tensor_init(&params.bias, (int32_t[]){2}, 1,
                       EDGENN_DTYPE_INT32, EDGENN_LAYOUT_NC);
    params.bias.data = bias_data;

    ASSERT_OK(edgenn_dense_execute(&input, &output, &params, NULL));

    ASSERT_EQ(output_data[0], 55);  /* requant(110, 0.5, 0) = 55 */
    ASSERT_EQ(output_data[1], 25);  /* requant(50, 0.5, 0) = 25 */

    TEST_PASS();
}

static void test_dense_int8_with_bias(void)
{
    TEST_CASE("dense INT8 with INT32 bias");

    int8_t input_data[]  = {10, 20};
    int8_t weight_data[] = {1, 1,   /* row 0: sum */
                            2, 0};  /* row 1: 2*input[0] */
    int32_t bias_data[]  = {100, 50};
    int8_t output_data[2];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 2}, 2, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 2}, 2, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NC);
    input.data  = input_data;
    output.data = output_data;

    input.qparams.zero_point = 0;
    output.qparams.zero_point = 0;
    output.qparams.scale = 0.1f;
    output.qparams.multiplier = 1073741824; /* ~0.5 */
    output.qparams.shift = 0;

    edgenn_dense_params_t params;
    init_dense_params(&params);
    params.in_features  = 2;
    params.out_features = 2;
    params.fused_act    = EDGENN_ACT_NONE;
    edgenn_tensor_init(&params.weights, (int32_t[]){2, 2}, 2,
                       EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NC);
    params.weights.data = weight_data;
    edgenn_tensor_init(&params.bias, (int32_t[]){2}, 1,
                       EDGENN_DTYPE_INT32, EDGENN_LAYOUT_NC);
    params.bias.data = bias_data;

    ASSERT_OK(edgenn_dense_execute(&input, &output, &params, NULL));

    /* acc[0] = 100 + 10*1 + 20*1 = 130 → requant(130, 0.5, 0) = 65 */
    /* acc[1] = 50 + 10*2 + 20*0 = 70 → requant(70, 0.5, 0) = 35 */
    ASSERT_EQ(output_data[0], 65);
    ASSERT_EQ(output_data[1], 35);

    TEST_PASS();
}

static void test_dense_int8_relu(void)
{
    TEST_CASE("dense INT8 with fused ReLU");

    /* Input so that one output is negative before relu */
    int8_t input_data[]  = {10, -20};
    int8_t weight_data[] = {1, 1,    /* row 0: 10 + (-20) = -10 */
                            1, -1};  /* row 1: 10 - (-20) = 30  */
    int8_t output_data[2];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 2}, 2, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 2}, 2, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NC);
    input.data  = input_data;
    output.data = output_data;

    input.qparams.zero_point = 0;
    output.qparams.zero_point = 0;
    output.qparams.scale = 0.1f;
    output.qparams.multiplier = 1073741824; /* ~0.5 */
    output.qparams.shift = 0;

    edgenn_dense_params_t params;
    init_dense_params(&params);
    params.in_features  = 2;
    params.out_features = 2;
    params.fused_act    = EDGENN_ACT_RELU;
    edgenn_tensor_init(&params.weights, (int32_t[]){2, 2}, 2,
                       EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NC);
    params.weights.data = weight_data;

    ASSERT_OK(edgenn_dense_execute(&input, &output, &params, NULL));

    /* acc[0] = -10 → requant(-10, 0.5, 0) = -5 → relu → 0 */
    /* acc[1] = 30 → requant(30, 0.5, 0) = 15 → relu → 15 */
    ASSERT_EQ(output_data[0], 0);
    ASSERT_EQ(output_data[1], 15);

    TEST_PASS();
}

static void test_dense_int8_per_channel(void)
{
    TEST_CASE("dense INT8 per-channel quantization");

    int8_t input_data[]  = {10, 20};
    int8_t weight_data[] = {3, 4,
                            1, 2};
    int32_t bias_data[]  = {0, 0};
    int8_t output_data[2];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 2}, 2, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 2}, 2, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NC);
    input.data  = input_data;
    output.data = output_data;

    input.qparams.zero_point = 0;
    output.qparams.zero_point = 0;
    output.qparams.scale = 0.1f;

    /* Per-channel: different multiplier/shift per output feature */
    int32_t ch_mult[]  = {1073741824, 1073741824}; /* both ~0.5 */
    int8_t  ch_shift[] = {0, 1};  /* channel 1 has extra right-shift */

    edgenn_dense_params_t params;
    init_dense_params(&params);
    params.in_features  = 2;
    params.out_features = 2;
    params.fused_act    = EDGENN_ACT_NONE;
    params.output_mult  = ch_mult;
    params.output_shift = ch_shift;
    edgenn_tensor_init(&params.weights, (int32_t[]){2, 2}, 2,
                       EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NC);
    params.weights.data = weight_data;
    edgenn_tensor_init(&params.bias, (int32_t[]){2}, 1,
                       EDGENN_DTYPE_INT32, EDGENN_LAYOUT_NC);
    params.bias.data = bias_data;

    ASSERT_OK(edgenn_dense_execute(&input, &output, &params, NULL));

    /* acc[0] = 10*3+20*4 = 110 → requant(110, 0.5, shift=0) = 55 */
    /* acc[1] = 10*1+20*2 = 50 → requant(50, 0.5, shift=1) = 13 (50*0.5=25, 25>>1=13 rounded) */
    ASSERT_EQ(output_data[0], 55);
    ASSERT_EQ(output_data[1], 13);

    TEST_PASS();
}

static void test_dense_int8_saturation(void)
{
    TEST_CASE("dense INT8 output saturation");

    /* Large values that will exceed INT8 range after requant */
    int8_t input_data[]  = {127, 127};
    int8_t weight_data[] = {127, 127};  /* single output: 127*127 + 127*127 = 32258 */
    int8_t output_data[1];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 2}, 2, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 1}, 2, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NC);
    input.data  = input_data;
    output.data = output_data;

    input.qparams.zero_point = 0;
    output.qparams.zero_point = 0;
    output.qparams.scale = 1.0f;
    /* multiplier near 1.0 → should saturate */
    output.qparams.multiplier = 2147483647; /* ~1.0 in Q31 */
    output.qparams.shift = 0;

    edgenn_dense_params_t params;
    init_dense_params(&params);
    params.in_features  = 2;
    params.out_features = 1;
    params.fused_act    = EDGENN_ACT_NONE;
    edgenn_tensor_init(&params.weights, (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NC);
    params.weights.data = weight_data;

    ASSERT_OK(edgenn_dense_execute(&input, &output, &params, NULL));

    /* Should saturate to 127 */
    ASSERT_EQ(output_data[0], 127);

    TEST_PASS();
}

/* ============================================================================
 * Dense INT8 vs FP32 Accuracy Comparison
 * ========================================================================= */

static void test_dense_int8_vs_fp32(void)
{
    TEST_CASE("dense INT8 vs FP32 accuracy (2×3 → 2×2)");

    /* FP32 reference */
    float fp_input[]  = {0.5f, -0.3f, 0.8f,
                         1.0f,  0.2f, -0.5f};
    float fp_weight[] = {0.4f, -0.2f,  0.6f,
                         0.1f,  0.5f, -0.3f};
    float fp_bias[]   = {0.1f, -0.05f};
    float fp_output[4];

    edgenn_tensor_t fp_in, fp_out;
    edgenn_tensor_init(&fp_in,  (int32_t[]){2, 3}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&fp_out, (int32_t[]){2, 2}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    fp_in.data  = fp_input;
    fp_out.data = fp_output;

    edgenn_dense_params_t fp_params;
    init_dense_params(&fp_params);
    fp_params.in_features  = 3;
    fp_params.out_features = 2;
    fp_params.fused_act    = EDGENN_ACT_NONE;
    edgenn_tensor_init(&fp_params.weights, (int32_t[]){2, 3}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    fp_params.weights.data = fp_weight;
    edgenn_tensor_init(&fp_params.bias, (int32_t[]){2}, 1,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    fp_params.bias.data = fp_bias;

    ASSERT_OK(edgenn_dense_execute(&fp_in, &fp_out, &fp_params, NULL));

    /* Now quantize and run INT8 path */
    float input_scale  = 1.0f / 127.0f;  /* range [-1, 1] */
    float weight_scale = 0.6f / 127.0f;  /* range [-0.6, 0.6] */
    float output_scale = 2.0f / 127.0f;  /* range [-2, 2] */
    float real_mult    = (input_scale * weight_scale) / output_scale;

    int32_t q_mult;
    int8_t  q_shift;
    edgenn_quant_compute_multiplier(real_mult, &q_mult, &q_shift);

    int8_t q_input[6], q_weight[6];
    edgenn_quant_fp32_to_int8(fp_input,  q_input,  6, input_scale,  0);
    edgenn_quant_fp32_to_int8(fp_weight, q_weight, 6, weight_scale, 0);

    /* Quantize bias in INT32: bias_q = round(bias / (input_scale * weight_scale)) */
    float bias_scale = input_scale * weight_scale;
    int32_t q_bias[2];
    for (int i = 0; i < 2; i++) {
        q_bias[i] = (int32_t)(fp_bias[i] / bias_scale + 0.5f);
    }

    int8_t q_output[4];

    edgenn_tensor_t q_in, q_out;
    edgenn_tensor_init(&q_in,  (int32_t[]){2, 3}, 2, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&q_out, (int32_t[]){2, 2}, 2, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NC);
    q_in.data  = q_input;
    q_out.data = q_output;

    q_in.qparams.zero_point  = 0;
    q_in.qparams.scale       = input_scale;
    q_out.qparams.zero_point = 0;
    q_out.qparams.scale      = output_scale;
    q_out.qparams.multiplier = q_mult;
    q_out.qparams.shift      = q_shift;

    edgenn_dense_params_t q_params;
    init_dense_params(&q_params);
    q_params.in_features  = 3;
    q_params.out_features = 2;
    q_params.fused_act    = EDGENN_ACT_NONE;
    edgenn_tensor_init(&q_params.weights, (int32_t[]){2, 3}, 2,
                       EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NC);
    q_params.weights.data = q_weight;
    edgenn_tensor_init(&q_params.bias, (int32_t[]){2}, 1,
                       EDGENN_DTYPE_INT32, EDGENN_LAYOUT_NC);
    q_params.bias.data = q_bias;

    ASSERT_OK(edgenn_dense_execute(&q_in, &q_out, &q_params, NULL));

    /* Dequantize INT8 output and compare with FP32 reference */
    float dequant_output[4];
    edgenn_quant_int8_to_fp32(q_output, dequant_output, 4, output_scale, 0);

    /* Tolerance: ~2 quantization steps for accumulated error */
    float tol = 2.0f * output_scale;
    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(dequant_output[i], fp_output[i], tol);
    }

    TEST_PASS();
}

/* ============================================================================
 * Dense Error / Edge Case Tests
 * ========================================================================= */

static void test_dense_null_ptrs(void)
{
    TEST_CASE("dense NULL pointer checks");

    edgenn_tensor_t input, output;
    edgenn_dense_params_t params;
    init_dense_params(&params);

    ASSERT_ERR(edgenn_dense_execute(NULL, &output, &params, NULL),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_dense_execute(&input, NULL, &params, NULL),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_dense_execute(&input, &output, NULL, NULL),
               EDGENN_ERR_NULL_PTR);

    TEST_PASS();
}

static void test_dense_shape_mismatch(void)
{
    TEST_CASE("dense shape mismatch error");

    float input_data[4], output_data[4];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){2, 3}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){2, 2}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = input_data;
    output.data = output_data;

    float weight_data[8];
    edgenn_dense_params_t params;
    init_dense_params(&params);
    params.in_features  = 4;  /* mismatch: input has 3 features */
    params.out_features = 2;
    params.fused_act    = EDGENN_ACT_NONE;
    edgenn_tensor_init(&params.weights, (int32_t[]){2, 4}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.weights.data = weight_data;

    ASSERT_ERR(edgenn_dense_execute(&input, &output, &params, NULL),
               EDGENN_ERR_SHAPE_MISMATCH);

    TEST_PASS();
}

static void test_dense_dtype_mismatch(void)
{
    TEST_CASE("dense dtype mismatch error");

    float input_data[2];
    float output_data[2];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 2}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 2}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = input_data;
    output.data = output_data;

    int8_t weight_data[4];
    edgenn_dense_params_t params;
    init_dense_params(&params);
    params.in_features  = 2;
    params.out_features = 2;
    params.fused_act    = EDGENN_ACT_NONE;
    edgenn_tensor_init(&params.weights, (int32_t[]){2, 2}, 2,
                       EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NC);
    params.weights.data = weight_data;

    ASSERT_ERR(edgenn_dense_execute(&input, &output, &params, NULL),
               EDGENN_ERR_DTYPE_MISMATCH);

    TEST_PASS();
}

static void test_dense_scratch_size(void)
{
    TEST_CASE("dense scratch_size returns 0");

    edgenn_dense_params_t params;
    init_dense_params(&params);
    params.in_features  = 128;
    params.out_features = 64;

    ASSERT_EQ(edgenn_dense_scratch_size(&params), 0u);

    TEST_PASS();
}

/* ============================================================================
 * Conv2D Tests
 * ========================================================================= */

static void init_conv2d_params(edgenn_conv2d_params_t *p)
{
    memset(p, 0, sizeof(*p));
}

static void test_conv2d_fp32_basic(void)
{
    TEST_CASE("conv2d FP32 basic 4x4, 2x2 kernel, valid");

    float input_data[16];
    for (int i = 0; i < 16; i++) input_data[i] = 1.0f;

    float weight_data[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float output_data[9];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 4, 4, 1}, 4, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NHWC);
    edgenn_tensor_init(&output, (int32_t[]){1, 3, 3, 1}, 4, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NHWC);
    input.data  = input_data;
    output.data = output_data;

    edgenn_conv2d_params_t params;
    init_conv2d_params(&params);
    params.kernel_h = 2; params.kernel_w = 2;
    params.stride_h = 1; params.stride_w = 1;
    params.padding  = EDGENN_PAD_VALID;
    params.fused_act = EDGENN_ACT_NONE;
    edgenn_tensor_init(&params.weights, (int32_t[]){1, 2, 2, 1}, 4,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NHWC);
    params.weights.data = weight_data;

    ASSERT_OK(edgenn_conv2d_execute(&input, &output, &params, NULL));

    for (int i = 0; i < 9; i++) {
        ASSERT_NEAR(output_data[i], 4.0f, 0.001f);
    }

    TEST_PASS();
}

static void test_conv2d_fp32_same_padding(void)
{
    TEST_CASE("conv2d FP32 SAME padding 3x3, 3x3 kernel");

    float input_data[9];
    for (int i = 0; i < 9; i++) input_data[i] = 1.0f;

    float weight_data[9];
    for (int i = 0; i < 9; i++) weight_data[i] = 1.0f;

    float output_data[9];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 3, 3, 1}, 4, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NHWC);
    edgenn_tensor_init(&output, (int32_t[]){1, 3, 3, 1}, 4, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NHWC);
    input.data  = input_data;
    output.data = output_data;

    edgenn_conv2d_params_t params;
    init_conv2d_params(&params);
    params.kernel_h = 3; params.kernel_w = 3;
    params.stride_h = 1; params.stride_w = 1;
    params.padding  = EDGENN_PAD_SAME;
    params.fused_act = EDGENN_ACT_NONE;
    edgenn_tensor_init(&params.weights, (int32_t[]){1, 3, 3, 1}, 4,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NHWC);
    params.weights.data = weight_data;

    ASSERT_OK(edgenn_conv2d_execute(&input, &output, &params, NULL));

    /* Corners: 4 valid neighbors */
    ASSERT_NEAR(output_data[0], 4.0f, 0.001f);
    ASSERT_NEAR(output_data[2], 4.0f, 0.001f);
    ASSERT_NEAR(output_data[6], 4.0f, 0.001f);
    ASSERT_NEAR(output_data[8], 4.0f, 0.001f);
    /* Edges: 6 valid neighbors */
    ASSERT_NEAR(output_data[1], 6.0f, 0.001f);
    ASSERT_NEAR(output_data[3], 6.0f, 0.001f);
    ASSERT_NEAR(output_data[5], 6.0f, 0.001f);
    ASSERT_NEAR(output_data[7], 6.0f, 0.001f);
    /* Center: 9 valid neighbors */
    ASSERT_NEAR(output_data[4], 9.0f, 0.001f);

    TEST_PASS();
}

static void test_conv2d_fp32_relu(void)
{
    TEST_CASE("conv2d FP32 with fused ReLU");

    float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float weight_data[] = {
        1.0f, 1.0f, 1.0f, 1.0f,    /* out_ch 0: sum */
       -1.0f,-1.0f,-1.0f,-1.0f,    /* out_ch 1: neg sum */
    };
    float output_data[2];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 2, 2, 1}, 4, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NHWC);
    edgenn_tensor_init(&output, (int32_t[]){1, 1, 1, 2}, 4, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NHWC);
    input.data  = input_data;
    output.data = output_data;

    edgenn_conv2d_params_t params;
    init_conv2d_params(&params);
    params.kernel_h = 2; params.kernel_w = 2;
    params.stride_h = 1; params.stride_w = 1;
    params.padding  = EDGENN_PAD_VALID;
    params.fused_act = EDGENN_ACT_RELU;
    edgenn_tensor_init(&params.weights, (int32_t[]){2, 2, 2, 1}, 4,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NHWC);
    params.weights.data = weight_data;

    ASSERT_OK(edgenn_conv2d_execute(&input, &output, &params, NULL));

    ASSERT_NEAR(output_data[0], 10.0f, 0.001f);
    ASSERT_NEAR(output_data[1],  0.0f, 0.001f);

    TEST_PASS();
}

static void test_conv2d_int8_basic(void)
{
    TEST_CASE("conv2d INT8 basic 2x2, 2x2 kernel");

    int8_t input_data[] = {10, 20, 30, 40};
    int8_t weight_data[] = {1, 1, 1, 1};
    int8_t output_data[1];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 2, 2, 1}, 4, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NHWC);
    edgenn_tensor_init(&output, (int32_t[]){1, 1, 1, 1}, 4, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NHWC);
    input.data  = input_data;
    output.data = output_data;

    input.qparams.zero_point = 0;
    output.qparams.zero_point = 0;
    output.qparams.scale = 1.0f;
    output.qparams.multiplier = 1073741824;
    output.qparams.shift = 0;

    edgenn_conv2d_params_t params;
    init_conv2d_params(&params);
    params.kernel_h = 2; params.kernel_w = 2;
    params.stride_h = 1; params.stride_w = 1;
    params.padding  = EDGENN_PAD_VALID;
    params.fused_act = EDGENN_ACT_NONE;
    edgenn_tensor_init(&params.weights, (int32_t[]){1, 2, 2, 1}, 4,
                       EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NHWC);
    params.weights.data = weight_data;

    ASSERT_OK(edgenn_conv2d_execute(&input, &output, &params, NULL));

    ASSERT_EQ(output_data[0], 50);

    TEST_PASS();
}

static void test_conv2d_null_ptrs(void)
{
    TEST_CASE("conv2d NULL pointer checks");

    edgenn_tensor_t input, output;
    edgenn_conv2d_params_t params;
    init_conv2d_params(&params);

    ASSERT_ERR(edgenn_conv2d_execute(NULL, &output, &params, NULL),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_conv2d_execute(&input, NULL, &params, NULL),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_conv2d_execute(&input, &output, NULL, NULL),
               EDGENN_ERR_NULL_PTR);

    TEST_PASS();
}

/* ============================================================================
 * Depthwise Conv2D Tests
 * ========================================================================= */

static void init_dwconv2d_params(edgenn_dwconv2d_params_t *p)
{
    memset(p, 0, sizeof(*p));
}

static void test_dwconv2d_fp32_basic(void)
{
    TEST_CASE("dwconv2d FP32 basic 3x3, 2x2 kernel, 2 ch");

    float input_data[18];
    for (int i = 0; i < 9; i++) {
        input_data[i * 2 + 0] = 1.0f;
        input_data[i * 2 + 1] = 2.0f;
    }

    float weight_data[] = {
        1.0f, 1.0f, 1.0f, 1.0f,  /* ch 0: sum filter */
        1.0f, 0.0f, 0.0f, 0.0f,  /* ch 1: top-left picker */
    };
    float output_data[8];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 3, 3, 2}, 4, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NHWC);
    edgenn_tensor_init(&output, (int32_t[]){1, 2, 2, 2}, 4, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NHWC);
    input.data  = input_data;
    output.data = output_data;

    edgenn_dwconv2d_params_t params;
    init_dwconv2d_params(&params);
    params.kernel_h = 2; params.kernel_w = 2;
    params.stride_h = 1; params.stride_w = 1;
    params.padding  = EDGENN_PAD_VALID;
    params.fused_act = EDGENN_ACT_NONE;
    edgenn_tensor_init(&params.weights, (int32_t[]){2, 2, 2}, 3,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NHWC);
    params.weights.data = weight_data;

    ASSERT_OK(edgenn_dwconv2d_execute(&input, &output, &params, NULL));

    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(output_data[i * 2 + 0], 4.0f, 0.001f);
        ASSERT_NEAR(output_data[i * 2 + 1], 2.0f, 0.001f);
    }

    TEST_PASS();
}

static void test_dwconv2d_fp32_with_bias(void)
{
    TEST_CASE("dwconv2d FP32 with bias");

    float input_data[18];
    for (int i = 0; i < 9; i++) {
        input_data[i * 2 + 0] = 1.0f;
        input_data[i * 2 + 1] = 1.0f;
    }

    float weight_data[] = {
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
    };
    float bias_data[] = {10.0f, 20.0f};
    float output_data[8];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 3, 3, 2}, 4, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NHWC);
    edgenn_tensor_init(&output, (int32_t[]){1, 2, 2, 2}, 4, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NHWC);
    input.data  = input_data;
    output.data = output_data;

    edgenn_dwconv2d_params_t params;
    init_dwconv2d_params(&params);
    params.kernel_h = 2; params.kernel_w = 2;
    params.stride_h = 1; params.stride_w = 1;
    params.padding  = EDGENN_PAD_VALID;
    params.fused_act = EDGENN_ACT_NONE;
    edgenn_tensor_init(&params.weights, (int32_t[]){2, 2, 2}, 3,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NHWC);
    params.weights.data = weight_data;
    edgenn_tensor_init(&params.bias, (int32_t[]){2}, 1,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.bias.data = bias_data;

    ASSERT_OK(edgenn_dwconv2d_execute(&input, &output, &params, NULL));

    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(output_data[i * 2 + 0], 14.0f, 0.001f);
        ASSERT_NEAR(output_data[i * 2 + 1], 24.0f, 0.001f);
    }

    TEST_PASS();
}

static void test_dwconv2d_int8_basic(void)
{
    TEST_CASE("dwconv2d INT8 basic 3x3, 2x2 kernel, 1 ch");

    int8_t input_data[9];
    for (int i = 0; i < 9; i++) input_data[i] = 10;

    int8_t weight_data[] = {1, 1, 1, 1};
    int8_t output_data[4];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 3, 3, 1}, 4, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NHWC);
    edgenn_tensor_init(&output, (int32_t[]){1, 2, 2, 1}, 4, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NHWC);
    input.data  = input_data;
    output.data = output_data;

    input.qparams.zero_point = 0;
    output.qparams.zero_point = 0;
    output.qparams.scale = 1.0f;
    output.qparams.multiplier = 1073741824;
    output.qparams.shift = 0;

    edgenn_dwconv2d_params_t params;
    init_dwconv2d_params(&params);
    params.kernel_h = 2; params.kernel_w = 2;
    params.stride_h = 1; params.stride_w = 1;
    params.padding  = EDGENN_PAD_VALID;
    params.fused_act = EDGENN_ACT_NONE;
    edgenn_tensor_init(&params.weights, (int32_t[]){1, 2, 2}, 3,
                       EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NHWC);
    params.weights.data = weight_data;

    ASSERT_OK(edgenn_dwconv2d_execute(&input, &output, &params, NULL));

    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(output_data[i], 20);
    }

    TEST_PASS();
}

static void test_dwconv2d_null_ptrs(void)
{
    TEST_CASE("dwconv2d NULL pointer checks");

    edgenn_tensor_t input, output;
    edgenn_dwconv2d_params_t params;
    init_dwconv2d_params(&params);

    ASSERT_ERR(edgenn_dwconv2d_execute(NULL, &output, &params, NULL),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_dwconv2d_execute(&input, NULL, &params, NULL),
               EDGENN_ERR_NULL_PTR);

    TEST_PASS();
}

/* ============================================================================
 * Pooling Tests
 * ========================================================================= */

static void test_maxpool2d_fp32(void)
{
    TEST_CASE("maxpool2d FP32 4x4, 2x2 kernel, stride 2");

    float input_data[] = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };
    float output_data[4];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 4, 4, 1}, 4, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NHWC);
    edgenn_tensor_init(&output, (int32_t[]){1, 2, 2, 1}, 4, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NHWC);
    input.data  = input_data;
    output.data = output_data;

    edgenn_pool_params_t params;
    memset(&params, 0, sizeof(params));
    params.kernel_h = 2; params.kernel_w = 2;
    params.stride_h = 2; params.stride_w = 2;
    params.padding  = EDGENN_PAD_VALID;

    ASSERT_OK(edgenn_maxpool2d_execute(&input, &output, &params));

    ASSERT_NEAR(output_data[0],  6.0f, 0.001f);
    ASSERT_NEAR(output_data[1],  8.0f, 0.001f);
    ASSERT_NEAR(output_data[2], 14.0f, 0.001f);
    ASSERT_NEAR(output_data[3], 16.0f, 0.001f);

    TEST_PASS();
}

static void test_avgpool2d_fp32(void)
{
    TEST_CASE("avgpool2d FP32 4x4, 2x2 kernel, stride 2");

    float input_data[] = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };
    float output_data[4];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 4, 4, 1}, 4, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NHWC);
    edgenn_tensor_init(&output, (int32_t[]){1, 2, 2, 1}, 4, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NHWC);
    input.data  = input_data;
    output.data = output_data;

    edgenn_pool_params_t params;
    memset(&params, 0, sizeof(params));
    params.kernel_h = 2; params.kernel_w = 2;
    params.stride_h = 2; params.stride_w = 2;
    params.padding  = EDGENN_PAD_VALID;

    ASSERT_OK(edgenn_avgpool2d_execute(&input, &output, &params));

    ASSERT_NEAR(output_data[0],  3.5f, 0.001f);
    ASSERT_NEAR(output_data[1],  5.5f, 0.001f);
    ASSERT_NEAR(output_data[2], 11.5f, 0.001f);
    ASSERT_NEAR(output_data[3], 13.5f, 0.001f);

    TEST_PASS();
}

static void test_global_avgpool_fp32(void)
{
    TEST_CASE("global_avgpool FP32 2x2x2 -> 2");

    float input_data[] = {
        1.0f, 10.0f,
        2.0f, 20.0f,
        3.0f, 30.0f,
        4.0f, 40.0f,
    };
    float output_data[2];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 2, 2, 2}, 4, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NHWC);
    edgenn_tensor_init(&output, (int32_t[]){1, 2}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = input_data;
    output.data = output_data;

    ASSERT_OK(edgenn_global_avgpool_execute(&input, &output));

    ASSERT_NEAR(output_data[0],  2.5f, 0.001f);
    ASSERT_NEAR(output_data[1], 25.0f, 0.001f);

    TEST_PASS();
}

static void test_maxpool2d_int8(void)
{
    TEST_CASE("maxpool2d INT8 4x4, 2x2 kernel, stride 2");

    int8_t input_data[] = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };
    int8_t output_data[4];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 4, 4, 1}, 4, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NHWC);
    edgenn_tensor_init(&output, (int32_t[]){1, 2, 2, 1}, 4, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NHWC);
    input.data  = input_data;
    output.data = output_data;

    edgenn_pool_params_t params;
    memset(&params, 0, sizeof(params));
    params.kernel_h = 2; params.kernel_w = 2;
    params.stride_h = 2; params.stride_w = 2;
    params.padding  = EDGENN_PAD_VALID;

    ASSERT_OK(edgenn_maxpool2d_execute(&input, &output, &params));

    ASSERT_EQ(output_data[0],  6);
    ASSERT_EQ(output_data[1],  8);
    ASSERT_EQ(output_data[2], 14);
    ASSERT_EQ(output_data[3], 16);

    TEST_PASS();
}

static void test_pool_null_ptrs(void)
{
    TEST_CASE("pool NULL pointer checks");

    edgenn_tensor_t output;
    edgenn_pool_params_t params;
    memset(&params, 0, sizeof(params));

    ASSERT_ERR(edgenn_maxpool2d_execute(NULL, &output, &params),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_avgpool2d_execute(NULL, &output, &params),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_global_avgpool_execute(NULL, &output),
               EDGENN_ERR_NULL_PTR);

    TEST_PASS();
}

/* ============================================================================
 * Activation Tests
 * ========================================================================= */

static void test_activation_fp32_relu(void)
{
    TEST_CASE("activation FP32 ReLU");

    float input_data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float output_data[5];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 5}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 5}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = input_data;
    output.data = output_data;

    ASSERT_OK(edgenn_activation_execute(&input, &output, EDGENN_ACT_RELU));

    ASSERT_NEAR(output_data[0], 0.0f, 0.001f);
    ASSERT_NEAR(output_data[1], 0.0f, 0.001f);
    ASSERT_NEAR(output_data[2], 0.0f, 0.001f);
    ASSERT_NEAR(output_data[3], 1.0f, 0.001f);
    ASSERT_NEAR(output_data[4], 2.0f, 0.001f);

    TEST_PASS();
}

static void test_activation_fp32_relu6(void)
{
    TEST_CASE("activation FP32 ReLU6");

    float input_data[] = {-1.0f, 0.0f, 3.0f, 6.0f, 10.0f};
    float output_data[5];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 5}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 5}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = input_data;
    output.data = output_data;

    ASSERT_OK(edgenn_activation_execute(&input, &output, EDGENN_ACT_RELU6));

    ASSERT_NEAR(output_data[0], 0.0f, 0.001f);
    ASSERT_NEAR(output_data[1], 0.0f, 0.001f);
    ASSERT_NEAR(output_data[2], 3.0f, 0.001f);
    ASSERT_NEAR(output_data[3], 6.0f, 0.001f);
    ASSERT_NEAR(output_data[4], 6.0f, 0.001f);

    TEST_PASS();
}

static void test_activation_fp32_sigmoid(void)
{
    TEST_CASE("activation FP32 sigmoid(0) = 0.5");

    float input_data[] = {0.0f};
    float output_data[1];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 1}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 1}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = input_data;
    output.data = output_data;

    ASSERT_OK(edgenn_activation_execute(&input, &output, EDGENN_ACT_SIGMOID));

    ASSERT_NEAR(output_data[0], 0.5f, 0.001f);

    TEST_PASS();
}

static void test_activation_int8_relu(void)
{
    TEST_CASE("activation INT8 ReLU");

    int8_t input_data[] = {-50, -1, 0, 1, 50};
    int8_t output_data[5];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 5}, 2, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 5}, 2, EDGENN_DTYPE_INT8, EDGENN_LAYOUT_NC);
    input.data  = input_data;
    output.data = output_data;
    input.qparams.zero_point = 0;

    ASSERT_OK(edgenn_activation_execute(&input, &output, EDGENN_ACT_RELU));

    ASSERT_EQ(output_data[0], 0);
    ASSERT_EQ(output_data[1], 0);
    ASSERT_EQ(output_data[2], 0);
    ASSERT_EQ(output_data[3], 1);
    ASSERT_EQ(output_data[4], 50);

    TEST_PASS();
}

static void test_softmax_fp32(void)
{
    TEST_CASE("softmax FP32 sums to 1");

    float input_data[] = {1.0f, 2.0f, 3.0f};
    float output_data[3];

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 3}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 3}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = input_data;
    output.data = output_data;

    ASSERT_OK(edgenn_softmax_execute(&input, &output));

    float sum = output_data[0] + output_data[1] + output_data[2];
    ASSERT_NEAR(sum, 1.0f, 0.001f);
    ASSERT_TRUE(output_data[0] < output_data[1]);
    ASSERT_TRUE(output_data[1] < output_data[2]);
    ASSERT_NEAR(output_data[0], 0.0900f, 0.01f);
    ASSERT_NEAR(output_data[1], 0.2447f, 0.01f);
    ASSERT_NEAR(output_data[2], 0.6652f, 0.01f);

    TEST_PASS();
}

static void test_activation_null_ptrs(void)
{
    TEST_CASE("activation NULL pointer checks");

    edgenn_tensor_t output;

    ASSERT_ERR(edgenn_activation_execute(NULL, &output, EDGENN_ACT_RELU),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_softmax_execute(NULL, &output),
               EDGENN_ERR_NULL_PTR);

    TEST_PASS();
}

/* ============================================================================
 * BatchNorm Tests
 * ========================================================================= */

static void test_batchnorm_fp32_4d(void)
{
    TEST_CASE("batchnorm FP32 NHWC 1x2x2x2");

    float input_data[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
    };
    float output_data[8];

    float gamma[] = {1.0f, 2.0f};
    float beta[]  = {0.0f, 1.0f};
    float mean[]  = {0.0f, 0.0f};
    float var[]   = {1.0f, 1.0f};

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 2, 2, 2}, 4, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NHWC);
    edgenn_tensor_init(&output, (int32_t[]){1, 2, 2, 2}, 4, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NHWC);
    input.data  = input_data;
    output.data = output_data;

    edgenn_batchnorm_params_t params;
    memset(&params, 0, sizeof(params));
    edgenn_tensor_init(&params.gamma,        (int32_t[]){2}, 1, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.beta,         (int32_t[]){2}, 1, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.running_mean, (int32_t[]){2}, 1, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.running_var,  (int32_t[]){2}, 1, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.gamma.data        = gamma;
    params.beta.data         = beta;
    params.running_mean.data = mean;
    params.running_var.data  = var;
    params.epsilon = 0.0f;

    ASSERT_OK(edgenn_batchnorm_execute(&input, &output, &params));

    /* Ch0: y=1*(x-0)/1+0 = x, Ch1: y=2*(x-0)/1+1 = 2x+1 */
    ASSERT_NEAR(output_data[0],  1.0f, 0.001f);
    ASSERT_NEAR(output_data[1],  5.0f, 0.001f);
    ASSERT_NEAR(output_data[2],  3.0f, 0.001f);
    ASSERT_NEAR(output_data[3],  9.0f, 0.001f);
    ASSERT_NEAR(output_data[4],  5.0f, 0.001f);
    ASSERT_NEAR(output_data[5], 13.0f, 0.001f);
    ASSERT_NEAR(output_data[6],  7.0f, 0.001f);
    ASSERT_NEAR(output_data[7], 17.0f, 0.001f);

    TEST_PASS();
}

static void test_batchnorm_fp32_2d(void)
{
    TEST_CASE("batchnorm FP32 NC 2x3");

    float input_data[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };
    float output_data[6];

    float gamma[] = {1.0f, 1.0f, 1.0f};
    float beta[]  = {0.0f, 0.0f, 0.0f};
    float mean[]  = {1.0f, 2.0f, 3.0f};
    float var[]   = {1.0f, 1.0f, 1.0f};

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){2, 3}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){2, 3}, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = input_data;
    output.data = output_data;

    edgenn_batchnorm_params_t params;
    memset(&params, 0, sizeof(params));
    edgenn_tensor_init(&params.gamma,        (int32_t[]){3}, 1, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.beta,         (int32_t[]){3}, 1, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.running_mean, (int32_t[]){3}, 1, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.running_var,  (int32_t[]){3}, 1, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.gamma.data        = gamma;
    params.beta.data         = beta;
    params.running_mean.data = mean;
    params.running_var.data  = var;
    params.epsilon = 0.0f;

    ASSERT_OK(edgenn_batchnorm_execute(&input, &output, &params));

    /* Row 0: [1-1, 2-2, 3-3] = [0, 0, 0] */
    ASSERT_NEAR(output_data[0], 0.0f, 0.001f);
    ASSERT_NEAR(output_data[1], 0.0f, 0.001f);
    ASSERT_NEAR(output_data[2], 0.0f, 0.001f);
    /* Row 1: [4-1, 5-2, 6-3] = [3, 3, 3] */
    ASSERT_NEAR(output_data[3], 3.0f, 0.001f);
    ASSERT_NEAR(output_data[4], 3.0f, 0.001f);
    ASSERT_NEAR(output_data[5], 3.0f, 0.001f);

    TEST_PASS();
}

static void test_batchnorm_null_ptrs(void)
{
    TEST_CASE("batchnorm NULL pointer checks");

    edgenn_tensor_t input, output;
    edgenn_batchnorm_params_t params;
    memset(&params, 0, sizeof(params));

    ASSERT_ERR(edgenn_batchnorm_execute(NULL, &output, &params),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_batchnorm_execute(&input, NULL, &params),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_batchnorm_execute(&input, &output, NULL),
               EDGENN_ERR_NULL_PTR);

    TEST_PASS();
}

/* ============================================================================
 * Main
 * ========================================================================= */

int main(void)
{
    TEST_SUITE_BEGIN("EdgeNN Phase 2 — DNN Operators");

    /* Dense FP32 tests */
    test_dense_fp32_basic();
    test_dense_fp32_with_bias();
    test_dense_fp32_relu();
    test_dense_fp32_batch1_1x1();

    /* Dense INT8 tests */
    test_dense_int8_basic();
    test_dense_int8_with_bias();
    test_dense_int8_relu();
    test_dense_int8_per_channel();
    test_dense_int8_saturation();
    test_dense_int8_vs_fp32();

    /* Dense error handling tests */
    test_dense_null_ptrs();
    test_dense_shape_mismatch();
    test_dense_dtype_mismatch();
    test_dense_scratch_size();

    /* Conv2D tests */
    test_conv2d_fp32_basic();
    test_conv2d_fp32_same_padding();
    test_conv2d_fp32_relu();
    test_conv2d_int8_basic();
    test_conv2d_null_ptrs();

    /* Depthwise Conv2D tests */
    test_dwconv2d_fp32_basic();
    test_dwconv2d_fp32_with_bias();
    test_dwconv2d_int8_basic();
    test_dwconv2d_null_ptrs();

    /* Pooling tests */
    test_maxpool2d_fp32();
    test_avgpool2d_fp32();
    test_global_avgpool_fp32();
    test_maxpool2d_int8();
    test_pool_null_ptrs();

    /* Activation tests */
    test_activation_fp32_relu();
    test_activation_fp32_relu6();
    test_activation_fp32_sigmoid();
    test_activation_int8_relu();
    test_softmax_fp32();
    test_activation_null_ptrs();

    /* BatchNorm tests */
    test_batchnorm_fp32_4d();
    test_batchnorm_fp32_2d();
    test_batchnorm_null_ptrs();

    TEST_SUITE_END();
}
