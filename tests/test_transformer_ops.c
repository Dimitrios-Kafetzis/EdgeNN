/**
 * @file test_transformer_ops.c
 * @brief Unit tests for Phase 4 Transformer operators
 *        (LayerNorm, PosEnc, FFN, Multi-Head Attention)
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn_test.h"
#include "edgenn/edgenn.h"
#include <string.h>
#include <math.h>

/* Scratch arena buffer shared by transformer tests */
static uint8_t scratch_buf[32768] __attribute__((aligned(16)));

static edgenn_arena_t make_scratch(void)
{
    edgenn_arena_t arena;
    edgenn_arena_init(&arena, scratch_buf, sizeof(scratch_buf));
    return arena;
}

/* ============================================================================
 * Layer Normalization Tests
 * ========================================================================= */

static void test_layernorm_fp32_basic(void)
{
    TEST_CASE("layernorm FP32 basic [2×3]");

    /* Two rows of 3 elements, gamma=1, beta=0, eps=0 */
    float x[] = {1.0f, 2.0f, 3.0f,
                 4.0f, 4.0f, 4.0f};
    float out[6];

    float gamma[] = {1.0f, 1.0f, 1.0f};
    float beta[]  = {0.0f, 0.0f, 0.0f};

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){2, 3}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){2, 3}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = x;
    output.data = out;

    edgenn_layernorm_params_t params;
    memset(&params, 0, sizeof(params));
    params.normalized_shape = 3;
    params.epsilon = 1e-5f;
    edgenn_tensor_init(&params.gamma, (int32_t[]){3}, 1,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.beta,  (int32_t[]){3}, 1,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.gamma.data = gamma;
    params.beta.data  = beta;

    ASSERT_OK(edgenn_layernorm_execute(&input, &output, &params));

    /* Row 0: mean=2, var=2/3 -> normalized = [-1.2247, 0, 1.2247] */
    ASSERT_NEAR(out[0], -1.2247f, 0.01f);
    ASSERT_NEAR(out[1],  0.0f,    0.01f);
    ASSERT_NEAR(out[2],  1.2247f, 0.01f);

    /* Row 1: all 4s -> mean=4, var=0 -> normalized ≈ [0, 0, 0] */
    ASSERT_NEAR(out[3], 0.0f, 0.01f);
    ASSERT_NEAR(out[4], 0.0f, 0.01f);
    ASSERT_NEAR(out[5], 0.0f, 0.01f);

    TEST_PASS();
}

static void test_layernorm_fp32_gamma_beta(void)
{
    TEST_CASE("layernorm FP32 with gamma/beta");

    /* Single row, gamma=2, beta=10 */
    float x[]   = {1.0f, 2.0f, 3.0f};
    float out[3];
    float gamma[] = {2.0f, 2.0f, 2.0f};
    float beta[]  = {10.0f, 10.0f, 10.0f};

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 3}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 3}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = x;
    output.data = out;

    edgenn_layernorm_params_t params;
    memset(&params, 0, sizeof(params));
    params.normalized_shape = 3;
    params.epsilon = 1e-5f;
    edgenn_tensor_init(&params.gamma, (int32_t[]){3}, 1,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.beta,  (int32_t[]){3}, 1,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.gamma.data = gamma;
    params.beta.data  = beta;

    ASSERT_OK(edgenn_layernorm_execute(&input, &output, &params));

    /* normalized ≈ [-1.2247, 0, 1.2247], then *2 + 10 */
    ASSERT_NEAR(out[0],  7.5505f, 0.02f);
    ASSERT_NEAR(out[1], 10.0f,    0.02f);
    ASSERT_NEAR(out[2], 12.4495f, 0.02f);

    TEST_PASS();
}

static void test_layernorm_fp32_3d(void)
{
    TEST_CASE("layernorm FP32 3D [2×2×3]");

    float x[] = {1, 2, 3,  4, 5, 6,
                 7, 8, 9,  10, 10, 10};
    float out[12];
    float gamma[] = {1, 1, 1};
    float beta[]  = {0, 0, 0};

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){2, 2, 3}, 3,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){2, 2, 3}, 3,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = x;
    output.data = out;

    edgenn_layernorm_params_t params;
    memset(&params, 0, sizeof(params));
    params.normalized_shape = 3;
    params.epsilon = 1e-5f;
    edgenn_tensor_init(&params.gamma, (int32_t[]){3}, 1,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.beta,  (int32_t[]){3}, 1,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.gamma.data = gamma;
    params.beta.data  = beta;

    ASSERT_OK(edgenn_layernorm_execute(&input, &output, &params));

    /* Row 0: [1,2,3] -> mean=2 -> normalized [-1.22, 0, 1.22] */
    ASSERT_NEAR(out[0], -1.2247f, 0.01f);
    ASSERT_NEAR(out[1],  0.0f,    0.01f);
    ASSERT_NEAR(out[2],  1.2247f, 0.01f);

    /* Last row: [10,10,10] -> all zero */
    ASSERT_NEAR(out[9],  0.0f, 0.01f);
    ASSERT_NEAR(out[10], 0.0f, 0.01f);
    ASSERT_NEAR(out[11], 0.0f, 0.01f);

    TEST_PASS();
}

static void test_layernorm_null_ptrs(void)
{
    TEST_CASE("layernorm NULL pointer checks");

    edgenn_tensor_t input, output;
    edgenn_layernorm_params_t params;
    memset(&params, 0, sizeof(params));

    ASSERT_ERR(edgenn_layernorm_execute(NULL, &output, &params),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_layernorm_execute(&input, NULL, &params),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_layernorm_execute(&input, &output, NULL),
               EDGENN_ERR_NULL_PTR);

    TEST_PASS();
}

/* ============================================================================
 * Positional Encoding Tests
 * ========================================================================= */

static void test_posenc_generate_sinusoidal(void)
{
    TEST_CASE("posenc generate sinusoidal 4×4");

    float table[16];
    ASSERT_OK(edgenn_posenc_generate_sinusoidal(table, 4, 4));

    /* Position 0: sin(0)=0, cos(0)=1, sin(0)=0, cos(0)=1 */
    ASSERT_NEAR(table[0], 0.0f, 0.001f);  /* sin(0/1) */
    ASSERT_NEAR(table[1], 1.0f, 0.001f);  /* cos(0/1) */
    ASSERT_NEAR(table[2], 0.0f, 0.001f);  /* sin(0/100) */
    ASSERT_NEAR(table[3], 1.0f, 0.001f);  /* cos(0/100) */

    /* Position 1: sin(1), cos(1), sin(1/100), cos(1/100) */
    ASSERT_NEAR(table[4], sinf(1.0f),       0.001f);
    ASSERT_NEAR(table[5], cosf(1.0f),       0.001f);
    ASSERT_NEAR(table[6], sinf(1.0f/100.0f), 0.001f);
    ASSERT_NEAR(table[7], cosf(1.0f/100.0f), 0.001f);

    TEST_PASS();
}

static void test_posenc_apply_2d(void)
{
    TEST_CASE("posenc apply 2D [2×4]");

    /* Input: 2 tokens × 4 dims, all ones */
    float x[8], out[8];
    for (int i = 0; i < 8; i++) x[i] = 1.0f;

    /* Generate encoding table */
    float table[20]; /* 5 × 4 */
    edgenn_posenc_generate_sinusoidal(table, 5, 4);

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){2, 4}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){2, 4}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = x;
    output.data = out;

    edgenn_posenc_params_t params;
    memset(&params, 0, sizeof(params));
    params.type = EDGENN_POSENC_SINUSOIDAL;
    params.max_seq_len = 5;
    params.d_model = 4;
    edgenn_tensor_init(&params.encoding_table, (int32_t[]){5, 4}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.encoding_table.data = table;

    ASSERT_OK(edgenn_posenc_apply(&input, &output, &params, 0));

    /* out[0] = 1 + PE(0,0) = 1 + sin(0) = 1 + 0 = 1 */
    ASSERT_NEAR(out[0], 1.0f + sinf(0.0f), 0.001f);
    /* out[4] = 1 + PE(1,0) = 1 + sin(1) */
    ASSERT_NEAR(out[4], 1.0f + sinf(1.0f), 0.001f);

    TEST_PASS();
}

static void test_posenc_apply_with_offset(void)
{
    TEST_CASE("posenc apply with position offset");

    float x[4], out[4];
    for (int i = 0; i < 4; i++) x[i] = 0.0f;

    float table[20];
    edgenn_posenc_generate_sinusoidal(table, 5, 4);

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 4}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 4}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = x;
    output.data = out;

    edgenn_posenc_params_t params;
    memset(&params, 0, sizeof(params));
    params.type = EDGENN_POSENC_SINUSOIDAL;
    params.max_seq_len = 5;
    params.d_model = 4;
    edgenn_tensor_init(&params.encoding_table, (int32_t[]){5, 4}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.encoding_table.data = table;

    /* Apply at offset=3 — should use PE[3] */
    ASSERT_OK(edgenn_posenc_apply(&input, &output, &params, 3));

    ASSERT_NEAR(out[0], table[3 * 4 + 0], 0.001f);
    ASSERT_NEAR(out[1], table[3 * 4 + 1], 0.001f);

    TEST_PASS();
}

static void test_posenc_null_ptrs(void)
{
    TEST_CASE("posenc NULL pointer checks");

    edgenn_tensor_t input, output;
    edgenn_posenc_params_t params;
    memset(&params, 0, sizeof(params));

    ASSERT_ERR(edgenn_posenc_apply(NULL, &output, &params, 0),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_posenc_apply(&input, NULL, &params, 0),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_posenc_apply(&input, &output, NULL, 0),
               EDGENN_ERR_NULL_PTR);

    ASSERT_ERR(edgenn_posenc_generate_sinusoidal(NULL, 10, 4),
               EDGENN_ERR_NULL_PTR);

    TEST_PASS();
}

static void test_posenc_generate_invalid_args(void)
{
    TEST_CASE("posenc generate invalid args");

    float table[4];
    ASSERT_ERR(edgenn_posenc_generate_sinusoidal(table, 0, 4),
               EDGENN_ERR_INVALID_ARG);
    ASSERT_ERR(edgenn_posenc_generate_sinusoidal(table, 4, 0),
               EDGENN_ERR_INVALID_ARG);

    TEST_PASS();
}

/* ============================================================================
 * Feed-Forward Network Tests
 * ========================================================================= */

static void test_ffn_fp32_identity(void)
{
    TEST_CASE("ffn FP32 identity (no activation)");

    /* d_model=2, d_ff=2. fc1 = identity, fc2 = identity, no activation.
     * Result should be input * I * I = input */
    float x[]  = {1.0f, 2.0f};
    float out[2];

    float w1[] = {1, 0, 0, 1};  /* [2×2] identity */
    float w2[] = {1, 0, 0, 1};

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = x;
    output.data = out;

    edgenn_ffn_params_t params;
    memset(&params, 0, sizeof(params));
    params.d_model = 2;
    params.d_ff    = 2;
    params.activation = EDGENN_ACT_NONE;

    /* fc1 */
    params.fc1.in_features  = 2;
    params.fc1.out_features = 2;
    params.fc1.fused_act    = EDGENN_ACT_NONE;
    edgenn_tensor_init(&params.fc1.weights, (int32_t[]){2, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.fc1.weights.data = w1;

    /* fc2 */
    params.fc2.in_features  = 2;
    params.fc2.out_features = 2;
    params.fc2.fused_act    = EDGENN_ACT_NONE;
    edgenn_tensor_init(&params.fc2.weights, (int32_t[]){2, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.fc2.weights.data = w2;

    edgenn_arena_t scratch = make_scratch();

    ASSERT_OK(edgenn_ffn_execute(&input, &output, &params, &scratch));

    ASSERT_NEAR(out[0], 1.0f, 0.001f);
    ASSERT_NEAR(out[1], 2.0f, 0.001f);

    TEST_PASS();
}

static void test_ffn_fp32_with_relu(void)
{
    TEST_CASE("ffn FP32 with ReLU activation");

    /* d_model=2, d_ff=2
     * fc1 weights = [[-1, 0], [0, 1]], fc2 = identity
     * x = [1, 2]
     * fc1(x) = [-1, 2], relu -> [0, 2]
     * fc2([0, 2]) = [0, 2] */
    float x[]  = {1.0f, 2.0f};
    float out[2];

    float w1[] = {-1, 0, 0, 1};
    float w2[] = {1, 0, 0, 1};

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = x;
    output.data = out;

    edgenn_ffn_params_t params;
    memset(&params, 0, sizeof(params));
    params.d_model = 2;
    params.d_ff    = 2;
    params.activation = EDGENN_ACT_RELU;

    params.fc1.in_features  = 2;
    params.fc1.out_features = 2;
    params.fc1.fused_act    = EDGENN_ACT_NONE;
    edgenn_tensor_init(&params.fc1.weights, (int32_t[]){2, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.fc1.weights.data = w1;

    params.fc2.in_features  = 2;
    params.fc2.out_features = 2;
    params.fc2.fused_act    = EDGENN_ACT_NONE;
    edgenn_tensor_init(&params.fc2.weights, (int32_t[]){2, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.fc2.weights.data = w2;

    edgenn_arena_t scratch = make_scratch();

    ASSERT_OK(edgenn_ffn_execute(&input, &output, &params, &scratch));

    ASSERT_NEAR(out[0], 0.0f, 0.001f);
    ASSERT_NEAR(out[1], 2.0f, 0.001f);

    TEST_PASS();
}

static void test_ffn_fp32_expansion(void)
{
    TEST_CASE("ffn FP32 expansion d_ff > d_model");

    /* d_model=2, d_ff=4.
     * fc1: [4×2] all ones -> each output = sum of inputs
     * fc2: [2×4] = [[1,0,0,0],[0,1,0,0]] -> picks first two from d_ff */
    float x[]  = {1.0f, 2.0f};
    float out[2];

    float w1[] = {1,1, 1,1, 1,1, 1,1}; /* [4×2] all ones */
    float w2[] = {1,0,0,0,  0,1,0,0};  /* [2×4] picks first two */

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = x;
    output.data = out;

    edgenn_ffn_params_t params;
    memset(&params, 0, sizeof(params));
    params.d_model = 2;
    params.d_ff    = 4;
    params.activation = EDGENN_ACT_NONE;

    params.fc1.in_features  = 2;
    params.fc1.out_features = 4;
    params.fc1.fused_act    = EDGENN_ACT_NONE;
    edgenn_tensor_init(&params.fc1.weights, (int32_t[]){4, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.fc1.weights.data = w1;

    params.fc2.in_features  = 4;
    params.fc2.out_features = 2;
    params.fc2.fused_act    = EDGENN_ACT_NONE;
    edgenn_tensor_init(&params.fc2.weights, (int32_t[]){2, 4}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.fc2.weights.data = w2;

    edgenn_arena_t scratch = make_scratch();

    ASSERT_OK(edgenn_ffn_execute(&input, &output, &params, &scratch));

    /* fc1(x) = [3, 3, 3, 3], fc2 picks [3, 3] */
    ASSERT_NEAR(out[0], 3.0f, 0.001f);
    ASSERT_NEAR(out[1], 3.0f, 0.001f);

    TEST_PASS();
}

static void test_ffn_null_ptrs(void)
{
    TEST_CASE("ffn NULL pointer checks");

    edgenn_tensor_t input, output;
    edgenn_ffn_params_t params;
    memset(&params, 0, sizeof(params));
    edgenn_arena_t scratch = make_scratch();

    ASSERT_ERR(edgenn_ffn_execute(NULL, &output, &params, &scratch),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_ffn_execute(&input, NULL, &params, &scratch),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_ffn_execute(&input, &output, NULL, &scratch),
               EDGENN_ERR_NULL_PTR);

    TEST_PASS();
}

static void test_ffn_scratch_size(void)
{
    TEST_CASE("ffn scratch_size");

    edgenn_ffn_params_t params;
    memset(&params, 0, sizeof(params));
    params.d_model = 8;
    params.d_ff    = 32;

    size_t sz = edgenn_ffn_scratch_size(&params);
    ASSERT_TRUE(sz >= 32 * sizeof(float));
    ASSERT_TRUE(sz % EDGENN_TENSOR_ALIGN == 0);

    ASSERT_EQ(edgenn_ffn_scratch_size(NULL), 0u);

    TEST_PASS();
}

/* ============================================================================
 * Multi-Head Attention Tests
 * ========================================================================= */

static void test_attention_fp32_single_head_identity(void)
{
    TEST_CASE("attention FP32 1-head identity projections");

    /* d_model=2, n_heads=1, d_head=2, seq_len=2
     * Wq=Wk=Wv=Wo=I (identity), no bias
     * Input: [[1,0], [0,1]]
     * Q=K=V=input, scores = Q×K^T / sqrt(2)
     */
    int32_t D = 2, nh = 1, dh = 2, T = 2;

    float x[] = {1, 0,
                 0, 1};
    float out[4];

    float eye[] = {1, 0, 0, 1};

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){T, D}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){T, D}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = x;
    output.data = out;

    edgenn_attention_params_t params;
    memset(&params, 0, sizeof(params));
    params.d_model = D;
    params.n_heads = nh;
    params.d_head  = dh;
    params.max_seq_len = 16;

    edgenn_tensor_init(&params.wq, (int32_t[]){D, D}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.wk, (int32_t[]){D, D}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.wv, (int32_t[]){D, D}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.wo, (int32_t[]){D, D}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.wq.data = eye;
    params.wk.data = eye;
    params.wv.data = eye;
    params.wo.data = eye;

    edgenn_arena_t scratch = make_scratch();

    ASSERT_OK(edgenn_attention_execute(
        &input, &output, &params, NULL, &scratch, T));

    /* With identity Wq=Wk=Wv=Wo:
     * Q=K=V = [[1,0],[0,1]]
     * scores = Q·K^T/√2 = I/√2
     * attn = softmax(I/√2) -> each row is [p, 1-p] where p > 0.5
     * context = attn × V -> weighted average of rows
     * output = Wo × context = context
     *
     * Since Q=K=identity, score[0,0]=1/√2, score[0,1]=0
     * softmax([0.707, 0]) = [0.6681, 0.3319]
     * context[0] = 0.6681*[1,0] + 0.3319*[0,1] = [0.6681, 0.3319]
     */
    float s = 1.0f / sqrtf(2.0f);
    float e_s = expf(s);
    float e_0 = 1.0f;
    float p0 = e_s / (e_s + e_0);
    float p1 = e_0 / (e_s + e_0);

    ASSERT_NEAR(out[0], p0,  0.01f);
    ASSERT_NEAR(out[1], p1,  0.01f);
    ASSERT_NEAR(out[2], p1,  0.01f);
    ASSERT_NEAR(out[3], p0,  0.01f);

    TEST_PASS();
}

static void test_attention_fp32_single_token(void)
{
    TEST_CASE("attention FP32 single token seq_len=1");

    /* seq_len=1: softmax of single score = 1.0
     * output = Wo × (Wv × x) */
    int32_t D = 2;
    float x[] = {3.0f, 4.0f};
    float out[2];

    float eye[] = {1, 0, 0, 1};

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, D}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, D}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = x;
    output.data = out;

    edgenn_attention_params_t params;
    memset(&params, 0, sizeof(params));
    params.d_model = D;
    params.n_heads = 1;
    params.d_head  = D;
    params.max_seq_len = 16;
    edgenn_tensor_init(&params.wq, (int32_t[]){D, D}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.wk, (int32_t[]){D, D}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.wv, (int32_t[]){D, D}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.wo, (int32_t[]){D, D}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.wq.data = eye;
    params.wk.data = eye;
    params.wv.data = eye;
    params.wo.data = eye;

    edgenn_arena_t scratch = make_scratch();

    ASSERT_OK(edgenn_attention_execute(
        &input, &output, &params, NULL, &scratch, 1));

    /* Single token: attn=1.0, output = Wo * Wv * x = x */
    ASSERT_NEAR(out[0], 3.0f, 0.001f);
    ASSERT_NEAR(out[1], 4.0f, 0.001f);

    TEST_PASS();
}

static void test_attention_fp32_multi_head(void)
{
    TEST_CASE("attention FP32 2-head d_model=4");

    /* d_model=4, n_heads=2, d_head=2, seq_len=1
     * Single token so softmax=1.0, output = Wo * Wv * x */
    int32_t D = 4;
    float x[] = {1, 2, 3, 4};
    float out[4];

    float eye4[] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, D}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, D}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = x;
    output.data = out;

    edgenn_attention_params_t params;
    memset(&params, 0, sizeof(params));
    params.d_model = D;
    params.n_heads = 2;
    params.d_head  = 2;
    params.max_seq_len = 16;
    edgenn_tensor_init(&params.wq, (int32_t[]){D, D}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.wk, (int32_t[]){D, D}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.wv, (int32_t[]){D, D}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.wo, (int32_t[]){D, D}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.wq.data = eye4;
    params.wk.data = eye4;
    params.wv.data = eye4;
    params.wo.data = eye4;

    edgenn_arena_t scratch = make_scratch();

    ASSERT_OK(edgenn_attention_execute(
        &input, &output, &params, NULL, &scratch, 1));

    /* With all identity and single token -> output = input */
    ASSERT_NEAR(out[0], 1.0f, 0.001f);
    ASSERT_NEAR(out[1], 2.0f, 0.001f);
    ASSERT_NEAR(out[2], 3.0f, 0.001f);
    ASSERT_NEAR(out[3], 4.0f, 0.001f);

    TEST_PASS();
}

static void test_attention_null_ptrs(void)
{
    TEST_CASE("attention NULL pointer checks");

    edgenn_tensor_t input, output;
    edgenn_attention_params_t params;
    memset(&params, 0, sizeof(params));
    edgenn_arena_t scratch = make_scratch();

    ASSERT_ERR(edgenn_attention_execute(NULL, &output, &params, NULL, &scratch, 1),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_attention_execute(&input, NULL, &params, NULL, &scratch, 1),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_attention_execute(&input, &output, NULL, NULL, &scratch, 1),
               EDGENN_ERR_NULL_PTR);

    TEST_PASS();
}

static void test_attention_scratch_size(void)
{
    TEST_CASE("attention scratch_size");

    edgenn_attention_params_t params;
    memset(&params, 0, sizeof(params));
    params.d_model = 8;
    params.n_heads = 2;
    params.d_head  = 4;

    size_t sz = edgenn_attention_scratch_size(&params, 4);
    /* Q+K+V = 3*4*8*4 + scores = 2*4*4*4 + ctx = 4*8*4 */
    ASSERT_TRUE(sz > 0);
    ASSERT_TRUE(sz % EDGENN_TENSOR_ALIGN == 0);

    ASSERT_EQ(edgenn_attention_scratch_size(NULL, 4), 0u);
    ASSERT_EQ(edgenn_attention_scratch_size(&params, 0), 0u);

    TEST_PASS();
}

static void test_attention_invalid_seqlen(void)
{
    TEST_CASE("attention invalid seq_len");

    float x[] = {1, 2};
    float out[2];
    float eye[] = {1, 0, 0, 1};

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = x;
    output.data = out;

    edgenn_attention_params_t params;
    memset(&params, 0, sizeof(params));
    params.d_model = 2;
    params.n_heads = 1;
    params.d_head  = 2;
    edgenn_tensor_init(&params.wq, (int32_t[]){2, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.wk, (int32_t[]){2, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.wv, (int32_t[]){2, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.wo, (int32_t[]){2, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.wq.data = eye;
    params.wk.data = eye;
    params.wv.data = eye;
    params.wo.data = eye;

    edgenn_arena_t scratch = make_scratch();

    ASSERT_ERR(edgenn_attention_execute(
        &input, &output, &params, NULL, &scratch, 0),
        EDGENN_ERR_INVALID_ARG);

    TEST_PASS();
}

static void test_kv_cache_init_reset(void)
{
    TEST_CASE("kv_cache init and reset");

    edgenn_attention_params_t params;
    memset(&params, 0, sizeof(params));
    params.d_model = 4;
    params.n_heads = 2;
    params.d_head  = 2;
    params.max_seq_len = 8;

    edgenn_arena_t arena = make_scratch();
    edgenn_kv_cache_t cache;
    memset(&cache, 0, sizeof(cache));

    ASSERT_OK(edgenn_kv_cache_init(&cache, &params, &arena));
    ASSERT_TRUE(cache.k_cache.data != NULL);
    ASSERT_TRUE(cache.v_cache.data != NULL);
    ASSERT_EQ(cache.cached_len, 0);

    cache.cached_len = 5;
    edgenn_kv_cache_reset(&cache);
    ASSERT_EQ(cache.cached_len, 0);

    TEST_PASS();
}

/* ============================================================================
 * Main
 * ========================================================================= */

int main(void)
{
    TEST_SUITE_BEGIN("EdgeNN Phase 4 — Transformer Operators");

    /* LayerNorm tests */
    test_layernorm_fp32_basic();
    test_layernorm_fp32_gamma_beta();
    test_layernorm_fp32_3d();
    test_layernorm_null_ptrs();

    /* Positional Encoding tests */
    test_posenc_generate_sinusoidal();
    test_posenc_apply_2d();
    test_posenc_apply_with_offset();
    test_posenc_null_ptrs();
    test_posenc_generate_invalid_args();

    /* FFN tests */
    test_ffn_fp32_identity();
    test_ffn_fp32_with_relu();
    test_ffn_fp32_expansion();
    test_ffn_null_ptrs();
    test_ffn_scratch_size();

    /* Multi-Head Attention tests */
    test_attention_fp32_single_head_identity();
    test_attention_fp32_single_token();
    test_attention_fp32_multi_head();
    test_attention_null_ptrs();
    test_attention_scratch_size();
    test_attention_invalid_seqlen();
    test_kv_cache_init_reset();

    TEST_SUITE_END();
}
