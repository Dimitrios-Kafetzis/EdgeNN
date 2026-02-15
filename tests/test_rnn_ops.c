/**
 * @file test_rnn_ops.c
 * @brief Unit tests for Phase 3 RNN operators (Simple RNN, LSTM, GRU)
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn_test.h"
#include "edgenn/edgenn.h"
#include <string.h>
#include <math.h>

/* Scratch arena buffer shared by all RNN tests */
static uint8_t scratch_buf[4096] __attribute__((aligned(16)));

static edgenn_arena_t make_scratch(void)
{
    edgenn_arena_t arena;
    edgenn_arena_init(&arena, scratch_buf, sizeof(scratch_buf));
    return arena;
}

/* ============================================================================
 * Simple RNN Cell Tests
 * ========================================================================= */

static void test_rnn_cell_fp32_tanh(void)
{
    TEST_CASE("rnn_cell FP32 tanh identity weights");

    /* I=2, H=2, batch=1
     * W_ih = I (identity), W_hh = 0, bias = 0
     * h = tanh(x)
     */
    float x_data[] = {0.5f, -0.5f};
    float h_data[] = {0.0f, 0.0f};
    float w_ih[]   = {1.0f, 0.0f,
                      0.0f, 1.0f};
    float w_hh[]   = {0.0f, 0.0f,
                      0.0f, 0.0f};

    edgenn_tensor_t input;
    edgenn_tensor_init(&input, (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data = x_data;

    edgenn_rnn_state_t state;
    edgenn_tensor_init(&state.h, (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    state.h.data = h_data;

    edgenn_rnn_cell_params_t params;
    memset(&params, 0, sizeof(params));
    params.input_size  = 2;
    params.hidden_size = 2;
    params.activation  = EDGENN_ACT_TANH;
    edgenn_tensor_init(&params.weight_ih, (int32_t[]){2, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.weight_hh, (int32_t[]){2, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.weight_ih.data = w_ih;
    params.weight_hh.data = w_hh;

    edgenn_arena_t scratch = make_scratch();

    ASSERT_OK(edgenn_rnn_cell_execute(&input, &state, &params, &scratch));

    ASSERT_NEAR(h_data[0], tanhf(0.5f),  0.001f);
    ASSERT_NEAR(h_data[1], tanhf(-0.5f), 0.001f);

    TEST_PASS();
}

static void test_rnn_cell_fp32_relu(void)
{
    TEST_CASE("rnn_cell FP32 relu");

    float x_data[] = {-1.0f, 2.0f};
    float h_data[] = {0.0f, 0.0f};
    float w_ih[]   = {1.0f, 0.0f,
                      0.0f, 1.0f};
    float w_hh[]   = {0.0f, 0.0f,
                      0.0f, 0.0f};

    edgenn_tensor_t input;
    edgenn_tensor_init(&input, (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data = x_data;

    edgenn_rnn_state_t state;
    edgenn_tensor_init(&state.h, (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    state.h.data = h_data;

    edgenn_rnn_cell_params_t params;
    memset(&params, 0, sizeof(params));
    params.input_size  = 2;
    params.hidden_size = 2;
    params.activation  = EDGENN_ACT_RELU;
    edgenn_tensor_init(&params.weight_ih, (int32_t[]){2, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.weight_hh, (int32_t[]){2, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.weight_ih.data = w_ih;
    params.weight_hh.data = w_hh;

    edgenn_arena_t scratch = make_scratch();

    ASSERT_OK(edgenn_rnn_cell_execute(&input, &state, &params, &scratch));

    ASSERT_NEAR(h_data[0], 0.0f, 0.001f);  /* relu(-1) = 0 */
    ASSERT_NEAR(h_data[1], 2.0f, 0.001f);  /* relu(2)  = 2 */

    TEST_PASS();
}

static void test_rnn_cell_fp32_state_carry(void)
{
    TEST_CASE("rnn_cell FP32 state carry-over");

    /* I=1, H=1, batch=1
     * W_ih = [1], W_hh = [0.5], bias = 0
     * Step 1: h = tanh(1.0*x + 0.5*0) = tanh(1.0) = 0.7616
     * Step 2: h = tanh(1.0*x + 0.5*0.7616) = tanh(1.0 + 0.3808)
     */
    float x_data[] = {1.0f};
    float h_data[] = {0.0f};
    float w_ih[]   = {1.0f};
    float w_hh[]   = {0.5f};

    edgenn_tensor_t input;
    edgenn_tensor_init(&input, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data = x_data;

    edgenn_rnn_state_t state;
    edgenn_tensor_init(&state.h, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    state.h.data = h_data;

    edgenn_rnn_cell_params_t params;
    memset(&params, 0, sizeof(params));
    params.input_size  = 1;
    params.hidden_size = 1;
    params.activation  = EDGENN_ACT_TANH;
    edgenn_tensor_init(&params.weight_ih, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.weight_hh, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.weight_ih.data = w_ih;
    params.weight_hh.data = w_hh;

    edgenn_arena_t scratch = make_scratch();

    /* Step 1 */
    ASSERT_OK(edgenn_rnn_cell_execute(&input, &state, &params, &scratch));
    float h1 = tanhf(1.0f);
    ASSERT_NEAR(h_data[0], h1, 0.001f);

    /* Step 2 — h carries over from step 1 */
    edgenn_arena_reset(&scratch);
    ASSERT_OK(edgenn_rnn_cell_execute(&input, &state, &params, &scratch));
    float h2 = tanhf(1.0f + 0.5f * h1);
    ASSERT_NEAR(h_data[0], h2, 0.001f);

    TEST_PASS();
}

static void test_rnn_cell_fp32_with_bias(void)
{
    TEST_CASE("rnn_cell FP32 with bias");

    /* I=1, H=1, batch=1, W_ih=1, W_hh=0, bias=10 */
    float x_data[] = {0.0f};
    float h_data[] = {0.0f};
    float w_ih[]   = {1.0f};
    float w_hh[]   = {0.0f};
    float bias[]   = {10.0f};

    edgenn_tensor_t input;
    edgenn_tensor_init(&input, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data = x_data;

    edgenn_rnn_state_t state;
    edgenn_tensor_init(&state.h, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    state.h.data = h_data;

    edgenn_rnn_cell_params_t params;
    memset(&params, 0, sizeof(params));
    params.input_size  = 1;
    params.hidden_size = 1;
    params.activation  = EDGENN_ACT_TANH;
    edgenn_tensor_init(&params.weight_ih, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.weight_hh, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.bias, (int32_t[]){1}, 1,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.weight_ih.data = w_ih;
    params.weight_hh.data = w_hh;
    params.bias.data      = bias;

    edgenn_arena_t scratch = make_scratch();

    ASSERT_OK(edgenn_rnn_cell_execute(&input, &state, &params, &scratch));

    /* h = tanh(0 + 0 + 10) ≈ 1.0 */
    ASSERT_NEAR(h_data[0], tanhf(10.0f), 0.001f);

    TEST_PASS();
}

static void test_rnn_cell_null_ptrs(void)
{
    TEST_CASE("rnn_cell NULL pointer checks");

    edgenn_tensor_t input;
    edgenn_rnn_state_t state;
    edgenn_rnn_cell_params_t params;
    memset(&params, 0, sizeof(params));
    edgenn_arena_t scratch = make_scratch();

    ASSERT_ERR(edgenn_rnn_cell_execute(NULL, &state, &params, &scratch),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_rnn_cell_execute(&input, NULL, &params, &scratch),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_rnn_cell_execute(&input, &state, NULL, &scratch),
               EDGENN_ERR_NULL_PTR);

    TEST_PASS();
}

static void test_rnn_cell_scratch_size(void)
{
    TEST_CASE("rnn_cell scratch_size");

    edgenn_rnn_cell_params_t params;
    memset(&params, 0, sizeof(params));
    params.hidden_size = 16;

    size_t sz = edgenn_rnn_cell_scratch_size(&params);
    ASSERT_TRUE(sz >= 16 * sizeof(float));
    ASSERT_TRUE(sz % EDGENN_TENSOR_ALIGN == 0);

    ASSERT_EQ(edgenn_rnn_cell_scratch_size(NULL), 0u);

    TEST_PASS();
}

/* ============================================================================
 * LSTM Tests
 * ========================================================================= */

static void test_lstm_cell_fp32_zero_input(void)
{
    TEST_CASE("lstm_cell FP32 zero input, bias-only");

    /* I=1, H=1, batch=1
     * All weights zero, bias_ih = [0,0,0,0], bias_hh = [0,0,0,0]
     * All gates get 0: i=sig(0)=0.5, f=sig(0)=0.5, g=tanh(0)=0, o=sig(0)=0.5
     * c = 0.5*0 + 0.5*0 = 0
     * h = 0.5 * tanh(0) = 0
     */
    float x_data[] = {0.0f};
    float h_data[] = {0.0f};
    float c_data[] = {0.0f};
    float w_ih[4]  = {0}; /* [4*1 x 1] */
    float w_hh[4]  = {0}; /* [4*1 x 1] */

    edgenn_tensor_t input;
    edgenn_tensor_init(&input, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data = x_data;

    edgenn_lstm_state_t state;
    edgenn_tensor_init(&state.h, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&state.c, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    state.h.data = h_data;
    state.c.data = c_data;

    edgenn_lstm_params_t params;
    memset(&params, 0, sizeof(params));
    params.input_size  = 1;
    params.hidden_size = 1;
    edgenn_tensor_init(&params.weight_ih, (int32_t[]){4, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.weight_hh, (int32_t[]){4, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.weight_ih.data = w_ih;
    params.weight_hh.data = w_hh;

    edgenn_arena_t scratch = make_scratch();

    ASSERT_OK(edgenn_lstm_cell_execute(&input, &state, &params, &scratch));

    ASSERT_NEAR(h_data[0], 0.0f, 0.001f);
    ASSERT_NEAR(c_data[0], 0.0f, 0.001f);

    TEST_PASS();
}

static void test_lstm_cell_fp32_known_weights(void)
{
    TEST_CASE("lstm_cell FP32 known weights");

    /* I=1, H=1, batch=1
     * W_ih = [1, 0, 0, 0] (only input gate gets input)
     * W_hh = [0, 0, 0, 0]
     * bias_ih = [0, 0, 1, 0] (cell gate bias=1 -> g=tanh(1))
     * bias_hh = [0, 0, 0, 0]
     * x=2: i=sig(2)=0.8808, f=sig(0)=0.5, g=tanh(1)=0.7616, o=sig(0)=0.5
     * c = 0.5*0 + 0.8808*0.7616 = 0.6710
     * h = 0.5 * tanh(0.6710) = 0.2931
     */
    float x_data[] = {2.0f};
    float h_data[] = {0.0f};
    float c_data[] = {0.0f};
    float w_ih[]   = {1.0f, 0.0f, 0.0f, 0.0f};
    float w_hh[]   = {0.0f, 0.0f, 0.0f, 0.0f};
    float b_ih[]   = {0.0f, 0.0f, 1.0f, 0.0f};

    edgenn_tensor_t input;
    edgenn_tensor_init(&input, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data = x_data;

    edgenn_lstm_state_t state;
    edgenn_tensor_init(&state.h, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&state.c, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    state.h.data = h_data;
    state.c.data = c_data;

    edgenn_lstm_params_t params;
    memset(&params, 0, sizeof(params));
    params.input_size  = 1;
    params.hidden_size = 1;
    edgenn_tensor_init(&params.weight_ih, (int32_t[]){4, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.weight_hh, (int32_t[]){4, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.bias_ih, (int32_t[]){4}, 1,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.weight_ih.data = w_ih;
    params.weight_hh.data = w_hh;
    params.bias_ih.data   = b_ih;

    edgenn_arena_t scratch = make_scratch();

    ASSERT_OK(edgenn_lstm_cell_execute(&input, &state, &params, &scratch));

    /* Compute expected: i=sig(2), f=sig(0), g=tanh(1), o=sig(0) */
    float i_g = 1.0f / (1.0f + expf(-2.0f));   /* sigmoid(2) */
    float f_g = 0.5f;                            /* sigmoid(0) */
    float g_g = tanhf(1.0f);
    float o_g = 0.5f;
    float c_exp = f_g * 0.0f + i_g * g_g;
    float h_exp = o_g * tanhf(c_exp);

    ASSERT_NEAR(c_data[0], c_exp, 0.001f);
    ASSERT_NEAR(h_data[0], h_exp, 0.001f);

    TEST_PASS();
}

static void test_lstm_cell_fp32_state_carry(void)
{
    TEST_CASE("lstm_cell FP32 state carry-over 2 steps");

    float x_data[] = {1.0f};
    float h_data[] = {0.0f};
    float c_data[] = {0.0f};
    /* All weights = 1 for simplicity */
    float w_ih[]   = {1.0f, 1.0f, 1.0f, 1.0f};
    float w_hh[]   = {1.0f, 1.0f, 1.0f, 1.0f};

    edgenn_tensor_t input;
    edgenn_tensor_init(&input, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data = x_data;

    edgenn_lstm_state_t state;
    edgenn_tensor_init(&state.h, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&state.c, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    state.h.data = h_data;
    state.c.data = c_data;

    edgenn_lstm_params_t params;
    memset(&params, 0, sizeof(params));
    params.input_size  = 1;
    params.hidden_size = 1;
    edgenn_tensor_init(&params.weight_ih, (int32_t[]){4, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.weight_hh, (int32_t[]){4, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.weight_ih.data = w_ih;
    params.weight_hh.data = w_hh;

    edgenn_arena_t scratch = make_scratch();

    /* Step 1: x=1, h=0, c=0 -> all gates see value 1 */
    ASSERT_OK(edgenn_lstm_cell_execute(&input, &state, &params, &scratch));
    float h1 = h_data[0];
    float c1 = c_data[0];
    ASSERT_TRUE(h1 != 0.0f);
    ASSERT_TRUE(c1 != 0.0f);

    /* Step 2: x=1, h=h1, c=c1 -> gates see 1+h1 */
    edgenn_arena_reset(&scratch);
    ASSERT_OK(edgenn_lstm_cell_execute(&input, &state, &params, &scratch));
    float h2 = h_data[0];
    float c2 = c_data[0];

    /* Cell state should grow since input/forget gates are active */
    ASSERT_TRUE(fabsf(c2) > fabsf(c1));
    /* h2 should differ from h1 */
    ASSERT_TRUE(fabsf(h2 - h1) > 0.001f);

    TEST_PASS();
}

static void test_lstm_sequence_fp32(void)
{
    TEST_CASE("lstm_sequence FP32 3-step");

    int32_t I = 1, H = 1, batch = 1, T = 3;

    /* Input sequence: [1, 2, 3] */
    float x_seq[] = {1.0f, 2.0f, 3.0f};
    float h_data[] = {0.0f};
    float c_data[] = {0.0f};
    float out_seq[3];

    /* Cell gate (index 2) sees input -> g=tanh(x) is nonzero */
    float w_ih[] = {0.0f, 0.0f, 1.0f, 0.0f};
    float w_hh[] = {0.0f, 0.0f, 0.0f, 0.0f};

    edgenn_tensor_t input_seq;
    edgenn_tensor_init(&input_seq, (int32_t[]){T, batch, I}, 3,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input_seq.data = x_seq;

    edgenn_tensor_t output_seq;
    edgenn_tensor_init(&output_seq, (int32_t[]){T, batch, H}, 3,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    output_seq.data = out_seq;

    edgenn_lstm_state_t state;
    edgenn_tensor_init(&state.h, (int32_t[]){batch, H}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&state.c, (int32_t[]){batch, H}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    state.h.data = h_data;
    state.c.data = c_data;

    edgenn_lstm_params_t params;
    memset(&params, 0, sizeof(params));
    params.input_size  = I;
    params.hidden_size = H;
    edgenn_tensor_init(&params.weight_ih, (int32_t[]){4, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.weight_hh, (int32_t[]){4, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.weight_ih.data = w_ih;
    params.weight_hh.data = w_hh;

    edgenn_arena_t scratch = make_scratch();

    ASSERT_OK(edgenn_lstm_sequence_execute(
        &input_seq, &output_seq, &state, &params, &scratch, T));

    /* Output should be 3 different values, each matching h after that step */
    ASSERT_TRUE(out_seq[0] != 0.0f);
    /* Each time step should be different (different input) */
    ASSERT_TRUE(fabsf(out_seq[0] - out_seq[1]) > 0.001f);
    ASSERT_TRUE(fabsf(out_seq[1] - out_seq[2]) > 0.001f);
    /* Final output should match final state */
    ASSERT_NEAR(out_seq[2], h_data[0], 0.0001f);

    TEST_PASS();
}

static void test_lstm_null_ptrs(void)
{
    TEST_CASE("lstm NULL pointer checks");

    edgenn_tensor_t input;
    edgenn_lstm_state_t state;
    edgenn_lstm_params_t params;
    memset(&params, 0, sizeof(params));
    edgenn_arena_t scratch = make_scratch();

    ASSERT_ERR(edgenn_lstm_cell_execute(NULL, &state, &params, &scratch),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_lstm_cell_execute(&input, NULL, &params, &scratch),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_lstm_cell_execute(&input, &state, NULL, &scratch),
               EDGENN_ERR_NULL_PTR);

    TEST_PASS();
}

static void test_lstm_scratch_size(void)
{
    TEST_CASE("lstm scratch_size");

    edgenn_lstm_params_t params;
    memset(&params, 0, sizeof(params));
    params.hidden_size = 8;

    size_t sz = edgenn_lstm_scratch_size(&params);
    ASSERT_TRUE(sz >= 4 * 8 * sizeof(float));
    ASSERT_TRUE(sz % EDGENN_TENSOR_ALIGN == 0);

    ASSERT_EQ(edgenn_lstm_scratch_size(NULL), 0u);

    TEST_PASS();
}

static void test_lstm_sequence_invalid_seqlen(void)
{
    TEST_CASE("lstm_sequence invalid seq_len");

    edgenn_tensor_t input_seq;
    edgenn_tensor_init(&input_seq, (int32_t[]){1, 1, 1}, 3,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    float dummy = 0.0f;
    input_seq.data = &dummy;

    edgenn_lstm_state_t state;
    float h_data = 0.0f, c_data = 0.0f;
    edgenn_tensor_init(&state.h, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&state.c, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    state.h.data = &h_data;
    state.c.data = &c_data;

    edgenn_lstm_params_t params;
    memset(&params, 0, sizeof(params));
    params.input_size = 1;
    params.hidden_size = 1;
    float w[4] = {0};
    edgenn_tensor_init(&params.weight_ih, (int32_t[]){4, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.weight_hh, (int32_t[]){4, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.weight_ih.data = w;
    params.weight_hh.data = w;

    edgenn_arena_t scratch = make_scratch();

    ASSERT_ERR(edgenn_lstm_sequence_execute(
        &input_seq, NULL, &state, &params, &scratch, 0),
        EDGENN_ERR_INVALID_ARG);
    ASSERT_ERR(edgenn_lstm_sequence_execute(
        &input_seq, NULL, &state, &params, &scratch, -1),
        EDGENN_ERR_INVALID_ARG);

    TEST_PASS();
}

/* ============================================================================
 * GRU Tests
 * ========================================================================= */

static void test_gru_cell_fp32_zero_input(void)
{
    TEST_CASE("gru_cell FP32 zero input");

    /* I=1, H=1, batch=1
     * All weights & biases zero, x=0, h=0
     * r = sig(0) = 0.5, z = sig(0) = 0.5, n = tanh(0) = 0
     * h' = (1-0.5)*0 + 0.5*0 = 0
     */
    float x_data[] = {0.0f};
    float h_data[] = {0.0f};
    float w_ih[3]  = {0};
    float w_hh[3]  = {0};

    edgenn_tensor_t input;
    edgenn_tensor_init(&input, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data = x_data;

    edgenn_gru_state_t state;
    edgenn_tensor_init(&state.h, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    state.h.data = h_data;

    edgenn_gru_params_t params;
    memset(&params, 0, sizeof(params));
    params.input_size  = 1;
    params.hidden_size = 1;
    edgenn_tensor_init(&params.weight_ih, (int32_t[]){3, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.weight_hh, (int32_t[]){3, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.weight_ih.data = w_ih;
    params.weight_hh.data = w_hh;

    edgenn_arena_t scratch = make_scratch();

    ASSERT_OK(edgenn_gru_cell_execute(&input, &state, &params, &scratch));

    ASSERT_NEAR(h_data[0], 0.0f, 0.001f);

    TEST_PASS();
}

static void test_gru_cell_fp32_known_weights(void)
{
    TEST_CASE("gru_cell FP32 known weights");

    /* I=1, H=1, batch=1, h=0
     * W_ih = [w_r=0, w_z=0, w_n=1]  (only new gate sees input)
     * W_hh = [0, 0, 0]
     * No bias
     * x = 2
     *
     * gates_ih = [0*2, 0*2, 1*2] = [0, 0, 2]
     * gates_hh = [0, 0, 0]
     * r = sig(0+0) = 0.5
     * z = sig(0+0) = 0.5
     * n = tanh(2 + 0.5*0) = tanh(2)
     * h' = (1-0.5)*tanh(2) + 0.5*0 = 0.5*tanh(2)
     */
    float x_data[] = {2.0f};
    float h_data[] = {0.0f};
    float w_ih[]   = {0.0f, 0.0f, 1.0f};
    float w_hh[]   = {0.0f, 0.0f, 0.0f};

    edgenn_tensor_t input;
    edgenn_tensor_init(&input, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data = x_data;

    edgenn_gru_state_t state;
    edgenn_tensor_init(&state.h, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    state.h.data = h_data;

    edgenn_gru_params_t params;
    memset(&params, 0, sizeof(params));
    params.input_size  = 1;
    params.hidden_size = 1;
    edgenn_tensor_init(&params.weight_ih, (int32_t[]){3, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.weight_hh, (int32_t[]){3, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.weight_ih.data = w_ih;
    params.weight_hh.data = w_hh;

    edgenn_arena_t scratch = make_scratch();

    ASSERT_OK(edgenn_gru_cell_execute(&input, &state, &params, &scratch));

    float expected = 0.5f * tanhf(2.0f);
    ASSERT_NEAR(h_data[0], expected, 0.001f);

    TEST_PASS();
}

static void test_gru_cell_fp32_state_carry(void)
{
    TEST_CASE("gru_cell FP32 state carry-over");

    float x_data[] = {1.0f};
    float h_data[] = {0.0f};
    float w_ih[]   = {1.0f, 1.0f, 1.0f};
    float w_hh[]   = {1.0f, 1.0f, 1.0f};

    edgenn_tensor_t input;
    edgenn_tensor_init(&input, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data = x_data;

    edgenn_gru_state_t state;
    edgenn_tensor_init(&state.h, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    state.h.data = h_data;

    edgenn_gru_params_t params;
    memset(&params, 0, sizeof(params));
    params.input_size  = 1;
    params.hidden_size = 1;
    edgenn_tensor_init(&params.weight_ih, (int32_t[]){3, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.weight_hh, (int32_t[]){3, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.weight_ih.data = w_ih;
    params.weight_hh.data = w_hh;

    edgenn_arena_t scratch = make_scratch();

    /* Step 1 */
    ASSERT_OK(edgenn_gru_cell_execute(&input, &state, &params, &scratch));
    float h1 = h_data[0];
    ASSERT_TRUE(h1 != 0.0f);

    /* Step 2 — state carries from step 1 */
    edgenn_arena_reset(&scratch);
    ASSERT_OK(edgenn_gru_cell_execute(&input, &state, &params, &scratch));
    float h2 = h_data[0];
    ASSERT_TRUE(fabsf(h2 - h1) > 0.001f);

    TEST_PASS();
}

static void test_gru_sequence_fp32(void)
{
    TEST_CASE("gru_sequence FP32 3-step");

    int32_t I = 1, H = 1, batch = 1, T = 3;

    float x_seq[] = {1.0f, 2.0f, 3.0f};
    float h_data[] = {0.0f};
    float out_seq[3];

    float w_ih[] = {0.0f, 0.0f, 1.0f};
    float w_hh[] = {0.0f, 0.0f, 0.0f};

    edgenn_tensor_t input_seq;
    edgenn_tensor_init(&input_seq, (int32_t[]){T, batch, I}, 3,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input_seq.data = x_seq;

    edgenn_tensor_t output_seq;
    edgenn_tensor_init(&output_seq, (int32_t[]){T, batch, H}, 3,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    output_seq.data = out_seq;

    edgenn_gru_state_t state;
    edgenn_tensor_init(&state.h, (int32_t[]){batch, H}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    state.h.data = h_data;

    edgenn_gru_params_t params;
    memset(&params, 0, sizeof(params));
    params.input_size  = I;
    params.hidden_size = H;
    edgenn_tensor_init(&params.weight_ih, (int32_t[]){3, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.weight_hh, (int32_t[]){3, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.weight_ih.data = w_ih;
    params.weight_hh.data = w_hh;

    edgenn_arena_t scratch = make_scratch();

    ASSERT_OK(edgenn_gru_sequence_execute(
        &input_seq, &output_seq, &state, &params, &scratch, T));

    /* Each step should produce different output */
    ASSERT_TRUE(out_seq[0] != 0.0f);
    ASSERT_TRUE(fabsf(out_seq[0] - out_seq[1]) > 0.001f);
    ASSERT_TRUE(fabsf(out_seq[1] - out_seq[2]) > 0.001f);
    /* Final output = final hidden state */
    ASSERT_NEAR(out_seq[2], h_data[0], 0.0001f);

    TEST_PASS();
}

static void test_gru_null_ptrs(void)
{
    TEST_CASE("gru NULL pointer checks");

    edgenn_tensor_t input;
    edgenn_gru_state_t state;
    edgenn_gru_params_t params;
    memset(&params, 0, sizeof(params));
    edgenn_arena_t scratch = make_scratch();

    ASSERT_ERR(edgenn_gru_cell_execute(NULL, &state, &params, &scratch),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_gru_cell_execute(&input, NULL, &params, &scratch),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_gru_cell_execute(&input, &state, NULL, &scratch),
               EDGENN_ERR_NULL_PTR);

    TEST_PASS();
}

static void test_gru_scratch_size(void)
{
    TEST_CASE("gru scratch_size");

    edgenn_gru_params_t params;
    memset(&params, 0, sizeof(params));
    params.hidden_size = 8;

    size_t sz = edgenn_gru_scratch_size(&params);
    /* Need 6*H floats total (two buffers of 3*H) */
    ASSERT_TRUE(sz >= 6 * 8 * sizeof(float));
    /* Each sub-allocation is aligned */
    ASSERT_TRUE(sz % EDGENN_TENSOR_ALIGN == 0);

    ASSERT_EQ(edgenn_gru_scratch_size(NULL), 0u);

    TEST_PASS();
}

static void test_gru_sequence_invalid_seqlen(void)
{
    TEST_CASE("gru_sequence invalid seq_len");

    edgenn_tensor_t input_seq;
    edgenn_tensor_init(&input_seq, (int32_t[]){1, 1, 1}, 3,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    float dummy = 0.0f;
    input_seq.data = &dummy;

    edgenn_gru_state_t state;
    float h_data = 0.0f;
    edgenn_tensor_init(&state.h, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    state.h.data = &h_data;

    edgenn_gru_params_t params;
    memset(&params, 0, sizeof(params));
    params.input_size = 1;
    params.hidden_size = 1;
    float w[3] = {0};
    edgenn_tensor_init(&params.weight_ih, (int32_t[]){3, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&params.weight_hh, (int32_t[]){3, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    params.weight_ih.data = w;
    params.weight_hh.data = w;

    edgenn_arena_t scratch = make_scratch();

    ASSERT_ERR(edgenn_gru_sequence_execute(
        &input_seq, NULL, &state, &params, &scratch, 0),
        EDGENN_ERR_INVALID_ARG);

    TEST_PASS();
}

/* ============================================================================
 * Main
 * ========================================================================= */

int main(void)
{
    TEST_SUITE_BEGIN("EdgeNN Phase 3 — RNN Operators");

    /* Simple RNN Cell tests */
    test_rnn_cell_fp32_tanh();
    test_rnn_cell_fp32_relu();
    test_rnn_cell_fp32_state_carry();
    test_rnn_cell_fp32_with_bias();
    test_rnn_cell_null_ptrs();
    test_rnn_cell_scratch_size();

    /* LSTM tests */
    test_lstm_cell_fp32_zero_input();
    test_lstm_cell_fp32_known_weights();
    test_lstm_cell_fp32_state_carry();
    test_lstm_sequence_fp32();
    test_lstm_null_ptrs();
    test_lstm_scratch_size();
    test_lstm_sequence_invalid_seqlen();

    /* GRU tests */
    test_gru_cell_fp32_zero_input();
    test_gru_cell_fp32_known_weights();
    test_gru_cell_fp32_state_carry();
    test_gru_sequence_fp32();
    test_gru_null_ptrs();
    test_gru_scratch_size();
    test_gru_sequence_invalid_seqlen();

    TEST_SUITE_END();
}
