/**
 * @file test_runtime.c
 * @brief Unit tests for Phase 5 Runtime (Graph executor, Model loader)
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn_test.h"
#include "edgenn/edgenn.h"
#include <string.h>

/* Shared arena buffers */
static uint8_t weight_buf[8192] __attribute__((aligned(16)));
static uint8_t scratch_buf[8192] __attribute__((aligned(16)));

/* ============================================================================
 * Helper: build a programmatic 2-layer Dense graph
 *
 *  Input [1×2] → Dense1 [1×3] → Dense2 [1×2] → Output
 *
 * Tensor layout:
 *   t0: input  [1×2] (activation)
 *   t1: mid    [1×3] (activation)
 *   t2: output [1×2] (activation)
 * ========================================================================= */

static edgenn_dense_params_t dense1_params;
static edgenn_dense_params_t dense2_params;

/* Dense1: [3×2] identity-like + sums row */
static float w1_data[] = {
    1, 0,   /* w1[0] = picks input[0] */
    0, 1,   /* w1[1] = picks input[1] */
    1, 1,   /* w1[2] = sum of inputs  */
};

/* Dense2: [2×3] picks first two outputs */
static float w2_data[] = {
    1, 0, 0,   /* w2[0] = picks mid[0] */
    0, 1, 0,   /* w2[1] = picks mid[1] */
};

static void setup_two_layer_graph(
    edgenn_graph_t *graph,
    edgenn_arena_t *warena,
    edgenn_arena_t *sarena,
    float *t0_data, float *t1_data, float *t2_data)
{
    edgenn_graph_init(graph, warena, sarena);

    /* Set up tensors */
    edgenn_tensor_init(&graph->tensors[0], (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    graph->tensors[0].data = t0_data;

    edgenn_tensor_init(&graph->tensors[1], (int32_t[]){1, 3}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    graph->tensors[1].data = t1_data;

    edgenn_tensor_init(&graph->tensors[2], (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    graph->tensors[2].data = t2_data;

    graph->n_tensors = 3;
    graph->input_tensor_idx  = 0;
    graph->output_tensor_idx = 2;

    /* Dense1 params: [1×2] → [1×3] */
    memset(&dense1_params, 0, sizeof(dense1_params));
    dense1_params.in_features  = 2;
    dense1_params.out_features = 3;
    dense1_params.fused_act    = EDGENN_ACT_NONE;
    edgenn_tensor_init(&dense1_params.weights, (int32_t[]){3, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    dense1_params.weights.data = w1_data;

    /* Dense2 params: [1×3] → [1×2] */
    memset(&dense2_params, 0, sizeof(dense2_params));
    dense2_params.in_features  = 3;
    dense2_params.out_features = 2;
    dense2_params.fused_act    = EDGENN_ACT_NONE;
    edgenn_tensor_init(&dense2_params.weights, (int32_t[]){2, 3}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    dense2_params.weights.data = w2_data;

    /* Add layers */
    edgenn_layer_desc_t layer1 = {
        .op_type    = EDGENN_OP_DENSE,
        .params     = &dense1_params,
        .input_idx  = 0,
        .output_idx = 1,
        .aux_idx    = -1,
    };
    edgenn_graph_add_layer(graph, &layer1);

    edgenn_layer_desc_t layer2 = {
        .op_type    = EDGENN_OP_DENSE,
        .params     = &dense2_params,
        .input_idx  = 1,
        .output_idx = 2,
        .aux_idx    = -1,
    };
    edgenn_graph_add_layer(graph, &layer2);
}

/* ============================================================================
 * Graph Tests
 * ========================================================================= */

static void test_graph_init(void)
{
    TEST_CASE("graph init");

    edgenn_arena_t warena, sarena;
    edgenn_arena_init(&warena, weight_buf, sizeof(weight_buf));
    edgenn_arena_init(&sarena, scratch_buf, sizeof(scratch_buf));

    edgenn_graph_t graph;
    ASSERT_OK(edgenn_graph_init(&graph, &warena, &sarena));
    ASSERT_EQ(graph.n_layers, 0);
    ASSERT_EQ(graph.n_tensors, 0);
    ASSERT_EQ(graph.input_tensor_idx, -1);
    ASSERT_EQ(graph.output_tensor_idx, -1);

    TEST_PASS();
}

static void test_graph_add_layer(void)
{
    TEST_CASE("graph add_layer");

    edgenn_graph_t graph;
    edgenn_graph_init(&graph, NULL, NULL);

    edgenn_layer_desc_t layer = {
        .op_type = EDGENN_OP_DENSE,
        .input_idx = 0,
        .output_idx = 1,
        .aux_idx = -1,
    };
    ASSERT_OK(edgenn_graph_add_layer(&graph, &layer));
    ASSERT_EQ(graph.n_layers, 1);
    ASSERT_EQ(graph.layers[0].op_type, EDGENN_OP_DENSE);

    TEST_PASS();
}

static void test_graph_execute_two_dense(void)
{
    TEST_CASE("graph execute 2-layer Dense");

    edgenn_arena_t warena, sarena;
    edgenn_arena_init(&warena, weight_buf, sizeof(weight_buf));
    edgenn_arena_init(&sarena, scratch_buf, sizeof(scratch_buf));

    float t0[2], t1[3], t2[2];
    edgenn_graph_t graph;
    setup_two_layer_graph(&graph, &warena, &sarena, t0, t1, t2);

    /* Input: [3, 4] */
    float input_data[] = {3.0f, 4.0f};
    float output_data[2] = {0};

    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = input_data;
    output.data = output_data;

    ASSERT_OK(edgenn_graph_execute(&graph, &input, &output));

    /* Dense1: [3,4] × w1^T = [3, 4, 7]
     * Dense2: [3,4,7] × w2^T = [3, 4]
     * (w2 picks first two from mid) */
    ASSERT_NEAR(output_data[0], 3.0f, 0.001f);
    ASSERT_NEAR(output_data[1], 4.0f, 0.001f);

    TEST_PASS();
}

static void test_graph_execute_vs_manual(void)
{
    TEST_CASE("graph execute matches manual execution");

    edgenn_arena_t warena, sarena;
    edgenn_arena_init(&warena, weight_buf, sizeof(weight_buf));
    edgenn_arena_init(&sarena, scratch_buf, sizeof(scratch_buf));

    float t0[2], t1[3], t2[2];
    edgenn_graph_t graph;
    setup_two_layer_graph(&graph, &warena, &sarena, t0, t1, t2);

    float x[] = {1.0f, 2.0f};

    /* Manual: Dense1 */
    float mid[3];
    edgenn_tensor_t in_m, out_m;
    edgenn_tensor_init(&in_m, (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&out_m, (int32_t[]){1, 3}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    in_m.data  = x;
    out_m.data = mid;
    edgenn_dense_execute(&in_m, &out_m, &dense1_params, NULL);

    /* Manual: Dense2 */
    float manual_out[2];
    edgenn_tensor_t in_m2, out_m2;
    edgenn_tensor_init(&in_m2, (int32_t[]){1, 3}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&out_m2, (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    in_m2.data  = mid;
    out_m2.data = manual_out;
    edgenn_dense_execute(&in_m2, &out_m2, &dense2_params, NULL);

    /* Graph execution */
    float graph_out[2];
    edgenn_tensor_t gin, gout;
    edgenn_tensor_init(&gin,  (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&gout, (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    gin.data  = x;
    gout.data = graph_out;
    edgenn_graph_execute(&graph, &gin, &gout);

    ASSERT_NEAR(graph_out[0], manual_out[0], 0.0001f);
    ASSERT_NEAR(graph_out[1], manual_out[1], 0.0001f);

    TEST_PASS();
}

static void test_graph_dense_relu_chain(void)
{
    TEST_CASE("graph Dense → ReLU chain");

    edgenn_arena_t sarena;
    edgenn_arena_init(&sarena, scratch_buf, sizeof(scratch_buf));

    edgenn_graph_t graph;
    edgenn_graph_init(&graph, NULL, &sarena);

    /* Tensors: t0=input[1×2], t1=dense_out[1×2], t2=relu_out[1×2] */
    float t0[2], t1[2], t2[2];
    edgenn_tensor_init(&graph.tensors[0], (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&graph.tensors[1], (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&graph.tensors[2], (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    graph.tensors[0].data = t0;
    graph.tensors[1].data = t1;
    graph.tensors[2].data = t2;
    graph.n_tensors = 3;
    graph.input_tensor_idx  = 0;
    graph.output_tensor_idx = 2;

    /* Dense: [-1, 0, 0, 1] maps [a,b] -> [-a, b] */
    static float w_neg[] = {-1, 0, 0, 1};
    static edgenn_dense_params_t dp;
    memset(&dp, 0, sizeof(dp));
    dp.in_features  = 2;
    dp.out_features = 2;
    dp.fused_act    = EDGENN_ACT_NONE;
    edgenn_tensor_init(&dp.weights, (int32_t[]){2, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    dp.weights.data = w_neg;

    edgenn_layer_desc_t l1 = {EDGENN_OP_DENSE, &dp, 0, 1, -1, 0};
    edgenn_layer_desc_t l2 = {EDGENN_OP_RELU, NULL, 1, 2, -1, 0};
    edgenn_graph_add_layer(&graph, &l1);
    edgenn_graph_add_layer(&graph, &l2);

    float x[] = {5.0f, -3.0f};
    float out[2];
    edgenn_tensor_t gin, gout;
    edgenn_tensor_init(&gin,  (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&gout, (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    gin.data  = x;
    gout.data = out;

    ASSERT_OK(edgenn_graph_execute(&graph, &gin, &gout));

    /* Dense: [5,-3] -> [-5, -3], then ReLU -> [0, 0] */
    ASSERT_NEAR(out[0], 0.0f, 0.001f);
    ASSERT_NEAR(out[1], 0.0f, 0.001f);

    TEST_PASS();
}

static void test_graph_add_operator(void)
{
    TEST_CASE("graph ADD element-wise operator");

    edgenn_graph_t graph;
    edgenn_graph_init(&graph, NULL, NULL);

    /* t0=input[1×2], t1=residual[1×2], t2=output[1×2] */
    float t0[] = {1, 2};
    float t1[] = {10, 20};
    float t2[2];
    edgenn_tensor_init(&graph.tensors[0], (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&graph.tensors[1], (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&graph.tensors[2], (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    graph.tensors[0].data = t0;
    graph.tensors[1].data = t1;
    graph.tensors[2].data = t2;
    graph.n_tensors = 3;
    graph.input_tensor_idx  = 0;
    graph.output_tensor_idx = 2;

    /* ADD: t2 = t0 + t1 (aux_idx = 1) */
    edgenn_layer_desc_t add_layer = {
        .op_type = EDGENN_OP_ADD,
        .input_idx = 0,
        .output_idx = 2,
        .aux_idx = 1,
    };
    edgenn_graph_add_layer(&graph, &add_layer);

    float x[] = {1, 2};
    float out[2];
    edgenn_tensor_t gin, gout;
    edgenn_tensor_init(&gin,  (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&gout, (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    gin.data  = x;
    gout.data = out;

    ASSERT_OK(edgenn_graph_execute(&graph, &gin, &gout));

    /* t0 gets [1,2], t1 is [10,20], ADD -> [11, 22] */
    ASSERT_NEAR(out[0], 11.0f, 0.001f);
    ASSERT_NEAR(out[1], 22.0f, 0.001f);

    TEST_PASS();
}

static void test_graph_execute_null_ptrs(void)
{
    TEST_CASE("graph execute NULL pointer checks");

    edgenn_graph_t graph;
    edgenn_graph_init(&graph, NULL, NULL);

    edgenn_tensor_t input, output;

    ASSERT_ERR(edgenn_graph_execute(NULL, &input, &output),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_graph_execute(&graph, NULL, &output),
               EDGENN_ERR_NULL_PTR);

    TEST_PASS();
}

static void test_graph_execute_not_initialized(void)
{
    TEST_CASE("graph execute not initialized");

    edgenn_graph_t graph;
    edgenn_graph_init(&graph, NULL, NULL);

    float x[] = {1}; float o[] = {0};
    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 1}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = x;
    output.data = o;

    /* No layers -> ERR_INVALID_ARG */
    ASSERT_ERR(edgenn_graph_execute(&graph, &input, &output),
               EDGENN_ERR_INVALID_ARG);

    /* Add a layer but don't set input/output tensor indices */
    edgenn_layer_desc_t layer = {EDGENN_OP_RELU, NULL, 0, 0, -1, 0};
    edgenn_graph_add_layer(&graph, &layer);

    ASSERT_ERR(edgenn_graph_execute(&graph, &input, &output),
               EDGENN_ERR_NOT_INITIALIZED);

    TEST_PASS();
}

static void test_graph_add_layer_null(void)
{
    TEST_CASE("graph add_layer NULL checks");

    edgenn_graph_t graph;
    edgenn_layer_desc_t layer = {0};

    ASSERT_ERR(edgenn_graph_add_layer(NULL, &layer),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_graph_add_layer(&graph, NULL),
               EDGENN_ERR_NULL_PTR);

    TEST_PASS();
}

/* ============================================================================
 * Model Loading Tests
 * ========================================================================= */

/* Helper: build a minimal valid model binary in memory */
static size_t build_test_model_binary(uint8_t *buf, size_t buf_size)
{
    (void)buf_size;
    size_t offset = 0;

    /* Header */
    edgenn_model_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic   = EDGENN_MODEL_MAGIC;
    hdr.version = EDGENN_MODEL_VERSION;
    hdr.n_layers  = 1;
    hdr.n_tensors = 2;
    hdr.weight_data_size = 0;
    memcpy(buf + offset, &hdr, sizeof(hdr));
    offset += sizeof(hdr);

    /* Layer descriptor: RELU, input=0, output=1 */
    uint32_t op = EDGENN_OP_RELU;
    int32_t in_idx = 0, out_idx = 1, aux_idx = -1;
    memcpy(buf + offset, &op, 4); offset += 4;
    memcpy(buf + offset, &in_idx, 4); offset += 4;
    memcpy(buf + offset, &out_idx, 4); offset += 4;
    memcpy(buf + offset, &aux_idx, 4); offset += 4;

    /* Tensor 0: input [1×2] FP32, activation */
    uint32_t ndim0 = 2;
    int32_t shape0[4] = {1, 2, 0, 0};
    uint32_t dtype0 = EDGENN_DTYPE_FP32;
    uint32_t layout0 = EDGENN_LAYOUT_NC;
    uint32_t data_offset0 = 0xFFFFFFFF; /* activation */
    uint32_t data_size0 = 2 * sizeof(float);
    memcpy(buf + offset, &ndim0, 4); offset += 4;
    memcpy(buf + offset, shape0, 16); offset += 16;
    memcpy(buf + offset, &dtype0, 4); offset += 4;
    memcpy(buf + offset, &layout0, 4); offset += 4;
    memcpy(buf + offset, &data_offset0, 4); offset += 4;
    memcpy(buf + offset, &data_size0, 4); offset += 4;

    /* Tensor 1: output [1×2] FP32, activation */
    memcpy(buf + offset, &ndim0, 4); offset += 4;
    memcpy(buf + offset, shape0, 16); offset += 16;
    memcpy(buf + offset, &dtype0, 4); offset += 4;
    memcpy(buf + offset, &layout0, 4); offset += 4;
    memcpy(buf + offset, &data_offset0, 4); offset += 4;
    memcpy(buf + offset, &data_size0, 4); offset += 4;

    return offset;
}

static void test_model_load_valid(void)
{
    TEST_CASE("model load valid binary");

    uint8_t model_buf[512];
    size_t sz = build_test_model_binary(model_buf, sizeof(model_buf));

    edgenn_arena_t warena;
    edgenn_arena_init(&warena, weight_buf, sizeof(weight_buf));
    edgenn_arena_t sarena;
    edgenn_arena_init(&sarena, scratch_buf, sizeof(scratch_buf));

    edgenn_model_t model;
    ASSERT_OK(edgenn_model_load_buffer(&model, model_buf, sz,
                                        &warena, &sarena));

    ASSERT_EQ(model.header.magic, EDGENN_MODEL_MAGIC);
    ASSERT_EQ(model.header.version, EDGENN_MODEL_VERSION);
    ASSERT_EQ(model.graph.n_layers, 1);
    ASSERT_EQ(model.graph.n_tensors, 2);
    ASSERT_EQ(model.graph.input_tensor_idx, 0);
    ASSERT_EQ(model.graph.output_tensor_idx, 1);

    TEST_PASS();
}

static void test_model_load_bad_magic(void)
{
    TEST_CASE("model load bad magic");

    uint8_t model_buf[512];
    size_t sz = build_test_model_binary(model_buf, sizeof(model_buf));

    /* Corrupt magic */
    model_buf[0] = 0xFF;

    edgenn_model_t model;
    ASSERT_ERR(edgenn_model_load_buffer(&model, model_buf, sz, NULL, NULL),
               EDGENN_ERR_MODEL_INVALID);

    TEST_PASS();
}

static void test_model_load_bad_version(void)
{
    TEST_CASE("model load bad version");

    uint8_t model_buf[512];
    size_t sz = build_test_model_binary(model_buf, sizeof(model_buf));

    /* Corrupt version (byte 4-7) */
    uint32_t bad_ver = 99;
    memcpy(model_buf + 4, &bad_ver, 4);

    edgenn_model_t model;
    ASSERT_ERR(edgenn_model_load_buffer(&model, model_buf, sz, NULL, NULL),
               EDGENN_ERR_MODEL_VERSION);

    TEST_PASS();
}

static void test_model_load_truncated(void)
{
    TEST_CASE("model load truncated buffer");

    uint8_t model_buf[512];
    build_test_model_binary(model_buf, sizeof(model_buf));

    edgenn_model_t model;
    /* Pass only partial header */
    ASSERT_ERR(edgenn_model_load_buffer(&model, model_buf, 8, NULL, NULL),
               EDGENN_ERR_MODEL_INVALID);

    TEST_PASS();
}

static void test_model_load_null_ptrs(void)
{
    TEST_CASE("model load NULL pointer checks");

    uint8_t buf[64];
    edgenn_model_t model;

    ASSERT_ERR(edgenn_model_load_buffer(NULL, buf, 64, NULL, NULL),
               EDGENN_ERR_NULL_PTR);
    ASSERT_ERR(edgenn_model_load_buffer(&model, NULL, 64, NULL, NULL),
               EDGENN_ERR_NULL_PTR);

    TEST_PASS();
}

static void test_model_infer_null(void)
{
    TEST_CASE("model infer NULL check");

    edgenn_tensor_t input, output;
    ASSERT_ERR(edgenn_model_infer(NULL, &input, &output),
               EDGENN_ERR_NULL_PTR);

    TEST_PASS();
}

static void test_model_infer_relu(void)
{
    TEST_CASE("model infer with ReLU layer");

    uint8_t model_buf[512];
    size_t sz = build_test_model_binary(model_buf, sizeof(model_buf));

    edgenn_arena_t warena, sarena;
    edgenn_arena_init(&warena, weight_buf, sizeof(weight_buf));
    edgenn_arena_init(&sarena, scratch_buf, sizeof(scratch_buf));

    edgenn_model_t model;
    ASSERT_OK(edgenn_model_load_buffer(&model, model_buf, sz,
                                        &warena, &sarena));

    /* Input: [-5, 3] */
    float x[] = {-5.0f, 3.0f};
    float out[2];
    edgenn_tensor_t input, output;
    edgenn_tensor_init(&input,  (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, (int32_t[]){1, 2}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    input.data  = x;
    output.data = out;

    ASSERT_OK(edgenn_model_infer(&model, &input, &output));

    /* ReLU: [-5, 3] -> [0, 3] */
    ASSERT_NEAR(out[0], 0.0f, 0.001f);
    ASSERT_NEAR(out[1], 3.0f, 0.001f);

    TEST_PASS();
}

/* ============================================================================
 * Main
 * ========================================================================= */

int main(void)
{
    TEST_SUITE_BEGIN("EdgeNN Phase 5 — Runtime");

    /* Graph tests */
    test_graph_init();
    test_graph_add_layer();
    test_graph_execute_two_dense();
    test_graph_execute_vs_manual();
    test_graph_dense_relu_chain();
    test_graph_add_operator();
    test_graph_execute_null_ptrs();
    test_graph_execute_not_initialized();
    test_graph_add_layer_null();

    /* Model loading tests */
    test_model_load_valid();
    test_model_load_bad_magic();
    test_model_load_bad_version();
    test_model_load_truncated();
    test_model_load_null_ptrs();
    test_model_infer_null();
    test_model_infer_relu();

    TEST_SUITE_END();
}
