/**
 * @file edgenn_graph.c
 * @brief Static graph executor — dispatches layers sequentially
 *
 * The graph holds an array of layer descriptors. Each layer references
 * tensors by index into graph->tensors[]. The executor copies input
 * data into the graph's input tensor, runs each layer, and copies
 * the final output tensor to the caller's output buffer.
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn/runtime/edgenn_graph.h"
#include "edgenn/edgenn.h"
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * Layer dispatch — calls the appropriate operator execute function
 * ========================================================================= */

static edgenn_status_t dispatch_layer(
    const edgenn_layer_desc_t *layer,
    edgenn_tensor_t           *tensors,
    edgenn_arena_t            *scratch)
{
    edgenn_tensor_t *inp = &tensors[layer->input_idx];
    edgenn_tensor_t *out = &tensors[layer->output_idx];

    switch (layer->op_type) {

    /* --- DNN operators --- */
    case EDGENN_OP_DENSE:
        return edgenn_dense_execute(
            inp, out, (const edgenn_dense_params_t *)layer->params, scratch);

    case EDGENN_OP_CONV2D:
        return edgenn_conv2d_execute(
            inp, out, (const edgenn_conv2d_params_t *)layer->params, scratch);

    case EDGENN_OP_DWCONV2D:
        return edgenn_dwconv2d_execute(
            inp, out, (const edgenn_dwconv2d_params_t *)layer->params, scratch);

    case EDGENN_OP_MAXPOOL2D:
        return edgenn_maxpool2d_execute(
            inp, out, (const edgenn_pool_params_t *)layer->params);

    case EDGENN_OP_AVGPOOL2D:
        return edgenn_avgpool2d_execute(
            inp, out, (const edgenn_pool_params_t *)layer->params);

    case EDGENN_OP_BATCHNORM:
        return edgenn_batchnorm_execute(
            inp, out, (const edgenn_batchnorm_params_t *)layer->params);

    /* --- Activation operators --- */
    case EDGENN_OP_RELU:
        return edgenn_activation_execute(inp, out, EDGENN_ACT_RELU);
    case EDGENN_OP_RELU6:
        return edgenn_activation_execute(inp, out, EDGENN_ACT_RELU6);
    case EDGENN_OP_SIGMOID:
        return edgenn_activation_execute(inp, out, EDGENN_ACT_SIGMOID);
    case EDGENN_OP_TANH:
        return edgenn_activation_execute(inp, out, EDGENN_ACT_TANH);
    case EDGENN_OP_GELU:
        return edgenn_activation_execute(inp, out, EDGENN_ACT_GELU);
    case EDGENN_OP_SOFTMAX:
        return edgenn_softmax_execute(inp, out);

    /* --- Transformer operators --- */
    case EDGENN_OP_LAYERNORM:
        return edgenn_layernorm_execute(
            inp, out, (const edgenn_layernorm_params_t *)layer->params);

    case EDGENN_OP_FFN:
        return edgenn_ffn_execute(
            inp, out, (const edgenn_ffn_params_t *)layer->params, scratch);

    /* --- Element-wise operators --- */
    case EDGENN_OP_ADD: {
        if (layer->aux_idx < 0) return EDGENN_ERR_INVALID_ARG;
        const edgenn_tensor_t *aux = &tensors[layer->aux_idx];
        if (inp->dtype != EDGENN_DTYPE_FP32) return EDGENN_ERR_UNSUPPORTED;
        int32_t total = 1;
        for (int32_t i = 0; i < inp->ndim; i++) total *= inp->shape[i];
        edgenn_fp32_vec_add((const float *)inp->data,
                            (const float *)aux->data,
                            (float *)out->data, total);
        return EDGENN_OK;
    }

    default:
        return EDGENN_ERR_UNSUPPORTED;
    }
}

/* ============================================================================
 * Public API
 * ========================================================================= */

edgenn_status_t edgenn_graph_init(
    edgenn_graph_t *graph,
    edgenn_arena_t *weight_arena,
    edgenn_arena_t *scratch_arena)
{
    EDGENN_CHECK_NULL(graph);
    memset(graph, 0, sizeof(*graph));
    graph->weight_arena  = weight_arena;
    graph->scratch_arena = scratch_arena;
    graph->input_tensor_idx  = -1;
    graph->output_tensor_idx = -1;
    return EDGENN_OK;
}

edgenn_status_t edgenn_graph_add_layer(
    edgenn_graph_t            *graph,
    const edgenn_layer_desc_t *layer)
{
    EDGENN_CHECK_NULL(graph);
    EDGENN_CHECK_NULL(layer);
    if (graph->n_layers >= EDGENN_MAX_LAYERS) return EDGENN_ERR_LAYER_LIMIT;
    graph->layers[graph->n_layers++] = *layer;
    return EDGENN_OK;
}

edgenn_status_t edgenn_graph_execute(
    edgenn_graph_t       *graph,
    const edgenn_tensor_t *input,
    edgenn_tensor_t       *output)
{
    EDGENN_CHECK_NULL(graph);
    EDGENN_CHECK_NULL(input);
    EDGENN_CHECK_NULL(output);
    EDGENN_CHECK_NULL(input->data);
    EDGENN_CHECK_NULL(output->data);

    if (graph->n_layers <= 0) return EDGENN_ERR_INVALID_ARG;
    if (graph->input_tensor_idx < 0 || graph->output_tensor_idx < 0)
        return EDGENN_ERR_NOT_INITIALIZED;

    /* Copy caller's input data into the graph's input tensor */
    edgenn_tensor_t *graph_in = &graph->tensors[graph->input_tensor_idx];
    int32_t in_elems = 1;
    for (int32_t i = 0; i < graph_in->ndim; i++)
        in_elems *= graph_in->shape[i];

    size_t in_bytes;
    if (graph_in->dtype == EDGENN_DTYPE_FP32)
        in_bytes = (size_t)in_elems * sizeof(float);
    else
        in_bytes = (size_t)in_elems * sizeof(int8_t);

    memcpy(graph_in->data, input->data, in_bytes);

    /* Execute each layer sequentially */
    for (int32_t l = 0; l < graph->n_layers; l++) {
        size_t saved = 0;
        if (graph->scratch_arena)
            saved = edgenn_arena_save(graph->scratch_arena);

        edgenn_status_t s = dispatch_layer(
            &graph->layers[l], graph->tensors, graph->scratch_arena);
        if (s != EDGENN_OK) return s;

        if (graph->scratch_arena)
            edgenn_arena_restore(graph->scratch_arena, saved);
    }

    /* Copy graph's output tensor to caller's output */
    edgenn_tensor_t *graph_out = &graph->tensors[graph->output_tensor_idx];
    int32_t out_elems = 1;
    for (int32_t i = 0; i < graph_out->ndim; i++)
        out_elems *= graph_out->shape[i];

    size_t out_bytes;
    if (graph_out->dtype == EDGENN_DTYPE_FP32)
        out_bytes = (size_t)out_elems * sizeof(float);
    else
        out_bytes = (size_t)out_elems * sizeof(int8_t);

    memcpy(output->data, graph_out->data, out_bytes);

    return EDGENN_OK;
}

void edgenn_graph_print_summary(const edgenn_graph_t *graph)
{
    if (!graph) return;

    printf("EdgeNN Graph: %d layers, %d tensors\n",
           graph->n_layers, graph->n_tensors);
    printf("  Input tensor:  %d\n", graph->input_tensor_idx);
    printf("  Output tensor: %d\n", graph->output_tensor_idx);

    for (int32_t l = 0; l < graph->n_layers; l++) {
        const edgenn_layer_desc_t *layer = &graph->layers[l];
        printf("  Layer %d: op=%d, in=%d, out=%d",
               l, (int)layer->op_type, layer->input_idx, layer->output_idx);
        if (layer->aux_idx >= 0)
            printf(", aux=%d", layer->aux_idx);
        if (layer->cycles > 0)
            printf(", cycles=%u", layer->cycles);
        printf("\n");
    }
}
