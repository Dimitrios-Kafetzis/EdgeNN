/**
 * @file edgenn_graph.h
 * @brief Static graph executor for layer-by-layer inference
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */
#ifndef EDGENN_GRAPH_H
#define EDGENN_GRAPH_H
#include "../core/edgenn_types.h"
#include "../core/edgenn_status.h"
#include "../core/edgenn_tensor.h"
#include "../core/edgenn_arena.h"
#ifdef __cplusplus
extern "C" {
#endif

/** Layer descriptor in the execution graph */
typedef struct {
    edgenn_op_type_t  op_type;
    void             *params;       /**< Pointer to op-specific params struct  */
    int32_t           input_idx;    /**< Tensor index for input                */
    int32_t           output_idx;   /**< Tensor index for output               */
    int32_t           aux_idx;      /**< Secondary input (e.g., residual add)  */
    uint32_t          cycles;       /**< Profiling: last execution cycle count */
} edgenn_layer_desc_t;

/** Graph execution context */
typedef struct {
    edgenn_layer_desc_t  layers[EDGENN_MAX_LAYERS];
    edgenn_tensor_t      tensors[EDGENN_MAX_TENSORS];
    int32_t              n_layers;
    int32_t              n_tensors;
    int32_t              input_tensor_idx;
    int32_t              output_tensor_idx;
    edgenn_arena_t      *weight_arena;
    edgenn_arena_t      *scratch_arena;
} edgenn_graph_t;

edgenn_status_t edgenn_graph_init(edgenn_graph_t *graph, edgenn_arena_t *weight_arena, edgenn_arena_t *scratch_arena);
edgenn_status_t edgenn_graph_add_layer(edgenn_graph_t *graph, const edgenn_layer_desc_t *layer);
edgenn_status_t edgenn_graph_execute(edgenn_graph_t *graph, const edgenn_tensor_t *input, edgenn_tensor_t *output);
void edgenn_graph_print_summary(const edgenn_graph_t *graph);
#ifdef __cplusplus
}
#endif
#endif
