/**
 * @file edgenn_model.h
 * @brief Model loading and binary format (.edgenn)
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */
#ifndef EDGENN_MODEL_H
#define EDGENN_MODEL_H
#include "../core/edgenn_types.h"
#include "../core/edgenn_status.h"
#include "../core/edgenn_arena.h"
#include "edgenn_graph.h"
#ifdef __cplusplus
extern "C" {
#endif

#define EDGENN_MODEL_MAGIC   0x454E4E45  /* "ENNE" */
#define EDGENN_MODEL_VERSION 1

/** Model file header (binary format) */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t n_layers;
    uint32_t n_tensors;
    uint32_t weight_data_size;
    uint32_t metadata_size;
    uint32_t flags;
    uint32_t reserved[4];
} edgenn_model_header_t;

/** Loaded model handle */
typedef struct {
    edgenn_model_header_t header;
    edgenn_graph_t        graph;
    const uint8_t        *weight_data;   /**< Pointer to weight blob (Flash or RAM) */
    size_t                total_params;
    size_t                total_memory;
} edgenn_model_t;

edgenn_status_t edgenn_model_load_buffer(edgenn_model_t *model, const uint8_t *buffer, size_t size, edgenn_arena_t *weight_arena, edgenn_arena_t *scratch_arena);
edgenn_status_t edgenn_model_infer(edgenn_model_t *model, const edgenn_tensor_t *input, edgenn_tensor_t *output);
void edgenn_model_print_info(const edgenn_model_t *model);
#ifdef __cplusplus
}
#endif
#endif
