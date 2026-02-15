/**
 * @file edgenn_model.c
 * @brief Model loading from binary .edgenn format
 *
 * Binary format:
 *   [header]               — edgenn_model_header_t (44 bytes)
 *   [layer descriptors]    — n_layers × layer_desc records
 *   [tensor descriptors]   — n_tensors × tensor_desc records
 *   [weight data]          — raw weight blob
 *
 * Layer descriptor record (on-disk):
 *   uint32_t op_type
 *   int32_t  input_idx
 *   int32_t  output_idx
 *   int32_t  aux_idx
 *
 * Tensor descriptor record (on-disk):
 *   uint32_t ndim
 *   int32_t  shape[4]
 *   uint32_t dtype
 *   uint32_t layout
 *   uint32_t data_offset    — offset into weight blob (0xFFFFFFFF = activation)
 *   uint32_t data_size      — size in bytes
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn/runtime/edgenn_model.h"
#include <string.h>
#include <stdio.h>

/* On-disk layer descriptor */
typedef struct {
    uint32_t op_type;
    int32_t  input_idx;
    int32_t  output_idx;
    int32_t  aux_idx;
} layer_desc_record_t;

/* On-disk tensor descriptor */
typedef struct {
    uint32_t ndim;
    int32_t  shape[4];
    uint32_t dtype;
    uint32_t layout;
    uint32_t data_offset;
    uint32_t data_size;
} tensor_desc_record_t;

#define ACTIVATION_MARKER 0xFFFFFFFFu

edgenn_status_t edgenn_model_load_buffer(
    edgenn_model_t *model,
    const uint8_t  *buffer,
    size_t          size,
    edgenn_arena_t *weight_arena,
    edgenn_arena_t *scratch_arena)
{
    EDGENN_CHECK_NULL(model);
    EDGENN_CHECK_NULL(buffer);

    memset(model, 0, sizeof(*model));

    /* 1. Parse and validate header */
    if (size < sizeof(edgenn_model_header_t))
        return EDGENN_ERR_MODEL_INVALID;

    memcpy(&model->header, buffer, sizeof(edgenn_model_header_t));

    if (model->header.magic != EDGENN_MODEL_MAGIC)
        return EDGENN_ERR_MODEL_INVALID;
    if (model->header.version != EDGENN_MODEL_VERSION)
        return EDGENN_ERR_MODEL_VERSION;

    uint32_t n_layers  = model->header.n_layers;
    uint32_t n_tensors = model->header.n_tensors;

    if (n_layers > EDGENN_MAX_LAYERS || n_tensors > EDGENN_MAX_TENSORS)
        return EDGENN_ERR_MODEL_INVALID;

    /* 2. Initialize graph */
    EDGENN_CHECK(edgenn_graph_init(&model->graph, weight_arena, scratch_arena));

    /* 3. Parse layer descriptors */
    size_t offset = sizeof(edgenn_model_header_t);
    size_t layers_size = (size_t)n_layers * sizeof(layer_desc_record_t);
    if (offset + layers_size > size)
        return EDGENN_ERR_MODEL_INVALID;

    for (uint32_t l = 0; l < n_layers; l++) {
        layer_desc_record_t rec;
        memcpy(&rec, buffer + offset, sizeof(rec));
        offset += sizeof(rec);

        edgenn_layer_desc_t layer;
        memset(&layer, 0, sizeof(layer));
        layer.op_type   = (edgenn_op_type_t)rec.op_type;
        layer.input_idx = rec.input_idx;
        layer.output_idx = rec.output_idx;
        layer.aux_idx   = rec.aux_idx;
        layer.params    = NULL; /* Would be set up by a full converter */

        EDGENN_CHECK(edgenn_graph_add_layer(&model->graph, &layer));
    }

    /* 4. Parse tensor descriptors */
    size_t tensors_size = (size_t)n_tensors * sizeof(tensor_desc_record_t);
    if (offset + tensors_size > size)
        return EDGENN_ERR_MODEL_INVALID;

    const uint8_t *weight_blob = buffer + offset + tensors_size;
    size_t weight_blob_offset = offset + tensors_size;

    model->total_params = 0;
    model->total_memory = 0;

    for (uint32_t t = 0; t < n_tensors; t++) {
        tensor_desc_record_t rec;
        memcpy(&rec, buffer + offset, sizeof(rec));
        offset += sizeof(rec);

        edgenn_tensor_t *tensor = &model->graph.tensors[t];
        edgenn_tensor_init(tensor,
                           rec.shape,
                           (uint8_t)rec.ndim,
                           (edgenn_dtype_t)rec.dtype,
                           (edgenn_layout_t)rec.layout);

        if (rec.data_offset != ACTIVATION_MARKER) {
            /* Weight tensor — zero-copy pointer into buffer */
            if (weight_blob_offset + rec.data_offset + rec.data_size > size)
                return EDGENN_ERR_MODEL_INVALID;
            tensor->data = (void *)(weight_blob + rec.data_offset);
            model->total_params += rec.data_size;
        } else {
            /* Activation tensor — allocate from weight arena */
            if (weight_arena) {
                EDGENN_CHECK(edgenn_arena_alloc_tensor(weight_arena, tensor));
            }
            model->total_memory += rec.data_size;
        }
    }

    model->graph.n_tensors = (int32_t)n_tensors;

    /* Set input/output tensor indices (first and last tensor by convention) */
    if (n_tensors > 0) {
        model->graph.input_tensor_idx  = 0;
        model->graph.output_tensor_idx = (int32_t)(n_tensors - 1);
    }

    model->weight_data = weight_blob;

    return EDGENN_OK;
}

edgenn_status_t edgenn_model_infer(
    edgenn_model_t        *model,
    const edgenn_tensor_t *input,
    edgenn_tensor_t       *output)
{
    EDGENN_CHECK_NULL(model);
    return edgenn_graph_execute(&model->graph, input, output);
}

void edgenn_model_print_info(const edgenn_model_t *model)
{
    if (!model) return;

    printf("EdgeNN Model v%u\n", model->header.version);
    printf("  Layers:  %u\n", model->header.n_layers);
    printf("  Tensors: %u\n", model->header.n_tensors);
    printf("  Weight data: %u bytes\n", model->header.weight_data_size);
    printf("  Total params: %lu bytes\n", (unsigned long)model->total_params);
    printf("  Total activation memory: %lu bytes\n", (unsigned long)model->total_memory);

    edgenn_graph_print_summary(&model->graph);
}
