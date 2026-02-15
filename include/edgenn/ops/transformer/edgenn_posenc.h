/**
 * @file edgenn_posenc.h
 * @brief Positional Encoding operator
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#ifndef EDGENN_POSENC_H
#define EDGENN_POSENC_H
#include "../../core/edgenn_types.h"
#include "../../core/edgenn_status.h"
#include "../../core/edgenn_tensor.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
    EDGENN_POSENC_SINUSOIDAL = 0,
    EDGENN_POSENC_LEARNED    = 1,
    EDGENN_POSENC_ROPE       = 2,
} edgenn_posenc_type_t;
typedef struct {
    edgenn_posenc_type_t type;
    edgenn_tensor_t      encoding_table;   /**< Precomputed [max_seq × d_model] */
    int32_t              max_seq_len;
    int32_t              d_model;
} edgenn_posenc_params_t;
edgenn_status_t edgenn_posenc_apply(const edgenn_tensor_t *input, edgenn_tensor_t *output, const edgenn_posenc_params_t *params, int32_t position_offset);
edgenn_status_t edgenn_posenc_generate_sinusoidal(float *table, int32_t max_seq_len, int32_t d_model);
#ifdef __cplusplus
}
#endif
#endif
