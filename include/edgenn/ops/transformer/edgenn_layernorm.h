/**
 * @file edgenn_layernorm.h
 * @brief Layer Normalization operator
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#ifndef EDGENN_LAYERNORM_H
#define EDGENN_LAYERNORM_H
#include "../../core/edgenn_types.h"
#include "../../core/edgenn_status.h"
#include "../../core/edgenn_tensor.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    edgenn_tensor_t gamma;
    edgenn_tensor_t beta;
    int32_t         normalized_shape;
    float           epsilon;
} edgenn_layernorm_params_t;
edgenn_status_t edgenn_layernorm_execute(const edgenn_tensor_t *input, edgenn_tensor_t *output, const edgenn_layernorm_params_t *params);
#ifdef __cplusplus
}
#endif
#endif
