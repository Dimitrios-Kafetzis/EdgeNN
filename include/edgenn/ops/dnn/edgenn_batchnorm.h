/**
 * @file edgenn_batchnorm.h
 * @brief Batch Normalization operator
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#ifndef EDGENN_BATCHNORM_H
#define EDGENN_BATCHNORM_H
#include "../../core/edgenn_types.h"
#include "../../core/edgenn_status.h"
#include "../../core/edgenn_tensor.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    edgenn_tensor_t gamma;
    edgenn_tensor_t beta;
    edgenn_tensor_t running_mean;
    edgenn_tensor_t running_var;
    float           epsilon;
} edgenn_batchnorm_params_t;
edgenn_status_t edgenn_batchnorm_execute(const edgenn_tensor_t *input, edgenn_tensor_t *output, const edgenn_batchnorm_params_t *params);
#ifdef __cplusplus
}
#endif
#endif
