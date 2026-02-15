/**
 * @file edgenn_pool.h
 * @brief Pooling operators (MaxPool2D, AvgPool2D)
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#ifndef EDGENN_POOL_H
#define EDGENN_POOL_H
#include "../../core/edgenn_types.h"
#include "../../core/edgenn_status.h"
#include "../../core/edgenn_tensor.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int32_t kernel_h, kernel_w;
    int32_t stride_h, stride_w;
    int32_t pad_top, pad_bottom, pad_left, pad_right;
    edgenn_padding_t padding;
} edgenn_pool_params_t;
edgenn_status_t edgenn_maxpool2d_execute(const edgenn_tensor_t *input, edgenn_tensor_t *output, const edgenn_pool_params_t *params);
edgenn_status_t edgenn_avgpool2d_execute(const edgenn_tensor_t *input, edgenn_tensor_t *output, const edgenn_pool_params_t *params);
edgenn_status_t edgenn_global_avgpool_execute(const edgenn_tensor_t *input, edgenn_tensor_t *output);
#ifdef __cplusplus
}
#endif
#endif
