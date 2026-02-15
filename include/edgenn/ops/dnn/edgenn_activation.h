/**
 * @file edgenn_activation.h
 * @brief Activation function operators
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#ifndef EDGENN_ACTIVATION_H
#define EDGENN_ACTIVATION_H
#include "../../core/edgenn_types.h"
#include "../../core/edgenn_status.h"
#include "../../core/edgenn_tensor.h"
#ifdef __cplusplus
extern "C" {
#endif
edgenn_status_t edgenn_activation_execute(const edgenn_tensor_t *input, edgenn_tensor_t *output, edgenn_act_type_t act_type);
edgenn_status_t edgenn_softmax_execute(const edgenn_tensor_t *input, edgenn_tensor_t *output);
#ifdef __cplusplus
}
#endif
#endif
