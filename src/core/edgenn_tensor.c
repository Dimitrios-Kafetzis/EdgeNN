/**
 * @file edgenn_tensor.c
 * @brief Tensor descriptor operations implementation
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn/core/edgenn_tensor.h"
#include "edgenn/core/edgenn_log.h"
#include <string.h>

edgenn_status_t edgenn_tensor_init(
    edgenn_tensor_t  *tensor,
    const int32_t    *shape,
    uint8_t           ndim,
    edgenn_dtype_t    dtype,
    edgenn_layout_t   layout)
{
    EDGENN_CHECK_NULL(tensor);
    EDGENN_CHECK_NULL(shape);

    if (ndim == 0 || ndim > EDGENN_MAX_DIMS) {
        return EDGENN_ERR_INVALID_ARG;
    }

    memset(tensor, 0, sizeof(edgenn_tensor_t));
    tensor->ndim   = ndim;
    tensor->dtype  = dtype;
    tensor->layout = layout;
    tensor->data   = NULL;

    /* Copy shape and compute row-major strides */
    for (uint8_t i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            return EDGENN_ERR_INVALID_ARG;
        }
        tensor->shape[i] = shape[i];
    }

    /* Row-major strides: stride[last] = 1, stride[i] = stride[i+1] * shape[i+1] */
    tensor->strides[ndim - 1] = 1;
    for (int i = (int)ndim - 2; i >= 0; i--) {
        tensor->strides[i] = tensor->strides[i + 1] * tensor->shape[i + 1];
    }

    /* Default quantization: none */
    tensor->qparams.scheme = EDGENN_QSCHEME_NONE;
    tensor->qparams.scale  = 1.0f;
    tensor->qparams.zero_point = 0;

    return EDGENN_OK;
}

int32_t edgenn_tensor_numel(const edgenn_tensor_t *tensor)
{
    if (!tensor || tensor->ndim == 0) return 0;

    int32_t total = 1;
    for (uint8_t i = 0; i < tensor->ndim; i++) {
        total *= tensor->shape[i];
    }
    return total;
}

size_t edgenn_tensor_byte_size(const edgenn_tensor_t *tensor)
{
    if (!tensor) return 0;
    int32_t n = edgenn_tensor_numel(tensor);
    if (n <= 0) return 0;

    if (tensor->dtype == EDGENN_DTYPE_INT4) {
        /* INT4: 2 values packed per byte */
        return (size_t)((n + 1) / 2);
    }
    return (size_t)n * edgenn_dtype_size(tensor->dtype);
}

size_t edgenn_tensor_aligned_size(const edgenn_tensor_t *tensor)
{
    size_t raw = edgenn_tensor_byte_size(tensor);
    return EDGENN_ALIGN_UP(raw, EDGENN_TENSOR_ALIGN);
}

edgenn_status_t edgenn_tensor_set_qparams(
    edgenn_tensor_t        *tensor,
    const edgenn_qparams_t *qparams)
{
    EDGENN_CHECK_NULL(tensor);
    EDGENN_CHECK_NULL(qparams);
    tensor->qparams = *qparams;
    return EDGENN_OK;
}

bool edgenn_tensor_shapes_equal(
    const edgenn_tensor_t *a,
    const edgenn_tensor_t *b)
{
    if (!a || !b) return false;
    if (a->ndim != b->ndim) return false;
    for (uint8_t i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return false;
    }
    return true;
}

void edgenn_tensor_print_info(const edgenn_tensor_t *tensor, const char *name)
{
    (void)tensor;
    (void)name;
#ifdef EDGENN_LOGGING
    if (!tensor) {
        EDGENN_LOG_I("Tensor '%s': NULL", name ? name : "?");
        return;
    }

    static const char *dtype_names[] = {
        "INT8", "UINT8", "INT16", "INT32", "FP32", "INT4"
    };

    const char *dname = (tensor->dtype < EDGENN_DTYPE_COUNT)
        ? dtype_names[tensor->dtype] : "UNKNOWN";

    EDGENN_LOG_I("Tensor '%s': dtype=%s ndim=%u shape=[%d",
        name ? name : "?", dname, tensor->ndim,
        tensor->ndim > 0 ? tensor->shape[0] : 0);

    for (uint8_t i = 1; i < tensor->ndim; i++) {
        EDGENN_LOG_I(", %d", tensor->shape[i]);
    }
    EDGENN_LOG_I("] bytes=%zu data=%p",
        edgenn_tensor_byte_size(tensor), tensor->data);
#endif
}

edgenn_status_t edgenn_tensor_slice_dim0(
    const edgenn_tensor_t *src,
    edgenn_tensor_t       *out,
    int32_t                start,
    int32_t                count)
{
    EDGENN_CHECK_NULL(src);
    EDGENN_CHECK_NULL(out);
    EDGENN_CHECK_NULL(src->data);

    if (start < 0 || count <= 0 || (start + count) > src->shape[0]) {
        return EDGENN_ERR_INVALID_ARG;
    }

    *out = *src;
    out->shape[0] = count;

    /* Advance data pointer by start * stride[0] * element_size */
    size_t elem_sz = edgenn_dtype_size(src->dtype);
    size_t offset  = (size_t)start * (size_t)src->strides[0] * elem_sz;
    out->data = (uint8_t *)src->data + offset;

    return EDGENN_OK;
}
