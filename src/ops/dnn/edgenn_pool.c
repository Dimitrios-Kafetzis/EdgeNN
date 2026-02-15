/**
 * @file edgenn_pool.c
 * @brief Pooling operators — MaxPool2D, AvgPool2D, GlobalAvgPool
 *
 * Input layout: NHWC [batch x height x width x channels]
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn/ops/dnn/edgenn_pool.h"
#include "edgenn/core/edgenn_math_q.h"
#include <float.h>

/* ============================================================================
 * Padding Computation
 * ========================================================================= */

static void compute_same_padding(
    int32_t in_h, int32_t in_w,
    int32_t kernel_h, int32_t kernel_w,
    int32_t stride_h, int32_t stride_w,
    int32_t *pad_top, int32_t *pad_bottom,
    int32_t *pad_left, int32_t *pad_right)
{
    int32_t out_h = (in_h + stride_h - 1) / stride_h;
    int32_t out_w = (in_w + stride_w - 1) / stride_w;
    int32_t pad_h = EDGENN_MAX(0, (out_h - 1) * stride_h + kernel_h - in_h);
    int32_t pad_w = EDGENN_MAX(0, (out_w - 1) * stride_w + kernel_w - in_w);
    *pad_top    = pad_h / 2;
    *pad_bottom = pad_h - *pad_top;
    *pad_left   = pad_w / 2;
    *pad_right  = pad_w - *pad_left;
}

/* ============================================================================
 * MaxPool2D
 * ========================================================================= */

edgenn_status_t edgenn_maxpool2d_execute(
    const edgenn_tensor_t      *input,
    edgenn_tensor_t            *output,
    const edgenn_pool_params_t *params)
{
    EDGENN_CHECK_NULL(input);
    EDGENN_CHECK_NULL(output);
    EDGENN_CHECK_NULL(params);
    EDGENN_CHECK_NULL(input->data);
    EDGENN_CHECK_NULL(output->data);

    if (input->ndim != 4) return EDGENN_ERR_INVALID_ARG;

    int32_t batch    = input->shape[0];
    int32_t in_h     = input->shape[1];
    int32_t in_w     = input->shape[2];
    int32_t channels = input->shape[3];

    int32_t kernel_h = params->kernel_h;
    int32_t kernel_w = params->kernel_w;
    int32_t stride_h = params->stride_h;
    int32_t stride_w = params->stride_w;

    int32_t pad_top = params->pad_top;
    int32_t pad_bottom = params->pad_bottom;
    int32_t pad_left = params->pad_left;
    int32_t pad_right = params->pad_right;

    if (params->padding == EDGENN_PAD_SAME) {
        compute_same_padding(in_h, in_w, kernel_h, kernel_w,
                             stride_h, stride_w,
                             &pad_top, &pad_bottom, &pad_left, &pad_right);
    }

    int32_t out_h = (in_h + pad_top + pad_bottom - kernel_h) / stride_h + 1;
    int32_t out_w = (in_w + pad_left + pad_right - kernel_w) / stride_w + 1;

    if (output->shape[0] != batch || output->shape[1] != out_h ||
        output->shape[2] != out_w || output->shape[3] != channels) {
        return EDGENN_ERR_SHAPE_MISMATCH;
    }

    if (input->dtype == EDGENN_DTYPE_FP32) {
        const float *in = (const float *)input->data;
        float *out = (float *)output->data;

        for (int32_t n = 0; n < batch; n++) {
            for (int32_t oh = 0; oh < out_h; oh++) {
                for (int32_t ow = 0; ow < out_w; ow++) {
                    for (int32_t c = 0; c < channels; c++) {
                        float max_val = -FLT_MAX;
                        for (int32_t kh = 0; kh < kernel_h; kh++) {
                            int32_t ih = oh * stride_h - pad_top + kh;
                            for (int32_t kw = 0; kw < kernel_w; kw++) {
                                int32_t iw = ow * stride_w - pad_left + kw;
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    float v = in[((n * in_h + ih) * in_w + iw) * channels + c];
                                    if (v > max_val) max_val = v;
                                }
                            }
                        }
                        out[((n * out_h + oh) * out_w + ow) * channels + c] = max_val;
                    }
                }
            }
        }
        return EDGENN_OK;
    }

    if (input->dtype == EDGENN_DTYPE_INT8) {
        const int8_t *in = (const int8_t *)input->data;
        int8_t *out = (int8_t *)output->data;

        for (int32_t n = 0; n < batch; n++) {
            for (int32_t oh = 0; oh < out_h; oh++) {
                for (int32_t ow = 0; ow < out_w; ow++) {
                    for (int32_t c = 0; c < channels; c++) {
                        int8_t max_val = -128;
                        for (int32_t kh = 0; kh < kernel_h; kh++) {
                            int32_t ih = oh * stride_h - pad_top + kh;
                            for (int32_t kw = 0; kw < kernel_w; kw++) {
                                int32_t iw = ow * stride_w - pad_left + kw;
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    int8_t v = in[((n * in_h + ih) * in_w + iw) * channels + c];
                                    if (v > max_val) max_val = v;
                                }
                            }
                        }
                        out[((n * out_h + oh) * out_w + ow) * channels + c] = max_val;
                    }
                }
            }
        }
        return EDGENN_OK;
    }

    return EDGENN_ERR_UNSUPPORTED;
}

/* ============================================================================
 * AvgPool2D
 * ========================================================================= */

edgenn_status_t edgenn_avgpool2d_execute(
    const edgenn_tensor_t      *input,
    edgenn_tensor_t            *output,
    const edgenn_pool_params_t *params)
{
    EDGENN_CHECK_NULL(input);
    EDGENN_CHECK_NULL(output);
    EDGENN_CHECK_NULL(params);
    EDGENN_CHECK_NULL(input->data);
    EDGENN_CHECK_NULL(output->data);

    if (input->ndim != 4) return EDGENN_ERR_INVALID_ARG;

    int32_t batch    = input->shape[0];
    int32_t in_h     = input->shape[1];
    int32_t in_w     = input->shape[2];
    int32_t channels = input->shape[3];

    int32_t kernel_h = params->kernel_h;
    int32_t kernel_w = params->kernel_w;
    int32_t stride_h = params->stride_h;
    int32_t stride_w = params->stride_w;

    int32_t pad_top = params->pad_top;
    int32_t pad_bottom = params->pad_bottom;
    int32_t pad_left = params->pad_left;
    int32_t pad_right = params->pad_right;

    if (params->padding == EDGENN_PAD_SAME) {
        compute_same_padding(in_h, in_w, kernel_h, kernel_w,
                             stride_h, stride_w,
                             &pad_top, &pad_bottom, &pad_left, &pad_right);
    }

    int32_t out_h = (in_h + pad_top + pad_bottom - kernel_h) / stride_h + 1;
    int32_t out_w = (in_w + pad_left + pad_right - kernel_w) / stride_w + 1;

    if (output->shape[0] != batch || output->shape[1] != out_h ||
        output->shape[2] != out_w || output->shape[3] != channels) {
        return EDGENN_ERR_SHAPE_MISMATCH;
    }

    if (input->dtype == EDGENN_DTYPE_FP32) {
        const float *in = (const float *)input->data;
        float *out = (float *)output->data;

        for (int32_t n = 0; n < batch; n++) {
            for (int32_t oh = 0; oh < out_h; oh++) {
                for (int32_t ow = 0; ow < out_w; ow++) {
                    for (int32_t c = 0; c < channels; c++) {
                        float sum = 0.0f;
                        int32_t count = 0;
                        for (int32_t kh = 0; kh < kernel_h; kh++) {
                            int32_t ih = oh * stride_h - pad_top + kh;
                            for (int32_t kw = 0; kw < kernel_w; kw++) {
                                int32_t iw = ow * stride_w - pad_left + kw;
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    sum += in[((n * in_h + ih) * in_w + iw) * channels + c];
                                    count++;
                                }
                            }
                        }
                        out[((n * out_h + oh) * out_w + ow) * channels + c] =
                            count > 0 ? sum / (float)count : 0.0f;
                    }
                }
            }
        }
        return EDGENN_OK;
    }

    if (input->dtype == EDGENN_DTYPE_INT8) {
        const int8_t *in = (const int8_t *)input->data;
        int8_t *out = (int8_t *)output->data;

        for (int32_t n = 0; n < batch; n++) {
            for (int32_t oh = 0; oh < out_h; oh++) {
                for (int32_t ow = 0; ow < out_w; ow++) {
                    for (int32_t c = 0; c < channels; c++) {
                        int32_t sum = 0;
                        int32_t count = 0;
                        for (int32_t kh = 0; kh < kernel_h; kh++) {
                            int32_t ih = oh * stride_h - pad_top + kh;
                            for (int32_t kw = 0; kw < kernel_w; kw++) {
                                int32_t iw = ow * stride_w - pad_left + kw;
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    sum += (int32_t)in[((n * in_h + ih) * in_w + iw) * channels + c];
                                    count++;
                                }
                            }
                        }
                        out[((n * out_h + oh) * out_w + ow) * channels + c] =
                            count > 0 ? edgenn_q_sat_i8(sum / count) : 0;
                    }
                }
            }
        }
        return EDGENN_OK;
    }

    return EDGENN_ERR_UNSUPPORTED;
}

/* ============================================================================
 * Global Average Pooling: [N x H x W x C] -> [N x C]
 * ========================================================================= */

edgenn_status_t edgenn_global_avgpool_execute(
    const edgenn_tensor_t *input,
    edgenn_tensor_t       *output)
{
    EDGENN_CHECK_NULL(input);
    EDGENN_CHECK_NULL(output);
    EDGENN_CHECK_NULL(input->data);
    EDGENN_CHECK_NULL(output->data);

    if (input->ndim != 4) return EDGENN_ERR_INVALID_ARG;

    int32_t batch    = input->shape[0];
    int32_t in_h     = input->shape[1];
    int32_t in_w     = input->shape[2];
    int32_t channels = input->shape[3];
    int32_t spatial  = in_h * in_w;

    int32_t out_channels;
    if (output->ndim == 4) {
        out_channels = output->shape[3];
    } else if (output->ndim == 2) {
        out_channels = output->shape[1];
    } else {
        return EDGENN_ERR_INVALID_ARG;
    }

    if (output->shape[0] != batch || out_channels != channels) {
        return EDGENN_ERR_SHAPE_MISMATCH;
    }

    if (input->dtype == EDGENN_DTYPE_FP32) {
        const float *in = (const float *)input->data;
        float *out = (float *)output->data;

        for (int32_t n = 0; n < batch; n++) {
            for (int32_t c = 0; c < channels; c++) {
                float sum = 0.0f;
                for (int32_t h = 0; h < in_h; h++) {
                    for (int32_t w = 0; w < in_w; w++) {
                        sum += in[((n * in_h + h) * in_w + w) * channels + c];
                    }
                }
                out[n * channels + c] = sum / (float)spatial;
            }
        }
        return EDGENN_OK;
    }

    if (input->dtype == EDGENN_DTYPE_INT8) {
        const int8_t *in = (const int8_t *)input->data;
        int8_t *out = (int8_t *)output->data;

        for (int32_t n = 0; n < batch; n++) {
            for (int32_t c = 0; c < channels; c++) {
                int32_t sum = 0;
                for (int32_t h = 0; h < in_h; h++) {
                    for (int32_t w = 0; w < in_w; w++) {
                        sum += (int32_t)in[((n * in_h + h) * in_w + w) * channels + c];
                    }
                }
                out[n * channels + c] = edgenn_q_sat_i8(sum / spatial);
            }
        }
        return EDGENN_OK;
    }

    return EDGENN_ERR_UNSUPPORTED;
}
