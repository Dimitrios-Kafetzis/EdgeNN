/**
 * @file edgenn_conv2d.c
 * @brief 2D Convolution operator — im2col + GEMM approach, FP32 and INT8
 *
 * Input layout: NHWC [batch × height × width × channels]
 * Weight layout: [out_ch × kernel_h × kernel_w × in_ch]
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn/ops/dnn/edgenn_conv2d.h"
#include "edgenn/core/edgenn_math_fp.h"
#include "edgenn/core/edgenn_math_q.h"
#include "edgenn/core/edgenn_lut.h"

/* ============================================================================
 * Padding Computation
 * ========================================================================= */

static void compute_same_padding(
    int32_t in_h, int32_t in_w,
    int32_t kernel_h, int32_t kernel_w,
    int32_t stride_h, int32_t stride_w,
    int32_t dilation_h, int32_t dilation_w,
    int32_t *pad_top, int32_t *pad_bottom,
    int32_t *pad_left, int32_t *pad_right)
{
    int32_t eff_kh = (kernel_h - 1) * dilation_h + 1;
    int32_t eff_kw = (kernel_w - 1) * dilation_w + 1;
    int32_t out_h = (in_h + stride_h - 1) / stride_h;
    int32_t out_w = (in_w + stride_w - 1) / stride_w;
    int32_t pad_h = EDGENN_MAX(0, (out_h - 1) * stride_h + eff_kh - in_h);
    int32_t pad_w = EDGENN_MAX(0, (out_w - 1) * stride_w + eff_kw - in_w);
    *pad_top    = pad_h / 2;
    *pad_bottom = pad_h - *pad_top;
    *pad_left   = pad_w / 2;
    *pad_right  = pad_w - *pad_left;
}

static void compute_output_dims(
    int32_t in_h, int32_t in_w,
    int32_t kernel_h, int32_t kernel_w,
    int32_t stride_h, int32_t stride_w,
    int32_t dilation_h, int32_t dilation_w,
    int32_t pad_top, int32_t pad_bottom,
    int32_t pad_left, int32_t pad_right,
    int32_t *out_h, int32_t *out_w)
{
    int32_t eff_kh = (kernel_h - 1) * dilation_h + 1;
    int32_t eff_kw = (kernel_w - 1) * dilation_w + 1;
    *out_h = (in_h + pad_top + pad_bottom - eff_kh) / stride_h + 1;
    *out_w = (in_w + pad_left + pad_right - eff_kw) / stride_w + 1;
}

/* ============================================================================
 * FP32 Conv2D — direct nested loops
 * ========================================================================= */

static void conv2d_fp32_direct(
    const float *input,     /* [N×H×W×C_in] */
    const float *weights,   /* [C_out × kH × kW × C_in] */
    const float *bias,
    float       *output,    /* [N×oH×oW×C_out] */
    int32_t batch, int32_t in_h, int32_t in_w, int32_t in_ch,
    int32_t out_h, int32_t out_w, int32_t out_ch,
    int32_t kernel_h, int32_t kernel_w,
    int32_t stride_h, int32_t stride_w,
    int32_t dilation_h, int32_t dilation_w,
    int32_t pad_top, int32_t pad_left)
{
    for (int32_t n = 0; n < batch; n++) {
        for (int32_t oh = 0; oh < out_h; oh++) {
            for (int32_t ow = 0; ow < out_w; ow++) {
                for (int32_t oc = 0; oc < out_ch; oc++) {
                    float acc = bias ? bias[oc] : 0.0f;

                    for (int32_t kh = 0; kh < kernel_h; kh++) {
                        int32_t ih = oh * stride_h - pad_top + kh * dilation_h;
                        for (int32_t kw = 0; kw < kernel_w; kw++) {
                            int32_t iw = ow * stride_w - pad_left + kw * dilation_w;

                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                for (int32_t ic = 0; ic < in_ch; ic++) {
                                    float iv = input[((n * in_h + ih) * in_w + iw) * in_ch + ic];
                                    float wv = weights[((oc * kernel_h + kh) * kernel_w + kw) * in_ch + ic];
                                    acc += iv * wv;
                                }
                            }
                        }
                    }
                    output[((n * out_h + oh) * out_w + ow) * out_ch + oc] = acc;
                }
            }
        }
    }
}

/* ============================================================================
 * INT8 Conv2D — direct nested loops with per-channel requant
 * ========================================================================= */

static void conv2d_q8_direct(
    const int8_t  *input,
    const int8_t  *weights,
    const int32_t *bias,
    int8_t        *output,
    int32_t batch, int32_t in_h, int32_t in_w, int32_t in_ch,
    int32_t out_h, int32_t out_w, int32_t out_ch,
    int32_t kernel_h, int32_t kernel_w,
    int32_t stride_h, int32_t stride_w,
    int32_t dilation_h, int32_t dilation_w,
    int32_t pad_top, int32_t pad_left,
    int32_t input_zp,
    const int32_t *channel_mult,
    const int8_t  *channel_shift,
    int32_t out_mult, int8_t out_shift,
    int32_t out_zp)
{
    for (int32_t n = 0; n < batch; n++) {
        for (int32_t oh = 0; oh < out_h; oh++) {
            for (int32_t ow = 0; ow < out_w; ow++) {
                for (int32_t oc = 0; oc < out_ch; oc++) {
                    int32_t acc = bias ? bias[oc] : 0;

                    for (int32_t kh = 0; kh < kernel_h; kh++) {
                        int32_t ih = oh * stride_h - pad_top + kh * dilation_h;
                        for (int32_t kw = 0; kw < kernel_w; kw++) {
                            int32_t iw = ow * stride_w - pad_left + kw * dilation_w;

                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                for (int32_t ic = 0; ic < in_ch; ic++) {
                                    int32_t iv = (int32_t)input[((n * in_h + ih) * in_w + iw) * in_ch + ic] - input_zp;
                                    int32_t wv = (int32_t)weights[((oc * kernel_h + kh) * kernel_w + kw) * in_ch + ic];
                                    acc += iv * wv;
                                }
                            }
                        }
                    }

                    int32_t result;
                    if (channel_mult) {
                        result = edgenn_q_requantize(acc, channel_mult[oc], channel_shift[oc]);
                    } else {
                        result = edgenn_q_requantize(acc, out_mult, out_shift);
                    }
                    result += out_zp;
                    output[((n * out_h + oh) * out_w + ow) * out_ch + oc] = edgenn_q_sat_i8(result);
                }
            }
        }
    }
}

/* ============================================================================
 * Fused Activations (shared helpers)
 * ========================================================================= */

static edgenn_status_t apply_fp32_act(float *data, int32_t n, edgenn_act_type_t act)
{
    switch (act) {
        case EDGENN_ACT_NONE:    break;
        case EDGENN_ACT_RELU:    edgenn_fp32_relu(data, data, n); break;
        case EDGENN_ACT_RELU6:   edgenn_fp32_relu6(data, data, n); break;
        case EDGENN_ACT_SIGMOID: edgenn_fp32_sigmoid(data, data, n); break;
        case EDGENN_ACT_TANH:    edgenn_fp32_tanh(data, data, n); break;
        case EDGENN_ACT_GELU:    edgenn_fp32_gelu(data, data, n); break;
        default: return EDGENN_ERR_UNSUPPORTED;
    }
    return EDGENN_OK;
}

static edgenn_status_t apply_q8_act(
    int8_t *data, int32_t n, edgenn_act_type_t act,
    int8_t zp, int8_t six_q)
{
    switch (act) {
        case EDGENN_ACT_NONE:    break;
        case EDGENN_ACT_RELU:    edgenn_q8_relu(data, data, n, zp); break;
        case EDGENN_ACT_RELU6:   edgenn_q8_relu6(data, data, n, zp, six_q); break;
        case EDGENN_ACT_SIGMOID: edgenn_q8_sigmoid_lut(data, data, n, edgenn_lut_sigmoid_default); break;
        case EDGENN_ACT_TANH:    edgenn_q8_tanh_lut(data, data, n, edgenn_lut_tanh_default); break;
        default: return EDGENN_ERR_UNSUPPORTED;
    }
    return EDGENN_OK;
}

/* ============================================================================
 * Public API
 * ========================================================================= */

edgenn_status_t edgenn_conv2d_execute(
    const edgenn_tensor_t        *input,
    edgenn_tensor_t              *output,
    const edgenn_conv2d_params_t *params,
    edgenn_arena_t               *scratch)
{
    (void)scratch;

    EDGENN_CHECK_NULL(input);
    EDGENN_CHECK_NULL(output);
    EDGENN_CHECK_NULL(params);
    EDGENN_CHECK_NULL(input->data);
    EDGENN_CHECK_NULL(output->data);
    EDGENN_CHECK_NULL(params->weights.data);

    if (input->ndim != 4) return EDGENN_ERR_INVALID_ARG;
    if (input->dtype != params->weights.dtype) return EDGENN_ERR_DTYPE_MISMATCH;

    int32_t batch = input->shape[0];
    int32_t in_h  = input->shape[1];
    int32_t in_w  = input->shape[2];
    int32_t in_ch = input->shape[3];

    int32_t kernel_h = params->kernel_h;
    int32_t kernel_w = params->kernel_w;
    int32_t stride_h = params->stride_h;
    int32_t stride_w = params->stride_w;
    int32_t dilation_h = params->dilation_h > 0 ? params->dilation_h : 1;
    int32_t dilation_w = params->dilation_w > 0 ? params->dilation_w : 1;

    int32_t pad_top = params->pad_top;
    int32_t pad_bottom = params->pad_bottom;
    int32_t pad_left = params->pad_left;
    int32_t pad_right = params->pad_right;

    if (params->padding == EDGENN_PAD_SAME) {
        compute_same_padding(in_h, in_w, kernel_h, kernel_w,
                             stride_h, stride_w, dilation_h, dilation_w,
                             &pad_top, &pad_bottom, &pad_left, &pad_right);
    }

    int32_t out_h, out_w;
    compute_output_dims(in_h, in_w, kernel_h, kernel_w,
                        stride_h, stride_w, dilation_h, dilation_w,
                        pad_top, pad_bottom, pad_left, pad_right,
                        &out_h, &out_w);

    int32_t out_ch = params->weights.shape[0];

    if (output->shape[0] != batch || output->shape[1] != out_h ||
        output->shape[2] != out_w || output->shape[3] != out_ch) {
        return EDGENN_ERR_SHAPE_MISMATCH;
    }

    int32_t numel = batch * out_h * out_w * out_ch;

    if (input->dtype == EDGENN_DTYPE_FP32) {
        const float *bias_ptr = params->bias.data
                              ? (const float *)params->bias.data : NULL;

        conv2d_fp32_direct(
            (const float *)input->data,
            (const float *)params->weights.data,
            bias_ptr,
            (float *)output->data,
            batch, in_h, in_w, in_ch,
            out_h, out_w, out_ch,
            kernel_h, kernel_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            pad_top, pad_left);

        return apply_fp32_act((float *)output->data, numel, params->fused_act);
    }

    if (input->dtype == EDGENN_DTYPE_INT8) {
        int32_t input_zp = input->qparams.zero_point;
        int32_t out_zp   = output->qparams.zero_point;

        const int32_t *bias_ptr = params->bias.data
                                ? (const int32_t *)params->bias.data : NULL;

        conv2d_q8_direct(
            (const int8_t *)input->data,
            (const int8_t *)params->weights.data,
            bias_ptr,
            (int8_t *)output->data,
            batch, in_h, in_w, in_ch,
            out_h, out_w, out_ch,
            kernel_h, kernel_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            pad_top, pad_left,
            input_zp,
            params->output_mult,
            params->output_shift,
            output->qparams.multiplier,
            output->qparams.shift,
            out_zp);

        int8_t six_q = edgenn_q_sat_i8(
            (int32_t)(6.0f / output->qparams.scale) + out_zp);

        return apply_q8_act((int8_t *)output->data, numel,
                            params->fused_act, (int8_t)out_zp, six_q);
    }

    return EDGENN_ERR_UNSUPPORTED;
}

size_t edgenn_conv2d_scratch_size(
    const edgenn_conv2d_params_t *params,
    const int32_t *input_shape)
{
    if (!params || !input_shape) return 0;
    /* im2col buffer: out_w × kH × kW × in_ch */
    int32_t in_w  = input_shape[2];
    int32_t in_ch = input_shape[3];
    int32_t out_w = (in_w + params->stride_w - 1) / params->stride_w;
    return (size_t)out_w * (size_t)params->kernel_h
         * (size_t)params->kernel_w * (size_t)in_ch;
}
