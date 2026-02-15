/**
 * @file hello_edgenn.c
 * @brief Minimal EdgeNN example — arena, tensor, FP32 matmul, quantization
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn/edgenn.h"
#include <stdio.h>
#include <string.h>

/* Static memory pools (simulates MCU SRAM) */
static uint8_t weight_pool[4096];
static uint8_t scratch_pool[4096];

int main(void)
{
    printf("EdgeNN v%d.%d.%d — %s\n\n",
        0, 1, 0, edgenn_hal_platform_name());

    /* 1. Initialize arenas */
    edgenn_arena_t weight_arena, scratch_arena;
    edgenn_arena_init(&weight_arena, weight_pool, sizeof(weight_pool));
    edgenn_arena_init(&scratch_arena, scratch_pool, sizeof(scratch_pool));

    /* 2. Create input tensor [1 × 4] */
    edgenn_tensor_t input;
    int32_t in_shape[] = {1, 4};
    edgenn_tensor_init(&input, in_shape, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_arena_alloc_tensor(&scratch_arena, &input);
    float *in_data = (float *)input.data;
    in_data[0] = 1.0f; in_data[1] = 2.0f; in_data[2] = 3.0f; in_data[3] = 4.0f;

    /* 3. Create weight tensor [4 × 2] and output [1 × 2] */
    edgenn_tensor_t weights, output;
    int32_t w_shape[] = {4, 2};
    int32_t o_shape[] = {1, 2};
    edgenn_tensor_init(&weights, w_shape, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&output, o_shape, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_arena_alloc_tensor(&weight_arena, &weights);
    edgenn_arena_alloc_tensor(&scratch_arena, &output);

    float *w_data = (float *)weights.data;
    float w_init[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    memcpy(w_data, w_init, sizeof(w_init));

    /* 4. FP32 matmul: [1×4] × [4×2] = [1×2] */
    edgenn_fp32_matmul(in_data, w_data, NULL, (float *)output.data, 1, 2, 4);
    float *out = (float *)output.data;
    printf("FP32 matmul result: [%.4f, %.4f]\n", out[0], out[1]);
    printf("  Expected:         [5.0000, 6.0000]\n");

    /* 5. Apply activations */
    float relu_out[2], sigmoid_out[2];
    float neg_test[] = {-1.0f, 3.0f};
    edgenn_fp32_relu(neg_test, relu_out, 2);
    edgenn_fp32_sigmoid(neg_test, sigmoid_out, 2);
    printf("\nReLU([-1, 3]):    [%.4f, %.4f]\n", relu_out[0], relu_out[1]);
    printf("Sigmoid([-1, 3]): [%.4f, %.4f]\n", sigmoid_out[0], sigmoid_out[1]);

    /* 6. Quantize the output to INT8 */
    edgenn_qparams_t qp;
    edgenn_quant_params_symmetric(-6.0f, 6.0f, &qp);
    int8_t q_output[2];
    edgenn_quant_fp32_to_int8(out, q_output, 2, qp.scale, qp.zero_point);
    printf("\nQuantized INT8 (scale=%.6f): [%d, %d]\n", qp.scale, q_output[0], q_output[1]);

    /* 7. Dequantize back */
    float deq[2];
    edgenn_quant_int8_to_fp32(q_output, deq, 2, qp.scale, qp.zero_point);
    printf("Dequantized FP32:            [%.4f, %.4f]\n", deq[0], deq[1]);

    /* 8. Memory usage */
    printf("\nMemory usage:\n");
    printf("  Weight arena:  %zu / %zu bytes (peak: %zu)\n",
        weight_arena.offset, weight_arena.capacity, edgenn_arena_peak(&weight_arena));
    printf("  Scratch arena: %zu / %zu bytes (peak: %zu)\n",
        scratch_arena.offset, scratch_arena.capacity, edgenn_arena_peak(&scratch_arena));

    printf("\nPhase 1 foundation working correctly!\n");
    return 0;
}
