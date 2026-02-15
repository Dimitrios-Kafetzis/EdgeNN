# EdgeNN — System Architecture Document

**Version:** 0.1.0
**Last Updated:** 2026-02-15
**Author:** Dimitrios Kafetzis / PhD Researcher @ Athens University of Economics and Business & Director of IoT Department @ DeepSea Technologies

---

## 1. Executive Summary

EdgeNN is a pure C11 library for high-performance, deterministic neural network inference on ARM microcontrollers. It supports Dense (Fully Connected), Convolutional, Recurrent (LSTM/GRU), and Transformer (Multi-Head Attention) architectures, with a quantization-first design targeting INT8 symmetric quantization as the primary execution path.

The library is designed for bare-metal deployment with zero dynamic memory allocation during inference, making it suitable for safety-critical and real-time applications including maritime autonomous navigation, industrial IoT, and defense systems.

---

## 2. Target Hardware

### 2.1 Primary Targets

| Platform | Cores | Key Features | SRAM | Flash |
|---|---|---|---|---|
| **Cortex-M4** | STM32F4, NXP LPC4xxx | DSP (SIMD MAC), optional SP FPU | 64–256 KB | 256 KB–1 MB |
| **Cortex-M7** | STM32F7/H7, NXP i.MX RT1060 | DSP, DP FPU, D/I cache, TCM | 256 KB–2 MB | 512 KB–2 MB |
| **Cortex-M33** | STM32L5, NXP LPC55S69 | ARMv8-M TrustZone, DSP | 64–640 KB | 256 KB–1 MB |
| **Cortex-M55** | Corstone-300, Alif Ensemble | **Helium/MVE** (128-bit SIMD) | 256 KB–4 MB | 512 KB–4 MB |

### 2.2 Stretch Targets

| Platform | Cores | Key Features |
|---|---|---|
| **Cortex-A53/A72** | Raspberry Pi 3/4, NXP i.MX8 | NEON (128-bit SIMD), 64-bit, MMU/cache |
| **Cortex-A7** | STM32MP1 (heterogeneous) | NEON (32-bit), co-processor with M4 |

### 2.3 Hard Constraints

- **No dynamic allocation in inference path** — all memory pre-planned at model load
- **No OS dependency** — bare-metal first, RTOS-compatible (FreeRTOS, Zephyr)
- **Deterministic execution** — bounded cycle count per layer, no FP variance
- **No C standard library dependency in core** — `memcpy`/`memset` may use HAL alternatives
- **C11 standard** — no compiler extensions required (GCC extensions optional for optimization)
- **Endianness** — little-endian only (all ARM targets)

---

## 3. Architecture Layers

The library is organized in 6 vertical layers. Each layer depends only on layers below it.

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 6: User API                                           │
│  edgenn_model_load(), edgenn_model_infer(), edgenn_*()       │
├──────────────────────────────────────────────────────────────┤
│  Layer 5: Graph Runtime                                      │
│  Static graph executor, layer dispatch, buffer management    │
├──────────────────────────────────────────────────────────────┤
│  Layer 4: Operator Kernels                                   │
│  ┌──────────┬──────────┬───────────┬─────────────────────┐   │
│  │ DNN Ops  │ RNN Ops  │ Xformer   │ Utility Ops         │   │
│  │ Dense    │ LSTM     │ Attention │ Reshape, Concat     │   │
│  │ Conv2D   │ GRU      │ LayerNorm │ Add, Multiply       │   │
│  │ DWConv   │ RNN Cell │ FFN       │ Flatten             │   │
│  │ Pool     │          │ PosEnc    │                     │   │
│  │ BN, Act  │          │           │                     │   │
│  └──────────┴──────────┴───────────┴─────────────────────┘   │
├──────────────────────────────────────────────────────────────┤
│  Layer 3: Quantization & Math Backend                        │
│  INT8/INT16/FP32 dispatch, fixed-point multiply/shift        │
│  Activation LUTs, requantization, per-channel scaling        │
├──────────────────────────────────────────────────────────────┤
│  Layer 2: Hardware Abstraction Layer (HAL)                   │
│  ┌─────────────────┬────────────────┬───────────────────┐    │
│  │ Generic C       │ CMSIS-NN/DSP   │ Helium/MVE        │    │
│  │ (all platforms) │ (Cortex-M)     │ (Cortex-M55)      │    │
│  │                 │                │ NEON (Cortex-A)   │    │
│  └─────────────────┴────────────────┴───────────────────┘    │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: Platform / Board Support                           │
│  Memory map, clock config, DMA, cycle counter, stdio         │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Directory Structure

```
edgenn/
├── CMakeLists.txt                  # Top-level build
├── LICENSE                         # MIT License
├── README.md                       # Project overview
├── cmake/
│   └── arm-cortex-m7.cmake         # Cross-compilation toolchain
├── docs/
│   └── ARCHITECTURE.md             # This document
├── examples/
│   ├── CMakeLists.txt
│   └── hello_edgenn.c              # Minimal inference demo
├── include/edgenn/
│   ├── edgenn.h                    # Umbrella header
│   ├── edgenn_config.h.in          # CMake-generated config
│   ├── core/
│   │   ├── edgenn_types.h          # Fundamental types, enums, macros
│   │   ├── edgenn_status.h         # Error codes, CHECK macros
│   │   ├── edgenn_tensor.h         # Tensor descriptor API
│   │   ├── edgenn_arena.h          # Arena allocator, ping-pong buffers
│   │   ├── edgenn_math_fp.h        # FP32 reference math
│   │   ├── edgenn_math_q.h         # INT8/INT16 quantized math
│   │   ├── edgenn_quant.h          # Quantize/dequantize utilities
│   │   ├── edgenn_lut.h            # Activation LUT generation
│   │   └── edgenn_log.h            # Compile-time removable logging
│   ├── hal/
│   │   └── edgenn_hal.h            # HAL: cycle counter, memcpy, platform detection
│   ├── ops/
│   │   ├── dnn/
│   │   │   ├── edgenn_dense.h      # Fully Connected
│   │   │   ├── edgenn_conv2d.h     # 2D Convolution
│   │   │   ├── edgenn_dwconv2d.h   # Depthwise Conv2D
│   │   │   ├── edgenn_pool.h       # MaxPool, AvgPool, GlobalAvgPool
│   │   │   ├── edgenn_activation.h # Standalone activations
│   │   │   └── edgenn_batchnorm.h  # Batch Normalization
│   │   ├── rnn/
│   │   │   ├── edgenn_lstm.h       # LSTM cell + sequence
│   │   │   ├── edgenn_gru.h        # GRU cell + sequence
│   │   │   └── edgenn_rnn_cell.h   # Simple RNN cell
│   │   └── transformer/
│   │       ├── edgenn_attention.h   # Multi-Head Attention + KV cache
│   │       ├── edgenn_layernorm.h   # Layer Normalization
│   │       ├── edgenn_posenc.h      # Positional Encoding
│   │       └── edgenn_ffn.h         # Feed-Forward Network
│   └── runtime/
│       ├── edgenn_graph.h           # Static graph executor
│       └── edgenn_model.h           # Model loading (.edgenn format)
├── src/
│   ├── core/                        # Corresponding .c implementations
│   ├── hal/
│   ├── ops/{dnn,rnn,transformer}/
│   └── runtime/
├── tests/
│   ├── CMakeLists.txt
│   ├── edgenn_test.h               # Minimal test framework
│   └── test_core.c                 # Phase 1 unit tests
└── tools/                           # Python converter (Phase 5)
```

---

## 5. Core Data Structures

### 5.1 Tensor Descriptor

The tensor is a lightweight descriptor — it does NOT own its data. Data is always managed externally (arena, Flash, user buffer).

```c
typedef struct {
    void            *data;                      // Raw data pointer
    int32_t          shape[EDGENN_MAX_DIMS];    // Up to 4 dimensions
    int32_t          strides[EDGENN_MAX_DIMS];  // Element strides (row-major)
    uint8_t          ndim;                      // Active dimensions (1-4)
    edgenn_dtype_t   dtype;                     // INT8, INT16, INT32, FP32
    edgenn_layout_t  layout;                    // NHWC, NCHW, NC, NTC
    edgenn_qparams_t qparams;                   // Quantization parameters
} edgenn_tensor_t;
```

**Design rationale:** No heap allocation, no reference counting, no virtual dispatch. A tensor is 80-100 bytes on stack. Operators receive tensor pointers and work directly on the underlying data.

### 5.2 Quantization Parameters

```c
typedef struct {
    edgenn_qscheme_t  scheme;          // NONE, SYMMETRIC, ASYMMETRIC, PER_CHANNEL
    float             scale;           // Per-tensor scale
    int32_t           zero_point;      // Per-tensor zero point
    const float      *channel_scales;  // Per-channel (NULL if unused)
    const int32_t    *channel_zps;     // Per-channel zero points
    int32_t           n_channels;      // Number of channels
    int32_t           multiplier;      // Fixed-point multiplier (runtime)
    int8_t            shift;           // Right-shift amount (runtime)
} edgenn_qparams_t;
```

**Key insight:** For pure-integer inference on Cortex-M (no FPU), the `multiplier` and `shift` fields allow requantization without any floating-point operations. The conversion is:

```
real_value ≈ scale × (quantized_value - zero_point)
output_q   = clamp(requantize(accumulator, multiplier, shift) + output_zp, -128, 127)
requantize = (int32_t)((int64_t)value * multiplier >> 31) >> shift
```

### 5.3 Arena Allocator

```c
typedef struct {
    uint8_t     *base;          // Base address
    size_t       capacity;      // Total bytes
    size_t       offset;        // Current watermark (bump pointer)
    size_t       peak;          // Lifetime high watermark
    uint32_t     alloc_count;   // Allocation counter
} edgenn_arena_t;
```

**Two arenas are used at runtime:**
1. **Weight arena** — loaded once at model init, holds all weight tensors, never freed during lifetime
2. **Scratch arena** — reset between layers via ping-pong scheme, holds activations and temporaries

### 5.4 Ping-Pong Buffer

```c
typedef struct {
    edgenn_arena_t  arena_a;
    edgenn_arena_t  arena_b;
    uint8_t         active;     // Which is currently "output"
} edgenn_pingpong_t;
```

During inference, the output of layer N becomes the input of layer N+1. Instead of copying, we swap which arena is "input" and which is "output". The old input arena (now consumed) is reset for reuse as the next output. This reduces peak memory by ~50% compared to separate input/output buffers.

---

## 6. Quantization Strategy

### 6.1 Primary Path: INT8 Symmetric

- Zero-point = 0 for weights (symmetric)
- Scale computed per-channel for weights, per-tensor for activations
- Accumulation in INT32 (32-bit MAC)
- Compatible with TFLite quantization-aware training (QAT)

### 6.2 Sensitive Layers: INT16

- LSTM cell state: INT16 to prevent precision loss across time steps
- Attention scores (Q×K^T): INT16 accumulation before softmax
- Layer normalization: INT32 accumulation for mean/variance

### 6.3 Fallback: FP32

- Reference implementation for every operator
- Used for Cortex-A targets with NEON (where FP32 throughput is high)
- Used for numerical validation in unit tests

### 6.4 Requantization Pipeline

For each operator, the quantization flow is:

```
Input (INT8) → Dequantize to INT32 accumulator
           → Compute (INT32 MAC)
           → Add INT32 bias
           → Requantize: multiply by output_multiplier, shift by output_shift
           → Add output_zero_point
           → Saturate to INT8 [-128, 127]
           → (Optional) Apply fused activation via LUT or clamping
```

The multiplier/shift pair is precomputed at model load time from:
```
real_multiplier = (input_scale × weight_scale) / output_scale
```

This is decomposed via `frexpf()` into a normalized significand in [0.5, 1.0) mapped to INT32 range [2^30, 2^31), and an exponent that becomes the shift amount.

---

## 7. Operator Specifications

### 7.1 Dense (Fully Connected)

**Computation:** `output[m][n] = Σ_k input[m][k] × weight[n][k] + bias[n]`

| Aspect | Specification |
|---|---|
| Input shape | [batch × in_features] |
| Weight shape | [out_features × in_features] (transposed) |
| Bias shape | [out_features], INT32 |
| Accumulator | INT32 (no overflow for K ≤ 4096 with INT8 inputs) |
| Quantization | Per-channel (one multiplier/shift per output feature) |
| Fused activations | ReLU, ReLU6, Sigmoid, Tanh, GELU |
| CMSIS-NN backend | `arm_fully_connected_s8()` |
| Scratch memory | 0 bytes (in-place accumulation) |

### 7.2 Conv2D

**Computation:** Standard 2D convolution with optional dilation and groups.

| Aspect | Specification |
|---|---|
| Input layout | NHWC [batch × height × width × channels] |
| Kernel layout | [out_ch × kH × kW × in_ch] |
| Strategy | im2col + GEMM for kH×kW > 1; direct for 1×1 |
| Padding | VALID, SAME, or explicit (top/bottom/left/right) |
| Scratch memory | im2col buffer: kH × kW × in_ch × sizeof(int8_t) per output row |
| CMSIS-NN backend | `arm_convolve_s8()` or `arm_convolve_1x1_s8_fast()` |

### 7.3 Depthwise Conv2D

| Aspect | Specification |
|---|---|
| Kernel layout | [channels × kH × kW × 1] |
| Depth multiplier | Default 1 (separable convolution) |
| CMSIS-NN backend | `arm_depthwise_conv_s8()` |

### 7.4 Pooling

| Operator | Description |
|---|---|
| MaxPool2D | Maximum over kernel window, dtype-preserving |
| AvgPool2D | Average with INT32 accumulation, requantize output |
| GlobalAvgPool | Average over spatial dimensions (H×W) |

### 7.5 LSTM Cell

**Gate equations (per time step):**
```
i = σ(W_ii × x + W_hi × h + b_i)    // input gate
f = σ(W_if × x + W_hf × h + b_f)    // forget gate
g = tanh(W_ig × x + W_hg × h + b_g)  // cell gate
o = σ(W_io × x + W_ho × h + b_o)    // output gate
c' = f ⊙ c + i ⊙ g                    // new cell state
h' = o ⊙ tanh(c')                     // new hidden state
```

| Aspect | Specification |
|---|---|
| Gate computation | Fused: concat [x, h], single matmul for all 4 gates |
| Gate activations | σ = sigmoid via 256-entry INT8 LUT |
| Cell state precision | INT16 (to avoid accumulation drift across steps) |
| Weight layout | [4×hidden × (input+hidden)] (interleaved IFGO) |
| Scratch | 4×hidden×sizeof(int32_t) for gate accumulators |

### 7.6 GRU Cell

```
z = σ(W_z × [x, h])    // update gate
r = σ(W_r × [x, h])    // reset gate
n = tanh(W_n × [x, r⊙h])  // new gate
h' = (1-z) ⊙ n + z ⊙ h
```

### 7.7 Multi-Head Attention

**Computation:**
```
Q = input × W_Q + b_Q    // [seq × d_model] → [seq × d_model]
K = input × W_K + b_K
V = input × W_V + b_V
// Split into heads: [seq × n_heads × d_head]
for each head h:
    scores[h] = Q[h] × K[h]^T / sqrt(d_head)   // [seq × seq]
    weights[h] = softmax(scores[h])
    context[h] = weights[h] × V[h]               // [seq × d_head]
output = concat(context[0..n_heads]) × W_O + b_O
```

| Aspect | Specification |
|---|---|
| Score computation | INT16 accumulation (Q×K^T can overflow INT8) |
| Softmax | Fixed-point approximation or INT16 LUT |
| KV cache | Optional, for autoregressive generation |
| Max sequence length | Configurable (default 512, limited by SRAM) |
| Scratch | Q,K,V projections + scores matrix + softmax buffer |

### 7.8 Layer Normalization

```
y = gamma × (x - mean) / sqrt(var + epsilon) + beta
```

| Aspect | Specification |
|---|---|
| Accumulation | INT32 for mean, INT64 for variance (if needed) |
| Division | Fixed-point reciprocal of sqrt(var+eps) |
| Output | Requantized to INT8 |

### 7.9 Feed-Forward Network (FFN)

```
hidden = GELU(input × W1 + b1)    // d_model → d_ff (typically 4×d_model)
output = hidden × W2 + b2          // d_ff → d_model
```

Two Dense layers with GELU activation between them. The GELU is implemented via a 256-entry INT8 LUT.

---

## 8. Memory Management Strategy

### 8.1 Memory Zones

```
Flash (read-only):
├── Model weights (quantized INT8)
├── Bias values (INT32)
├── Activation LUTs (256 bytes each)
├── Quantization scale/shift tables
└── Model metadata / layer descriptors

SRAM (read-write):
├── Weight Arena (loaded from Flash at init, or memory-mapped)
├── Scratch Arena A (ping-pong)
├── Scratch Arena B (ping-pong)
├── RNN state buffers (persistent across inference calls)
├── KV cache (persistent for autoregressive Transformer)
└── Stack (minimal, for local variables only)
```

### 8.2 Memory Planning

At model load time, the runtime performs a dry-run to compute:
1. Peak scratch memory needed (maximum over all layers)
2. Total weight memory needed
3. Tensor lifetimes (when each intermediate tensor is first produced and last consumed)

This enables the optimal ping-pong allocation where two buffers of size `max_layer_activation_size` suffice for the entire forward pass.

### 8.3 Memory Budget Examples

| Model | Weights | Peak Activations | Total SRAM |
|---|---|---|---|
| Keyword Spotting DNN (3×Dense) | 40 KB | 2 KB | 42 KB |
| Anomaly Detection LSTM (2 layers) | 120 KB | 8 KB | 128 KB |
| Tiny Transformer (2 layers, d=64) | 200 KB | 32 KB | 232 KB |

---

## 9. Build System

### 9.1 CMake Configuration

The build system uses CMake 3.18+ with the following key options:

| Option | Default | Description |
|---|---|---|
| `EDGENN_BUILD_TESTS` | ON | Build unit tests |
| `EDGENN_BUILD_EXAMPLES` | ON | Build example programs |
| `EDGENN_USE_CMSIS_NN` | OFF | Enable CMSIS-NN optimized kernels |
| `EDGENN_USE_CMSIS_DSP` | OFF | Enable CMSIS-DSP math functions |
| `EDGENN_USE_HELIUM` | OFF | Enable Helium/MVE intrinsics |
| `EDGENN_USE_NEON` | OFF | Enable NEON intrinsics |
| `EDGENN_ENABLE_PROFILING` | OFF | Per-layer cycle counting |
| `EDGENN_ENABLE_LOGGING` | OFF | Debug logging (compile-time removable) |
| `EDGENN_FP32_REFERENCE` | ON | Build FP32 reference kernels |

### 9.2 Cross-Compilation

```bash
# Host build (for testing)
cmake -B build && cmake --build build && ctest --test-dir build

# ARM Cortex-M7 cross-compile
cmake -B build-arm -DCMAKE_TOOLCHAIN_FILE=cmake/arm-cortex-m7.cmake \
      -DEDGENN_USE_CMSIS_NN=ON -DCMSIS_PATH=/path/to/CMSIS_5
cmake --build build-arm
```

### 9.3 Compiler Flags

- **Host:** `-O2 -g` (for testing and debugging)
- **Embedded:** `-Os -flto -ffunction-sections -fdata-sections` + `-Wl,--gc-sections` (minimal binary size)
- **Warnings:** `-Wall -Wextra -Wpedantic -Wshadow -Wdouble-promotion -Wconversion`

---

## 10. Error Handling

All public API functions return `edgenn_status_t`:

```c
typedef enum {
    EDGENN_OK                   =  0,
    EDGENN_ERR_NULL_PTR         = -1,
    EDGENN_ERR_INVALID_ARG      = -2,
    EDGENN_ERR_OUT_OF_MEMORY    = -3,
    EDGENN_ERR_SHAPE_MISMATCH   = -4,
    EDGENN_ERR_DTYPE_MISMATCH   = -5,
    EDGENN_ERR_UNSUPPORTED      = -6,
    EDGENN_ERR_BUFFER_TOO_SMALL = -7,
    EDGENN_ERR_MODEL_INVALID    = -8,
    EDGENN_ERR_MODEL_VERSION    = -9,
    EDGENN_ERR_LAYER_LIMIT      = -10,
    EDGENN_ERR_QUANT_OVERFLOW   = -11,
    EDGENN_ERR_NOT_INITIALIZED  = -12,
    EDGENN_ERR_INTERNAL         = -99,
} edgenn_status_t;
```

**Convention:** Use `EDGENN_CHECK(expr)` macro for early-return on error. Use `EDGENN_CHECK_NULL(ptr)` for null-pointer validation. Every function validates its arguments at entry.

---

## 11. Testing Strategy

### 11.1 Unit Tests

Every operator has tests comparing:
1. **INT8 quantized output** vs **FP32 reference output** (within quantization error tolerance)
2. **Edge cases:** zero-length inputs, maximum dimensions, overflow scenarios
3. **Bit-exact tests:** known input → known output (regression tests)

### 11.2 Test Framework

Minimal custom framework (`tests/edgenn_test.h`) with:
- `ASSERT_OK(status)` — check for EDGENN_OK
- `ASSERT_NEAR(a, b, tol)` — floating-point comparison
- `ASSERT_EQ(a, b)` — exact equality
- `TEST_CASE("name")` / `TEST_PASS()` / `TEST_FAIL("msg")`

### 11.3 Accuracy Metrics

For quantized operators, we measure:
- **Max absolute error** vs FP32 reference
- **Mean squared error** (MSE)
- **Signal-to-Quantization-Noise Ratio** (SQNR) in dB
- Target: SQNR > 30 dB for INT8, > 50 dB for INT16

---

## 12. Coding Standards

### 12.1 Naming Convention

| Entity | Convention | Example |
|---|---|---|
| Public functions | `edgenn_<module>_<action>` | `edgenn_tensor_init()` |
| Public types | `edgenn_<name>_t` | `edgenn_tensor_t` |
| Public enums | `EDGENN_<CATEGORY>_<VALUE>` | `EDGENN_DTYPE_INT8` |
| Public macros | `EDGENN_<NAME>` | `EDGENN_MAX_DIMS` |
| Internal functions | `_edgenn_<module>_<action>` | `_edgenn_arena_align()` |
| File names | `edgenn_<module>.h/c` | `edgenn_tensor.h` |

### 12.2 Code Style

- C11 standard, no compiler extensions required
- 4-space indentation, no tabs
- Opening brace on same line for functions and control flow
- Every public function documented with Doxygen `@brief`, `@param`, `@return`
- Header guards use `#ifndef EDGENN_<FILENAME>_H` / `#define` / `#endif`
- All headers wrapped in `extern "C"` for C++ compatibility

### 12.3 MISRA-C Compliance

The following MISRA-C:2012 rules are targeted for compliance (not mandatory but recommended for safety-critical paths):
- Rule 11.3: No cast between pointer to object and pointer to different object type
- Rule 14.4: Controlling expression of if/while must be boolean
- Rule 17.7: Return value of non-void function must be used
- Rule 21.3: No `<stdlib.h>` memory allocation functions (`malloc`, `free`, etc.)

---

## 13. Performance Targets

### 13.1 Benchmarks (Cortex-M7 @ 480 MHz, STM32H7)

| Model | Target Latency | Target Memory |
|---|---|---|
| MobileNet-v2 (224×224, INT8) | < 50 ms | < 512 KB SRAM |
| LSTM (128 hidden, 100 steps) | < 10 ms | < 32 KB SRAM |
| Tiny Transformer (2L, d=64, seq=32) | < 20 ms | < 128 KB SRAM |
| Keyword Spotting (DS-CNN) | < 5 ms | < 64 KB SRAM |

### 13.2 Comparison Targets

- Match or exceed CMSIS-NN for individual operator performance
- Match or exceed TFLite Micro for end-to-end model inference
- Smaller code footprint than TFLite Micro (target: < 50 KB Flash for core library)

---

## 14. Future / Advanced Features

### 14.1 Model Partitioning (Research Integration)

Leveraging the DNN partitioning research (WiOpt 2025), EdgeNN will support:
- Splitting a model at layer boundaries across multiple devices
- Mixed Flash/SRAM execution (weights in Flash, activations in SRAM)
- Speculative execution hints for multi-layer transformer partitioning

### 14.2 Operator Fusion

Compile-time and runtime fusion patterns:
- Conv2D + BatchNorm + ReLU → single fused kernel
- Dense + Activation → fused output clamping/LUT
- LayerNorm + Dense → fused accumulation

### 14.3 INT4 Quantization

Weight-only INT4 for extreme compression:
- 2 weights packed per byte
- Dequantize to INT8 at inference time (small overhead)
- ~2× weight size reduction vs INT8

---

## Appendix A: Dependency Graph

```
edgenn_types.h ← (no dependencies)
    ↑
edgenn_status.h ← edgenn_types.h
    ↑
edgenn_tensor.h ← edgenn_types.h, edgenn_status.h
edgenn_arena.h  ← edgenn_types.h, edgenn_status.h
edgenn_math_fp.h ← edgenn_types.h, edgenn_status.h
edgenn_math_q.h  ← edgenn_types.h, edgenn_status.h
edgenn_quant.h   ← edgenn_types.h, edgenn_status.h
edgenn_lut.h     ← edgenn_types.h, edgenn_status.h
edgenn_log.h     ← edgenn_types.h
edgenn_hal.h     ← edgenn_types.h, edgenn_status.h
    ↑
edgenn_dense.h   ← edgenn_tensor.h, edgenn_arena.h, edgenn_types.h
edgenn_conv2d.h  ← edgenn_tensor.h, edgenn_arena.h, edgenn_types.h
edgenn_lstm.h    ← edgenn_tensor.h, edgenn_arena.h, edgenn_types.h
edgenn_attention.h ← edgenn_tensor.h, edgenn_arena.h, edgenn_types.h
    ↑
edgenn_graph.h   ← edgenn_tensor.h, edgenn_arena.h, all ops
edgenn_model.h   ← edgenn_graph.h, edgenn_arena.h
    ↑
edgenn.h         ← everything (umbrella)
```

## Appendix B: Model Binary Format (.edgenn)

```
Offset  Size      Description
0x00    4 bytes   Magic: 0x454E4E45 ("ENNE")
0x04    4 bytes   Version: 1
0x08    4 bytes   Number of layers
0x0C    4 bytes   Number of tensors
0x10    4 bytes   Weight data size (bytes)
0x14    4 bytes   Metadata size (bytes)
0x18    4 bytes   Flags (bitfield)
0x1C    16 bytes  Reserved
0x2C    variable  Layer descriptors (n_layers × sizeof(layer_desc))
        variable  Tensor descriptors (n_tensors × sizeof(tensor_desc))
        variable  Weight data blob (aligned to 16 bytes)
        variable  Metadata (JSON or binary, optional)
```
