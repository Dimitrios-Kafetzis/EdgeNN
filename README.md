<p align="center">
  <h1 align="center">EdgeNN</h1>
  <p align="center">
    <strong>High-performance neural network inference for ARM microcontrollers</strong>
  </p>
  <p align="center">
    Pure C11 &nbsp;&bull;&nbsp; Zero malloc &nbsp;&bull;&nbsp; INT8 quantized &nbsp;&bull;&nbsp; 118 tests passing
  </p>
</p>

---

EdgeNN is a lightweight, deterministic neural network inference library written in pure C11. It runs **Dense, Convolutional, Recurrent (LSTM/GRU), and Transformer** models on resource-constrained ARM Cortex-M and Cortex-A devices with **zero dynamic memory allocation** during inference.

Designed for bare-metal deployment in safety-critical and real-time applications — maritime autonomous navigation, industrial IoT, defense systems, and beyond.

## Features

- **Zero-malloc inference** — Static arena allocator with save/restore, no heap usage, real-time safe
- **Quantization-first** — INT8 symmetric with INT32 accumulation, fixed-point requantization, FP32 reference fallback
- **Full operator coverage** — 18 operators across DNN, RNN, and Transformer architectures
- **Graph runtime** — Sequential layer executor with automatic scratch management and binary model loading
- **Platform-ready** — Generic C fallback today; CMSIS-NN, Helium/MVE, and NEON backends planned
- **Deterministic execution** — No floating-point variance, bounded cycle count per layer
- **Tiny footprint** — Dead code elimination via LTO, modular `static` library, `< 50 KB` Flash target

## Operator Coverage

| Category | Operators | FP32 | INT8 |
|----------|-----------|:----:|:----:|
| **DNN** | Dense (Fully Connected) | :white_check_mark: | :white_check_mark: |
| | Conv2D | :white_check_mark: | :white_check_mark: |
| | Depthwise Conv2D | :white_check_mark: | :white_check_mark: |
| | MaxPool2D / AvgPool2D | :white_check_mark: | :white_check_mark: |
| | Batch Normalization | :white_check_mark: | :white_check_mark: |
| | ReLU, ReLU6, Sigmoid, Tanh, GELU, Softmax | :white_check_mark: | :white_check_mark: |
| **RNN** | Simple RNN Cell | :white_check_mark: | — |
| | LSTM (cell + sequence) | :white_check_mark: | — |
| | GRU (cell + sequence) | :white_check_mark: | — |
| **Transformer** | Multi-Head Attention (+ KV cache) | :white_check_mark: | — |
| | Layer Normalization | :white_check_mark: | — |
| | Feed-Forward Network (FFN) | :white_check_mark: | — |
| | Positional Encoding (sinusoidal) | :white_check_mark: | — |
| **Utility** | Element-wise Add | :white_check_mark: | — |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  User API                                                     │
│  edgenn_model_load_buffer()  edgenn_model_infer()             │
├──────────────────────────────────────────────────────────────┤
│  Graph Runtime                                                │
│  Static graph executor · Layer dispatch · Scratch management  │
├──────────────────────────────────────────────────────────────┤
│  Operator Kernels                                             │
│  ┌──────────┬──────────┬─────────────┬──────────────────┐    │
│  │ DNN Ops  │ RNN Ops  │ Transformer │ Utility          │    │
│  │ Dense    │ LSTM     │ Attention   │ Add              │    │
│  │ Conv2D   │ GRU      │ LayerNorm   │ Reshape          │    │
│  │ DWConv   │ RNN Cell │ FFN         │ Concat           │    │
│  │ Pool     │          │ PosEnc      │                  │    │
│  │ BN · Act │          │             │                  │    │
│  └──────────┴──────────┴─────────────┴──────────────────┘    │
├──────────────────────────────────────────────────────────────┤
│  Quantization & Math Backend                                  │
│  INT8/INT16/FP32 dispatch · Fixed-point multiply/shift        │
│  Activation LUTs · Requantization · Per-channel scaling       │
├──────────────────────────────────────────────────────────────┤
│  Hardware Abstraction Layer (HAL)                              │
│  Generic C  │  CMSIS-NN/DSP  │  Helium/MVE  │  NEON          │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

### Build & Test

```bash
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

All **118 unit tests** across 5 test suites should pass:

```
1/5 Test #1: test_core ................   Passed
2/5 Test #2: test_dnn_ops .............   Passed
3/5 Test #3: test_rnn_ops .............   Passed
4/5 Test #4: test_transformer_ops .....   Passed
5/5 Test #5: test_runtime .............   Passed

100% tests passed, 0 tests failed out of 5
```

### Cross-Compile for ARM

```bash
cmake -B build-arm \
  -DCMAKE_TOOLCHAIN_FILE=cmake/arm-cortex-m7.cmake \
  -DEDGENN_USE_CMSIS_NN=ON \
  -DCMSIS_PATH=/path/to/CMSIS_5
cmake --build build-arm
```

### Usage Example

```c
#include "edgenn/edgenn.h"
#include <string.h>

/* Static memory pools — no malloc, ever */
static uint8_t weight_buf[32768];
static uint8_t scratch_buf[8192];

int main(void)
{
    /* 1. Initialize arenas */
    edgenn_arena_t weight_arena, scratch_arena;
    edgenn_arena_init(&weight_arena, weight_buf, sizeof(weight_buf));
    edgenn_arena_init(&scratch_arena, scratch_buf, sizeof(scratch_buf));

    /* 2. Create input tensor [1 x 4], FP32 */
    edgenn_tensor_t input;
    int32_t in_shape[] = {1, 4};
    edgenn_tensor_init(&input, in_shape, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_arena_alloc_tensor(&scratch_arena, &input);

    float *x = (float *)input.data;
    x[0] = 1.0f;  x[1] = 2.0f;  x[2] = 3.0f;  x[3] = 4.0f;

    /* 3. Run a Dense layer: [1x4] -> [1x2] */
    edgenn_tensor_t output;
    int32_t out_shape[] = {1, 2};
    edgenn_tensor_init(&output, out_shape, 2, EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_arena_alloc_tensor(&scratch_arena, &output);

    edgenn_dense_params_t params = {
        .weight    = &my_weight_tensor,   /* [2 x 4] */
        .bias      = &my_bias_tensor,     /* [2] */
        .activation = EDGENN_ACT_RELU,
    };
    edgenn_status_t status = edgenn_dense_execute(&input, &output, &params, NULL);
    if (status != EDGENN_OK) { /* handle error */ }

    /* 4. Or load and run a full model */
    edgenn_model_t model;
    edgenn_model_load_buffer(&model, model_data, model_size,
                             &weight_arena, &scratch_arena);
    edgenn_model_infer(&model, &input, &output);

    return 0;
}
```

## Target Hardware

| Tier | Platform | Cores | Key Features |
|------|----------|-------|-------------|
| **Primary** | Cortex-M4 | STM32F4, NXP LPC4xxx | DSP + SP FPU |
| **Primary** | Cortex-M7 | STM32F7/H7, NXP i.MX RT | DSP, DP FPU, D/I cache |
| **Primary** | Cortex-M33 | STM32L5, NXP LPC55S69 | ARMv8-M, TrustZone, DSP |
| **Primary** | Cortex-M55 | Corstone-300, Alif Ensemble | Helium/MVE (128-bit SIMD) |
| **Stretch** | Cortex-A53/A72 | Raspberry Pi 3/4, NXP i.MX8 | NEON SIMD, 64-bit |

## Memory Model

EdgeNN uses **zero dynamic allocation**. All memory is managed through two static arenas:

| Arena | Purpose | Lifetime |
|-------|---------|----------|
| **Weight arena** | Model weights, bias tensors, persistent state | Entire model lifetime |
| **Scratch arena** | Layer activations, temporaries | Reset between layers |

Example memory budgets:

| Model | Weights | Peak Activations | Total SRAM |
|-------|---------|-----------------|------------|
| Keyword Spotting DNN (3x Dense) | 40 KB | 2 KB | 42 KB |
| Anomaly Detection LSTM (2 layers) | 120 KB | 8 KB | 128 KB |
| Tiny Transformer (2L, d=64, seq=32) | 200 KB | 32 KB | 232 KB |

## Project Status

| Phase | Description | Tests | Status |
|-------|-------------|------:|--------|
| 1 | Foundation — types, arena, tensor, math, quantization, HAL | 24 | :white_check_mark: Complete |
| 2 | DNN Operators — Dense, Conv2D, DWConv2D, Pool, BN, Activations | 37 | :white_check_mark: Complete |
| 3 | RNN Operators — Simple RNN, LSTM, GRU (cell + sequence) | 20 | :white_check_mark: Complete |
| 4 | Transformer Operators — Attention, LayerNorm, FFN, PosEnc | 21 | :white_check_mark: Complete |
| 5 | Graph Runtime & Binary Model Format | 16 | :white_check_mark: Complete |
| 6 | Optimization — CMSIS-NN backends, operator fusion, INT4 | — | :construction: Planned |
| 7 | Benchmarks & Platform Testing | — | :construction: Planned |

**Total: 118 tests passing, 0 warnings**

## Project Structure

```
edgenn/
├── CMakeLists.txt                    # Build system (CMake 3.18+)
├── LICENSE                           # MIT License
│
├── include/edgenn/
│   ├── edgenn.h                      # Umbrella header
│   ├── core/
│   │   ├── edgenn_types.h            # Enums, macros, type definitions
│   │   ├── edgenn_status.h           # Error codes, CHECK macros
│   │   ├── edgenn_tensor.h           # Tensor descriptor
│   │   ├── edgenn_arena.h            # Arena allocator
│   │   ├── edgenn_math_fp.h          # FP32 math (matmul, activations, etc.)
│   │   ├── edgenn_math_q.h           # INT8/INT16 quantized math
│   │   ├── edgenn_quant.h            # Quantize / dequantize utilities
│   │   └── edgenn_lut.h              # Activation lookup tables
│   ├── ops/
│   │   ├── dnn/                      # Dense, Conv2D, DWConv, Pool, BN, Act
│   │   ├── rnn/                      # LSTM, GRU, Simple RNN
│   │   └── transformer/              # Attention, LayerNorm, FFN, PosEnc
│   ├── runtime/
│   │   ├── edgenn_graph.h            # Static graph executor
│   │   └── edgenn_model.h            # Binary model loader
│   └── hal/
│       └── edgenn_hal.h              # Hardware abstraction
│
├── src/                              # All .c implementations (mirrors include/)
├── tests/
│   ├── edgenn_test.h                 # Minimal test framework (no dependencies)
│   ├── test_core.c                   # 24 tests — foundation
│   ├── test_dnn_ops.c               # 37 tests — DNN operators
│   ├── test_rnn_ops.c               # 20 tests — RNN operators
│   ├── test_transformer_ops.c       # 21 tests — Transformer operators
│   └── test_runtime.c               # 16 tests — graph + model
│
├── examples/
│   └── hello_edgenn.c                # Minimal inference demo
├── docs/
│   └── ARCHITECTURE.md               # Full system architecture
└── cmake/
    └── arm-cortex-m7.cmake           # ARM cross-compilation toolchain
```

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `EDGENN_BUILD_TESTS` | ON | Build unit tests |
| `EDGENN_BUILD_EXAMPLES` | ON | Build example programs |
| `EDGENN_USE_CMSIS_NN` | OFF | CMSIS-NN optimized kernels |
| `EDGENN_USE_HELIUM` | OFF | Helium/MVE intrinsics (Cortex-M55) |
| `EDGENN_USE_NEON` | OFF | NEON intrinsics (Cortex-A) |
| `EDGENN_ENABLE_PROFILING` | OFF | Per-layer cycle counting |
| `EDGENN_ENABLE_LOGGING` | OFF | Debug logging (compile-time removable) |
| `EDGENN_FP32_REFERENCE` | ON | FP32 reference kernels |

## Design Principles

1. **No malloc, ever** — All memory through arena allocators. No `malloc`, `calloc`, `realloc`, or `free` anywhere in the codebase.

2. **Every function returns a status** — All public functions return `edgenn_status_t`. Errors propagate cleanly via the `EDGENN_CHECK()` macro. No `abort()`, `exit()`, or `assert()`.

3. **Quantization without floats** — INT8 inference uses INT32 accumulators and fixed-point requantization (`multiplier × shift`). No FPU required on the hot path.

4. **Data does not copy** — Tensors are lightweight descriptors (not owners). Weight tensors point directly into the model buffer (zero-copy). Scratch arenas swap between layers.

5. **C11, no dependencies** — Pure standard C11. No external libraries. All headers have `extern "C"` guards for C++ interop.

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full system architecture, data structures, operator specs |

## Contributing

EdgeNN is in active development. The core inference engine (Phases 1–5) is complete with full test coverage. Upcoming work includes:

- **CMSIS-NN backends** for Cortex-M acceleration
- **NEON/Helium SIMD** optimized kernels
- **Operator fusion** (Conv+BN+ReLU, Dense+Activation)
- **INT4 weight quantization** for extreme compression
- **Python model converter** (PyTorch/TFLite to `.edgenn` format)
- **On-device benchmarks** on STM32H7 and Raspberry Pi

Contributions, benchmarks, and bug reports are welcome.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built for the edge. No compromises.
</p>
