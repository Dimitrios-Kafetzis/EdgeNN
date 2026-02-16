# Changelog

All notable changes to EdgeNN will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for v0.6.0 — Phase 6: Optimization

- CMSIS-NN optimized backends for Cortex-M (Dense, Conv2D, DWConv2D, Pool)
- Operator fusion passes (Conv+BN+ReLU, Dense+Activation)
- INT4 weight quantization with INT8 accumulation
- Helium/MVE SIMD kernels for Cortex-M55
- NEON SIMD kernels for Cortex-A53/A72
- Per-layer cycle count profiling and benchmark suite

## [0.5.0] - 2026-02-16

### Added
- **Phase 5 — Graph Runtime & Binary Model Format**
- Static graph executor (`edgenn_graph_t`) with sequential layer dispatch
- Automatic scratch arena management between layers (save/restore)
- Binary model loader (`edgenn_model_load_buffer`) with zero-copy weight mapping
- Model inference API (`edgenn_model_infer`) for end-to-end execution
- Layer configuration dispatch for all 18 operator types
- 16 unit tests for graph execution, model loading, and multi-layer inference

## [0.4.0] - 2026-02-16

### Added
- **Phase 4 — Transformer Operators (FP32)**
- Multi-Head Attention with scaled dot-product and KV cache support
- Layer Normalization with learnable gain/bias
- Feed-Forward Network (FFN) with configurable hidden dimension and GELU/ReLU activation
- Sinusoidal Positional Encoding generator
- Element-wise Add utility operator
- 21 unit tests covering all transformer operators

## [0.3.0] - 2026-02-15

### Added
- **Phase 3 — RNN Operators (FP32)**
- Simple RNN cell with tanh activation
- LSTM cell with input/forget/cell/output gates and peephole-free design
- LSTM sequence processor with configurable time steps
- GRU cell with reset/update gates
- GRU sequence processor with configurable time steps
- 20 unit tests covering all RNN operators and sequence processing

## [0.2.0] - 2026-02-15

### Added
- **Phase 2 — DNN Operators**
- Dense (fully connected) layer — FP32 and INT8 per-channel quantized
- Conv2D — FP32 and INT8 with stride, padding (SAME/VALID), and dilation
- Depthwise Conv2D — FP32 and INT8 with per-channel quantization
- MaxPool2D and AvgPool2D — FP32 and INT8
- Batch Normalization — FP32 and INT8 (fused scale/offset)
- Activation functions — ReLU, ReLU6, Sigmoid, Tanh, GELU, Softmax (FP32 and INT8)
- 37 unit tests covering all DNN operators in both FP32 and INT8 paths

## [0.1.0] - 2026-02-15

### Added
- **Phase 1 — Foundation complete**
- Core type system: `edgenn_types.h` with dtype, op_type, layout, quantization enums
- Error handling: `edgenn_status_t` with 13 error codes and CHECK macros
- Tensor descriptor: init, numel, byte_size, slice, shape comparison
- Arena allocator: bump pointer, save/restore, tensor allocation, ping-pong buffers
- FP32 math: matmul, sigmoid, tanh, relu, relu6, gelu, softmax, layernorm
- INT8 math: matmul (per-tensor and per-channel), relu, relu6, sigmoid/tanh LUT, softmax
- Quantization utilities: fp32-to-int8/int16, compute_multiplier, symmetric/asymmetric params
- LUT generation: sigmoid, tanh, gelu lookup tables for INT8 activation
- HAL: platform detection, cycle counter (DWT on Cortex-M, clock_gettime on host)
- Logging: compile-time removable with callback support
- CMake build system with cross-compilation support
- 24 unit tests (all passing)
- Example application: hello_edgenn
- API headers for all Phase 2-5 operators (stubs)

### Infrastructure
- MIT License
- README with quick start, architecture overview, roadmap
- Architecture documentation (`docs/ARCHITECTURE.md`)
- ARM Cortex-M7 cross-compilation toolchain file

[Unreleased]: https://github.com/Dimitrios-Kafetzis/EdgeNN/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/Dimitrios-Kafetzis/EdgeNN/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Dimitrios-Kafetzis/EdgeNN/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Dimitrios-Kafetzis/EdgeNN/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Dimitrios-Kafetzis/EdgeNN/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Dimitrios-Kafetzis/EdgeNN/releases/tag/v0.1.0
