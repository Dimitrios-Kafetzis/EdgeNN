# Contributing to EdgeNN

Thank you for your interest in contributing to EdgeNN! This guide explains how to get started.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/EdgeNN.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Read `docs/ARCHITECTURE.md` for the system design

## Building and Testing

```bash
cmake -B build -DEDGENN_ENABLE_LOGGING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Code Style

- **Language:** C11, no compiler extensions required
- **Naming:** `edgenn_module_function()` for public API, `EDGENN_MACRO` for macros
- **Indentation:** 4 spaces, no tabs
- **Headers:** Include guards with `#ifndef EDGENN_FILENAME_H`
- **Documentation:** Every public function needs `@brief`, `@param`, `@return`
- **Error handling:** Return `edgenn_status_t`, use `EDGENN_CHECK()` macro
- **Memory:** NO malloc/free. Use `edgenn_arena_t` for all allocations.

## What to Contribute

### Good First Issues

Look for issues labeled `good-first-issue`. These are:
- Adding unit tests for existing operators
- Documentation improvements
- Small optimizations in math functions

### High-Impact Contributions

- CMSIS-NN backend integration
- NEON/Helium SIMD optimized kernels
- Benchmark suite for real ARM hardware
- Python model converter tool
- Operator fusion (Conv+BN+ReLU, Dense+Activation)

## Pull Request Process

1. Ensure all tests pass
2. Add tests for new functionality
3. Update documentation if API changes
4. Fill out the PR template completely
5. One approval required for merge

## Reporting Bugs

Use the GitHub issue templates. Include:
- EdgeNN version/commit
- Target platform
- Minimal reproduction code
- Expected vs actual behavior

## Questions?

Open a [GitHub Discussion](https://github.com/Dimitrios-Kafetzis/EdgeNN/discussions).
