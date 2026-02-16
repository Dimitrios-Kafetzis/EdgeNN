# Good First Issues

Welcome to EdgeNN! These starter issues are designed for new contributors to get familiar with the codebase. Each issue is self-contained and includes everything you need to get started.

**Before you begin:** Read [CONTRIBUTING.md](../CONTRIBUTING.md) for build instructions, code style, and PR guidelines.

---

## 1. Add unit tests for FP32 tanh edge cases

**Difficulty:** Easy

**Files to modify:**
- `tests/test_core.c`

**Description:**
The current FP32 tanh tests cover basic functionality but lack edge case coverage. Add tests that exercise boundary conditions: very large positive/negative inputs (should saturate to +1/-1), positive and negative infinity, zero, and values near the saturation region (e.g., +/- 6.0).

**Acceptance criteria:**
- [ ] Test `edgenn_fp32_tanh` with inputs: `0.0f`, `1e6f`, `-1e6f`, `INFINITY`, `-INFINITY`
- [ ] Verify output saturates to `+1.0` / `-1.0` for large magnitude inputs
- [ ] Verify `tanh(0.0) == 0.0` exactly
- [ ] All new tests pass via `ctest --test-dir build --output-on-failure`
- [ ] No modifications to existing tests

---

## 2. Add ASSERT_ARRAY_NEAR and ASSERT_ARRAY_EQ macros to test framework

**Difficulty:** Easy

**Files to modify:**
- `tests/edgenn_test.h`

**Description:**
Currently, testing arrays requires manual loops with `ASSERT_NEAR` or `ASSERT_EQ` for each element. Add two convenience macros that compare entire arrays in a single call, printing the first mismatched index and values on failure.

**Acceptance criteria:**
- [ ] `ASSERT_ARRAY_EQ(expected, actual, len)` — compares `len` elements for exact equality
- [ ] `ASSERT_ARRAY_NEAR(expected, actual, len, tol)` — compares `len` float elements within tolerance `tol`
- [ ] On failure, the macro prints the index, expected value, and actual value of the first mismatch
- [ ] Macros follow the existing `ASSERT_*` pattern in `edgenn_test.h` (use `do { ... } while(0)`)
- [ ] Add at least one test case in `tests/test_core.c` exercising each new macro

---

## 3. Implement edgenn_tensor_reshape()

**Difficulty:** Easy-Medium

**Files to modify:**
- `include/edgenn/core/edgenn_tensor.h` (add function declaration)
- `src/core/edgenn_tensor.c` (implement)
- `tests/test_core.c` (add tests)

**Description:**
Implement a logical reshape that changes the tensor's shape and ndim without copying data. The new shape must have the same total number of elements (`edgenn_tensor_numel`). This is a metadata-only operation — the `data` pointer must not change.

**Acceptance criteria:**
- [ ] Function signature: `edgenn_status_t edgenn_tensor_reshape(edgenn_tensor_t *t, const int32_t *new_shape, int32_t new_ndim)`
- [ ] Returns `EDGENN_ERR_SHAPE` if `numel(new_shape) != numel(old_shape)`
- [ ] Returns `EDGENN_ERR_PARAM` if `new_ndim > EDGENN_MAX_DIMS`
- [ ] Returns `EDGENN_ERR_NULL` if any pointer is NULL
- [ ] `t->data` pointer is unchanged after reshape
- [ ] At least 3 test cases: valid reshape, element count mismatch (error), ndim overflow (error)

---

## 4. Add Doxygen configuration file

**Difficulty:** Easy

**Files to modify:**
- `docs/Doxyfile` (new file)

**Description:**
Create a Doxygen configuration file that generates HTML documentation from the existing `/** @brief ... */` comments in all public headers under `include/edgenn/`. Verify that every public function in the generated output has at least `@brief`, `@param`, and `@return` documentation.

**Acceptance criteria:**
- [ ] `docs/Doxyfile` configured with `INPUT = ../include/edgenn`, `RECURSIVE = YES`
- [ ] `PROJECT_NAME = EdgeNN`, output to `docs/html/`
- [ ] Running `doxygen docs/Doxyfile` generates documentation without errors
- [ ] No undocumented public functions in the Doxygen warnings output
- [ ] Add `docs/html/` to `.gitignore`

---

## 5. Implement in-place operation support for activations

**Difficulty:** Medium

**Files to modify:**
- `src/core/edgenn_math_fp.c`
- `tests/test_core.c`

**Description:**
Several activation functions (ReLU, ReLU6) can safely operate in-place where the input and output buffers are the same pointer. Verify and test that `edgenn_fp32_relu` and `edgenn_fp32_relu6` work correctly when `output == input`. Document this guarantee in the function's header comment.

**Acceptance criteria:**
- [ ] `edgenn_fp32_relu(data, data, len)` produces correct results (input pointer == output pointer)
- [ ] `edgenn_fp32_relu6(data, data, len)` produces correct results
- [ ] Add test cases that call each function with `output == input` and verify results
- [ ] Add `@note Supports in-place operation (output == input)` to the header doc comments
- [ ] No changes to the function signatures

---

## 6. Add INT8 tanh accuracy test (LUT vs FP32 reference)

**Difficulty:** Easy

**Files to modify:**
- `tests/test_core.c`

**Description:**
The INT8 tanh implementation uses a 256-entry lookup table. Add a comprehensive accuracy test that feeds all 256 possible INT8 input values (-128 to +127) through both the INT8 LUT path and the FP32 reference path, then compares the results. The maximum error should be within 1 quantization step.

**Acceptance criteria:**
- [ ] Loop over all 256 INT8 values (-128 to +127)
- [ ] For each value: dequantize to FP32, compute `tanh` via FP32 reference, quantize result back to INT8
- [ ] Compare FP32-derived INT8 result with LUT result
- [ ] Maximum absolute error <= 1 (one quantization step)
- [ ] Print summary: number of exact matches, max error observed
- [ ] Test passes in the existing `test_core` suite

---

## 7. Create a benchmark harness for edgenn_fp32_matmul

**Difficulty:** Medium

**Files to modify:**
- `tests/bench_matmul.c` (new file)
- `tests/CMakeLists.txt` (add benchmark executable)

**Description:**
Create a standalone benchmark program that measures the cycle count (using `edgenn_hal_cycle_count()`) for `edgenn_fp32_matmul` at various matrix sizes. This helps establish baseline performance before SIMD optimization in Phase 6.

**Acceptance criteria:**
- [ ] Benchmark sizes: 8x8, 16x16, 32x32, 64x64, 128x128
- [ ] Each size runs at least 10 iterations, reports min/avg/max cycles
- [ ] Uses `edgenn_hal_cycle_count()` for timing (not `clock()`)
- [ ] Output is a readable table printed to stdout
- [ ] Added to `tests/CMakeLists.txt` as a separate executable (not part of `ctest`)
- [ ] Uses arena allocation for all buffers (no `malloc`)

---

## 8. Add CMake option to build as shared library

**Difficulty:** Medium

**Files to modify:**
- `CMakeLists.txt`

**Description:**
Currently EdgeNN builds only as a static library (`edgenn_static`). Add a CMake option `EDGENN_BUILD_SHARED` (default OFF) that also builds a shared library target (`edgenn_shared`). This is useful for dynamic linking in host-side testing and Python bindings (future work).

**Acceptance criteria:**
- [ ] New option: `option(EDGENN_BUILD_SHARED "Build shared library" OFF)`
- [ ] When ON, creates target `edgenn_shared` as a `SHARED` library with the same sources
- [ ] Shared library has proper `VERSION` and `SOVERSION` properties
- [ ] Static library (`edgenn_static`) continues to build regardless of this option
- [ ] `cmake -B build -DEDGENN_BUILD_SHARED=ON && cmake --build build` succeeds
- [ ] All existing tests still link against the static library and pass
