cmake_minimum_required(VERSION 3.18)
project(EdgeNN
    VERSION 0.1.0
    DESCRIPTION "High-performance DNN/RNN/Transformer inference library for ARM microcontrollers"
    LANGUAGES C
)

# ==============================================================================
# Build Options
# ==============================================================================
option(EDGENN_BUILD_TESTS      "Build unit tests"                        ON)
option(EDGENN_BUILD_EXAMPLES   "Build example applications"              ON)
option(EDGENN_USE_CMSIS_NN     "Use CMSIS-NN optimized kernels"          OFF)
option(EDGENN_USE_CMSIS_DSP    "Use CMSIS-DSP for math operations"       OFF)
option(EDGENN_USE_HELIUM       "Use ARM Helium/MVE intrinsics"           OFF)
option(EDGENN_USE_NEON         "Use ARM NEON intrinsics (Cortex-A)"      OFF)
option(EDGENN_ENABLE_PROFILING "Enable per-layer cycle profiling"        OFF)
option(EDGENN_ENABLE_LOGGING   "Enable debug logging"                    OFF)
option(EDGENN_FP32_REFERENCE   "Build FP32 reference kernels"            ON)
option(EDGENN_STRICT_WARNINGS  "Enable strict compiler warnings"         ON)

# ==============================================================================
# Compiler Settings
# ==============================================================================
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(EDGENN_STRICT_WARNINGS)
    add_compile_options(
        -Wall -Wextra -Wpedantic -Wshadow -Wdouble-promotion
        -Wformat=2 -Wformat-truncation -Wundef -Wconversion
        -fno-common -ffunction-sections -fdata-sections
    )
endif()

# Optimization: default to -Os for embedded, -O2 for host tests
if(CMAKE_CROSSCOMPILING)
    add_compile_options(-Os -flto)
    add_link_options(-Wl,--gc-sections -flto)
else()
    add_compile_options(-O2 -g)
endif()

# ==============================================================================
# Configuration Header Generation
# ==============================================================================
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/edgenn/edgenn_config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/edgenn/edgenn_config.h
    @ONLY
)

# ==============================================================================
# Core Library
# ==============================================================================
set(EDGENN_CORE_SOURCES
    src/core/edgenn_tensor.c
    src/core/edgenn_arena.c
    src/core/edgenn_math_fp.c
    src/core/edgenn_math_q.c
    src/core/edgenn_quant.c
    src/core/edgenn_lut.c
    src/core/edgenn_log.c
)

set(EDGENN_HAL_SOURCES
    src/hal/edgenn_hal_generic.c
)

set(EDGENN_OPS_DNN_SOURCES
    src/ops/dnn/edgenn_dense.c
    src/ops/dnn/edgenn_conv2d.c
    src/ops/dnn/edgenn_dwconv2d.c
    src/ops/dnn/edgenn_pool.c
    src/ops/dnn/edgenn_activation.c
    src/ops/dnn/edgenn_batchnorm.c
)

set(EDGENN_OPS_RNN_SOURCES
    src/ops/rnn/edgenn_lstm.c
    src/ops/rnn/edgenn_gru.c
    src/ops/rnn/edgenn_rnn_cell.c
)

set(EDGENN_OPS_TRANSFORMER_SOURCES
    src/ops/transformer/edgenn_attention.c
    src/ops/transformer/edgenn_layernorm.c
    src/ops/transformer/edgenn_posenc.c
    src/ops/transformer/edgenn_ffn.c
)

set(EDGENN_RUNTIME_SOURCES
    src/runtime/edgenn_graph.c
    src/runtime/edgenn_model.c
)

# Combine all sources
set(EDGENN_ALL_SOURCES
    ${EDGENN_CORE_SOURCES}
    ${EDGENN_HAL_SOURCES}
    ${EDGENN_OPS_DNN_SOURCES}
    ${EDGENN_OPS_RNN_SOURCES}
    ${EDGENN_OPS_TRANSFORMER_SOURCES}
    ${EDGENN_RUNTIME_SOURCES}
)

add_library(edgenn STATIC ${EDGENN_ALL_SOURCES})

target_include_directories(edgenn
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Conditional CMSIS integration
if(EDGENN_USE_CMSIS_NN)
    target_compile_definitions(edgenn PUBLIC EDGENN_HAS_CMSIS_NN=1)
    # User must provide CMSIS_PATH
    if(DEFINED CMSIS_PATH)
        target_include_directories(edgenn PUBLIC ${CMSIS_PATH}/CMSIS/NN/Include)
        target_include_directories(edgenn PUBLIC ${CMSIS_PATH}/CMSIS/DSP/Include)
        target_include_directories(edgenn PUBLIC ${CMSIS_PATH}/CMSIS/Core/Include)
    endif()
endif()

if(EDGENN_USE_HELIUM)
    target_compile_definitions(edgenn PUBLIC EDGENN_HAS_HELIUM=1)
endif()

if(EDGENN_USE_NEON)
    target_compile_definitions(edgenn PUBLIC EDGENN_HAS_NEON=1)
endif()

if(EDGENN_ENABLE_PROFILING)
    target_compile_definitions(edgenn PUBLIC EDGENN_PROFILING=1)
endif()

if(EDGENN_ENABLE_LOGGING)
    target_compile_definitions(edgenn PUBLIC EDGENN_LOGGING=1)
endif()

# ==============================================================================
# Tests
# ==============================================================================
if(EDGENN_BUILD_TESTS AND NOT CMAKE_CROSSCOMPILING)
    enable_testing()
    add_subdirectory(tests)
endif()

# ==============================================================================
# Examples
# ==============================================================================
if(EDGENN_BUILD_EXAMPLES AND NOT CMAKE_CROSSCOMPILING)
    add_subdirectory(examples)
endif()

# ==============================================================================
# Install Rules
# ==============================================================================
install(TARGETS edgenn
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(DIRECTORY include/edgenn DESTINATION include)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/edgenn/edgenn_config.h
    DESTINATION include/edgenn
)

message(STATUS "====================================")
message(STATUS "EdgeNN v${PROJECT_VERSION} Configuration:")
message(STATUS "  CMSIS-NN:    ${EDGENN_USE_CMSIS_NN}")
message(STATUS "  Helium/MVE:  ${EDGENN_USE_HELIUM}")
message(STATUS "  NEON:        ${EDGENN_USE_NEON}")
message(STATUS "  Profiling:   ${EDGENN_ENABLE_PROFILING}")
message(STATUS "  Logging:     ${EDGENN_ENABLE_LOGGING}")
message(STATUS "  Tests:       ${EDGENN_BUILD_TESTS}")
message(STATUS "  Examples:    ${EDGENN_BUILD_EXAMPLES}")
message(STATUS "====================================")
