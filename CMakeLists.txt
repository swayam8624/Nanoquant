cmake_minimum_required(VERSION 3.20)

project(NanoQuant
    VERSION 0.1.0
    DESCRIPTION "C++ model weight compression experiments for local and edge inference"
    LANGUAGES CXX)

option(NANOQUANT_BUILD_TESTS "Build NanoQuant tests" ON)
option(NANOQUANT_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(NANOQUANT_ENABLE_METAL "Build optional Apple Metal backend discovery and kernels" ON)
option(NANOQUANT_BUILD_C_API "Build the stable C ABI shared library" ON)

add_library(nanoquant_core
    src/backend.cpp
    src/binary_tensor.cpp
    src/bitpack.cpp
    src/gguf.cpp
    src/quantization.cpp
    src/tensor.cpp
    src/workflow.cpp)

target_include_directories(nanoquant_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

target_compile_features(nanoquant_core PUBLIC cxx_std_20)

if (APPLE AND NANOQUANT_ENABLE_METAL)
    enable_language(OBJCXX)
    target_sources(nanoquant_core PRIVATE src/metal_backend.mm)
    target_compile_definitions(nanoquant_core PRIVATE NANOQUANT_HAS_METAL=1)
    target_link_libraries(nanoquant_core PUBLIC "-framework Foundation" "-framework Metal")
endif()

if (MSVC)
    target_compile_options(nanoquant_core PRIVATE /W4)
    if (NANOQUANT_WARNINGS_AS_ERRORS)
        target_compile_options(nanoquant_core PRIVATE /WX)
    endif()
else()
    target_compile_options(nanoquant_core PRIVATE -Wall -Wextra -Wpedantic)
    if (NANOQUANT_WARNINGS_AS_ERRORS)
        target_compile_options(nanoquant_core PRIVATE -Werror)
    endif()
endif()

add_executable(nanoquant apps/nanoquant.cpp)
target_link_libraries(nanoquant PRIVATE nanoquant_core)

if (NANOQUANT_BUILD_C_API)
    set_target_properties(nanoquant_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(nanoquant_c SHARED src/c_api.cpp)
    target_link_libraries(nanoquant_c PRIVATE nanoquant_core)
    target_include_directories(nanoquant_c
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)
    target_compile_features(nanoquant_c PUBLIC cxx_std_20)
endif()

if (NANOQUANT_BUILD_TESTS)
    enable_testing()
    add_executable(nanoquant_tests tests/test_core.cpp)
    target_link_libraries(nanoquant_tests PRIVATE nanoquant_core)
    add_test(NAME nanoquant_tests COMMAND nanoquant_tests)
endif()
