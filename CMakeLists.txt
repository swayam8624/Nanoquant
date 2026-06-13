cmake_minimum_required(VERSION 3.20)

project(NanoQuant
    VERSION 0.1.0
    DESCRIPTION "C++ model weight compression experiments for local and edge inference"
    LANGUAGES CXX)

option(NANOQUANT_BUILD_TESTS "Build NanoQuant tests" ON)
option(NANOQUANT_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

add_library(nanoquant_core
    src/bitpack.cpp
    src/quantization.cpp
    src/tensor.cpp
    src/workflow.cpp)

target_include_directories(nanoquant_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

target_compile_features(nanoquant_core PUBLIC cxx_std_20)

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

if (NANOQUANT_BUILD_TESTS)
    enable_testing()
    add_executable(nanoquant_tests tests/test_core.cpp)
    target_link_libraries(nanoquant_tests PRIVATE nanoquant_core)
    add_test(NAME nanoquant_tests COMMAND nanoquant_tests)
endif()
