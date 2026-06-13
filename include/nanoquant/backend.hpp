#pragma once

#include "nanoquant/quantization.hpp"
#include "nanoquant/tensor.hpp"

#include <span>
#include <string>
#include <vector>

namespace nanoquant {

struct BackendInfo {
    std::string name;
    bool available = false;
    bool accelerated = false;
};

struct BenchmarkResult {
    std::string backend;
    double dequant_ms = 0.0;
    double matvec_ms = 0.0;
    double total_ms = 0.0;
    bool available = false;
};

BackendInfo cpu_backend_info();
Tensor dequantize_int4_cpu(const Int4Tensor& input);
std::vector<float> matvec_cpu(const Tensor& matrix, std::span<const float> vector);

BackendInfo metal_backend_info();
std::string metal_kernel_source();
Tensor dequantize_int4_metal(const Int4Tensor& input);
std::vector<float> matvec_metal(const Tensor& matrix, std::span<const float> vector);
std::vector<BenchmarkResult> benchmark_backends(const Int4Tensor& quantized,
                                                std::span<const float> vector,
                                                std::size_t iterations);

}  // namespace nanoquant
