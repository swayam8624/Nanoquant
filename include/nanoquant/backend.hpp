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

BackendInfo cpu_backend_info();
Tensor dequantize_int4_cpu(const Int4Tensor& input);
std::vector<float> matvec_cpu(const Tensor& matrix, std::span<const float> vector);

BackendInfo metal_backend_info();
std::string metal_kernel_source();

}  // namespace nanoquant
