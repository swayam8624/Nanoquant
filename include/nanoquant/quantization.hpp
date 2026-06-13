#pragma once

#include "nanoquant/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nanoquant {

struct ErrorStats {
    double rmse = 0.0;
    double mean_absolute_error = 0.0;
    float max_absolute_error = 0.0F;
};

struct CodecReport {
    std::string name;
    std::size_t original_bytes = 0;
    std::size_t compressed_bytes = 0;
    double compression_ratio = 0.0;
    ErrorStats error;
};

struct OneBitTensor {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<float> positive_means;
    std::vector<float> negative_means;
    std::vector<std::uint8_t> sign_bits;

    [[nodiscard]] std::size_t bytes() const;
};

struct Int4Tensor {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t group_size = 32;
    std::vector<float> scales;
    std::vector<std::uint8_t> packed_values;

    [[nodiscard]] std::size_t bytes() const;
};

struct StructuredSparsityReport {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t groups = 0;
    std::size_t kept_values = 0;
    std::size_t dropped_values = 0;
    double sparsity = 0.0;
};

OneBitTensor quantize_onebit_per_row(const Tensor& input);
Tensor dequantize(const OneBitTensor& input);

Int4Tensor quantize_int4_symmetric(const Tensor& input, std::size_t group_size = 32);
Tensor dequantize(const Int4Tensor& input);

StructuredSparsityReport analyze_2_to_4_sparsity(const Tensor& input);

ErrorStats compare(const Tensor& reference, const Tensor& candidate);
CodecReport report_onebit(const Tensor& input);
CodecReport report_int4(const Tensor& input, std::size_t group_size = 32);

}  // namespace nanoquant
