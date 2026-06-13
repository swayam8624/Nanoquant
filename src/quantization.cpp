#include "nanoquant/quantization.hpp"

#include "nanoquant/bitpack.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace nanoquant {
namespace {

double safe_ratio(std::size_t original, std::size_t compressed) {
    if (compressed == 0U) {
        return 0.0;
    }
    return static_cast<double>(original) / static_cast<double>(compressed);
}

std::uint8_t encode_int4(float value, float scale) {
    if (scale <= std::numeric_limits<float>::epsilon()) {
        return 8U;
    }
    const auto rounded = static_cast<int>(std::lrint(value / scale));
    const int clamped = std::clamp(rounded, -8, 7);
    return static_cast<std::uint8_t>(clamped + 8);
}

float decode_int4(std::uint8_t value, float scale) {
    return static_cast<float>(static_cast<int>(value) - 8) * scale;
}

}  // namespace

std::size_t OneBitTensor::bytes() const {
    return positive_means.size() * sizeof(float) + negative_means.size() * sizeof(float) + sign_bits.size();
}

std::size_t Int4Tensor::bytes() const {
    return scales.size() * sizeof(float) + packed_values.size();
}

OneBitTensor quantize_onebit_per_row(const Tensor& input) {
    OneBitTensor output;
    output.rows = input.rows();
    output.cols = input.cols();
    output.positive_means.resize(input.rows(), 0.0F);
    output.negative_means.resize(input.rows(), 0.0F);

    std::vector<std::uint8_t> signs;
    signs.reserve(input.elements());

    for (std::size_t row = 0; row < input.rows(); ++row) {
        double positive_sum = 0.0;
        double negative_sum = 0.0;
        std::size_t positive_count = 0;
        std::size_t negative_count = 0;

        for (std::size_t col = 0; col < input.cols(); ++col) {
            const float value = input.at(row, col);
            if (value >= 0.0F) {
                positive_sum += value;
                ++positive_count;
            } else {
                negative_sum += value;
                ++negative_count;
            }
        }

        output.positive_means[row] = positive_count == 0U ? 0.0F : static_cast<float>(positive_sum / positive_count);
        output.negative_means[row] = negative_count == 0U ? 0.0F : static_cast<float>(negative_sum / negative_count);

        for (std::size_t col = 0; col < input.cols(); ++col) {
            signs.push_back(input.at(row, col) >= 0.0F ? 1U : 0U);
        }
    }

    output.sign_bits = pack_bits(signs);
    return output;
}

Tensor dequantize(const OneBitTensor& input) {
    if (input.positive_means.size() != input.rows || input.negative_means.size() != input.rows) {
        throw std::invalid_argument("invalid one-bit tensor metadata");
    }

    Tensor output(input.rows, input.cols);
    const auto signs = unpack_bits(input.sign_bits, input.rows * input.cols);
    for (std::size_t row = 0; row < input.rows; ++row) {
        for (std::size_t col = 0; col < input.cols; ++col) {
            const std::size_t index = row * input.cols + col;
            output.set(row, col, signs[index] == 1U ? input.positive_means[row] : input.negative_means[row]);
        }
    }
    return output;
}

Int4Tensor quantize_int4_symmetric(const Tensor& input, std::size_t group_size) {
    if (group_size == 0U) {
        throw std::invalid_argument("group_size must be non-zero");
    }

    Int4Tensor output;
    output.rows = input.rows();
    output.cols = input.cols();
    output.group_size = group_size;

    const auto values = input.values();
    const std::size_t group_count = (values.size() + group_size - 1U) / group_size;
    output.scales.resize(group_count, 1.0F);

    std::vector<std::uint8_t> encoded;
    encoded.reserve(values.size());

    for (std::size_t group = 0; group < group_count; ++group) {
        const std::size_t begin = group * group_size;
        const std::size_t end = std::min(begin + group_size, values.size());
        float max_abs = 0.0F;
        for (std::size_t index = begin; index < end; ++index) {
            max_abs = std::max(max_abs, std::fabs(values[index]));
        }

        const float scale = max_abs <= std::numeric_limits<float>::epsilon() ? 1.0F : max_abs / 7.0F;
        output.scales[group] = scale;

        for (std::size_t index = begin; index < end; ++index) {
            encoded.push_back(encode_int4(values[index], scale));
        }
    }

    output.packed_values = pack_nibbles(encoded);
    return output;
}

Tensor dequantize(const Int4Tensor& input) {
    const std::size_t value_count = input.rows * input.cols;
    const auto encoded = unpack_nibbles(input.packed_values, value_count);
    Tensor output(input.rows, input.cols);

    for (std::size_t index = 0; index < value_count; ++index) {
        const std::size_t group = index / input.group_size;
        output.values()[index] = decode_int4(encoded[index], input.scales.at(group));
    }
    return output;
}

StructuredSparsityReport analyze_2_to_4_sparsity(const Tensor& input) {
    StructuredSparsityReport report;
    report.rows = input.rows();
    report.cols = input.cols();

    for (std::size_t row = 0; row < input.rows(); ++row) {
        for (std::size_t col = 0; col + 3U < input.cols(); col += 4U) {
            std::array<std::pair<float, std::size_t>, 4> scores{};
            for (std::size_t offset = 0; offset < 4U; ++offset) {
                scores[offset] = {std::fabs(input.at(row, col + offset)), offset};
            }
            std::nth_element(scores.begin(), scores.begin() + 2, scores.end(),
                             [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
            report.kept_values += 2U;
            report.dropped_values += 2U;
            ++report.groups;
        }

        const std::size_t remainder = input.cols() % 4U;
        report.kept_values += remainder;
    }

    const std::size_t total = report.kept_values + report.dropped_values;
    report.sparsity = total == 0U ? 0.0 : static_cast<double>(report.dropped_values) / static_cast<double>(total);
    return report;
}

ErrorStats compare(const Tensor& reference, const Tensor& candidate) {
    if (reference.rows() != candidate.rows() || reference.cols() != candidate.cols()) {
        throw std::invalid_argument("cannot compare tensors with different shapes");
    }

    ErrorStats stats;
    double squared_sum = 0.0;
    double absolute_sum = 0.0;
    for (std::size_t index = 0; index < reference.elements(); ++index) {
        const float delta = reference.values()[index] - candidate.values()[index];
        const float abs_delta = std::fabs(delta);
        squared_sum += static_cast<double>(delta) * static_cast<double>(delta);
        absolute_sum += abs_delta;
        stats.max_absolute_error = std::max(stats.max_absolute_error, abs_delta);
    }

    if (reference.elements() > 0U) {
        stats.rmse = std::sqrt(squared_sum / static_cast<double>(reference.elements()));
        stats.mean_absolute_error = absolute_sum / static_cast<double>(reference.elements());
    }
    return stats;
}

CodecReport report_onebit(const Tensor& input) {
    const OneBitTensor quantized = quantize_onebit_per_row(input);
    const Tensor restored = dequantize(quantized);
    return {"onebit-per-row", input.bytes(), quantized.bytes(), safe_ratio(input.bytes(), quantized.bytes()),
            compare(input, restored)};
}

CodecReport report_int4(const Tensor& input, std::size_t group_size) {
    const Int4Tensor quantized = quantize_int4_symmetric(input, group_size);
    const Tensor restored = dequantize(quantized);
    return {"int4-symmetric", input.bytes(), quantized.bytes(), safe_ratio(input.bytes(), quantized.bytes()),
            compare(input, restored)};
}

}  // namespace nanoquant
