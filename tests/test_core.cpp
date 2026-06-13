#include "nanoquant/bitpack.hpp"
#include "nanoquant/quantization.hpp"
#include "nanoquant/tensor.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

void test_bitpack_roundtrip() {
    const std::vector<std::uint8_t> bits{1, 0, 1, 1, 0, 0, 1, 0, 1};
    const auto packed = nanoquant::pack_bits(bits);
    const auto unpacked = nanoquant::unpack_bits(packed, bits.size());
    assert(unpacked == bits);

    const std::vector<std::uint8_t> nibbles{0, 1, 7, 8, 15, 4, 3};
    const auto packed_nibbles = nanoquant::pack_nibbles(nibbles);
    const auto unpacked_nibbles = nanoquant::unpack_nibbles(packed_nibbles, nibbles.size());
    assert(unpacked_nibbles == nibbles);
}

void test_onebit_roundtrip_shape() {
    nanoquant::Tensor tensor(2, 4, {-1.0F, -3.0F, 2.0F, 4.0F, 1.0F, 3.0F, -2.0F, -4.0F});
    const auto quantized = nanoquant::quantize_onebit_per_row(tensor);
    const auto restored = nanoquant::dequantize(quantized);
    assert(restored.rows() == tensor.rows());
    assert(restored.cols() == tensor.cols());
    assert(std::fabs(restored.at(0, 0) + 2.0F) < 1.0e-6F);
    assert(std::fabs(restored.at(0, 2) - 3.0F) < 1.0e-6F);
}

void test_int4_error_is_bounded_for_small_values() {
    const auto tensor = nanoquant::make_deterministic_weights(16, 16, 7);
    const auto report = nanoquant::report_int4(tensor, 32);
    assert(report.compressed_bytes < report.original_bytes);
    assert(report.error.rmse < 0.20);
}

void test_sparsity_analysis() {
    const nanoquant::Tensor tensor(2, 8, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F,
                                         -1.0F, -2.0F, -3.0F, -4.0F, -5.0F, -6.0F, -7.0F, -8.0F});
    const auto report = nanoquant::analyze_2_to_4_sparsity(tensor);
    assert(report.groups == 4);
    assert(report.kept_values == 8);
    assert(report.dropped_values == 8);
    assert(std::fabs(report.sparsity - 0.5) < 1.0e-12);
}

}  // namespace

int main() {
    test_bitpack_roundtrip();
    test_onebit_roundtrip_shape();
    test_int4_error_is_bounded_for_small_values();
    test_sparsity_analysis();
    std::cout << "nanoquant_tests passed\n";
}
