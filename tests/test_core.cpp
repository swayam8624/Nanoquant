#include "nanoquant/backend.hpp"
#include "nanoquant/binary_tensor.hpp"
#include "nanoquant/bitpack.hpp"
#include "nanoquant/gguf.hpp"
#include "nanoquant/quantization.hpp"
#include "nanoquant/tensor.hpp"
#include "nanoquant/workflow.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
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

void test_comparison_guardrail() {
    const auto healthy = nanoquant::compare_outputs(
        "Quantization reduces model weight precision while preserving useful behavior.",
        "Model quantization reduces weight precision and can preserve useful behavior.");
    assert(!healthy.likely_degraded);
    assert(healthy.lexical_overlap > 0.4);

    const auto degraded = nanoquant::compare_outputs(
        "Quantization reduces model weight precision while preserving useful behavior.",
        "banana");
    assert(degraded.likely_degraded);
}

template <typename T>
void write_scalar(std::ofstream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void write_gguf_string(std::ofstream& out, const std::string& value) {
    write_scalar<std::uint64_t>(out, value.size());
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void test_binary_tensor_file_roundtrip() {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "nanoquant-test.tensor";
    const auto tensor = nanoquant::make_deterministic_weights(4, 5, 11);
    nanoquant::save_binary_tensor(path, tensor);

    const auto info = nanoquant::inspect_binary_tensor(path);
    assert(info.version == 1U);
    assert(info.rows == 4U);
    assert(info.cols == 5U);
    assert(info.data_bytes == tensor.bytes());

    const auto loaded = nanoquant::load_binary_tensor(path);
    assert(loaded.rows() == tensor.rows());
    assert(loaded.cols() == tensor.cols());
    assert(std::fabs(loaded.at(2, 3) - tensor.at(2, 3)) < 1.0e-7F);

    const nanoquant::MappedTensor mapped(path);
    assert(mapped.rows() == tensor.rows());
    assert(mapped.cols() == tensor.cols());
    assert(std::fabs(mapped.values()[7] - tensor.values()[7]) < 1.0e-7F);
    std::filesystem::remove(path);
}

void test_minimal_gguf_inspection() {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "nanoquant-test.gguf";
    std::ofstream out(path, std::ios::binary);
    out.write("GGUF", 4);
    write_scalar<std::uint32_t>(out, 3U);
    write_scalar<std::uint64_t>(out, 1U);
    write_scalar<std::uint64_t>(out, 1U);

    write_gguf_string(out, "general.architecture");
    write_scalar<std::uint32_t>(out, 8U);
    write_gguf_string(out, "test-model");

    write_gguf_string(out, "blk.0.weight");
    write_scalar<std::uint32_t>(out, 2U);
    write_scalar<std::uint64_t>(out, 4U);
    write_scalar<std::uint64_t>(out, 8U);
    write_scalar<std::uint32_t>(out, 0U);
    write_scalar<std::uint64_t>(out, 0U);
    out.close();

    const auto info = nanoquant::inspect_gguf(path);
    assert(info.version == 3U);
    assert(info.metadata_count == 1U);
    assert(info.tensor_count == 1U);
    assert(info.metadata.at(0).key == "general.architecture");
    assert(info.metadata.at(0).value_preview == "test-model");
    assert(info.tensors.at(0).name == "blk.0.weight");
    assert(info.tensors.at(0).dimensions.size() == 2U);
    assert(nanoquant::gguf_type_name(info.tensors.at(0).type) == "F32");
    std::filesystem::remove(path);
}

void test_cpu_matvec() {
    const nanoquant::Tensor matrix(2, 3, {1.0F, 2.0F, 3.0F, -1.0F, 0.5F, 4.0F});
    const std::vector<float> vector{2.0F, 3.0F, 4.0F};
    const auto output = nanoquant::matvec_cpu(matrix, vector);
    assert(output.size() == 2U);
    assert(std::fabs(output[0] - 20.0F) < 1.0e-6F);
    assert(std::fabs(output[1] - 15.5F) < 1.0e-6F);
}

}  // namespace

int main() {
    test_bitpack_roundtrip();
    test_onebit_roundtrip_shape();
    test_int4_error_is_bounded_for_small_values();
    test_sparsity_analysis();
    test_comparison_guardrail();
    test_binary_tensor_file_roundtrip();
    test_minimal_gguf_inspection();
    test_cpu_matvec();
    std::cout << "nanoquant_tests passed\n";
}
