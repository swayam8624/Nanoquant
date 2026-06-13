#include "nanoquant/backend.hpp"

#include <chrono>
#include <stdexcept>
#include <vector>

namespace nanoquant {

BackendInfo cpu_backend_info() {
    return {"cpu", true, false};
}

Tensor dequantize_int4_cpu(const Int4Tensor& input) {
    return dequantize(input);
}

std::vector<float> matvec_cpu(const Tensor& matrix, std::span<const float> vector) {
    if (matrix.cols() != vector.size()) {
        throw std::invalid_argument("matvec dimension mismatch");
    }

    std::vector<float> output(matrix.rows(), 0.0F);
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        float sum = 0.0F;
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            sum += matrix.at(row, col) * vector[col];
        }
        output[row] = sum;
    }
    return output;
}

std::vector<BenchmarkResult> benchmark_backends(const Int4Tensor& quantized,
                                                std::span<const float> vector,
                                                std::size_t iterations) {
    if (iterations == 0U) {
        throw std::invalid_argument("benchmark iterations must be non-zero");
    }

    std::vector<BenchmarkResult> results;
    auto run_backend = [&](const std::string& name, auto dequant_fn, auto matvec_fn, bool available) {
        BenchmarkResult result;
        result.backend = name;
        result.available = available;
        if (!available) {
            results.push_back(result);
            return;
        }

        double dequant_ms = 0.0;
        double matvec_ms = 0.0;
        for (std::size_t index = 0; index < iterations; ++index) {
            const auto dequant_begin = std::chrono::steady_clock::now();
            Tensor dense = dequant_fn(quantized);
            const auto dequant_end = std::chrono::steady_clock::now();
            const auto matvec_begin = std::chrono::steady_clock::now();
            volatile float sink = matvec_fn(dense, vector).front();
            (void)sink;
            const auto matvec_end = std::chrono::steady_clock::now();
            dequant_ms += std::chrono::duration<double, std::milli>(dequant_end - dequant_begin).count();
            matvec_ms += std::chrono::duration<double, std::milli>(matvec_end - matvec_begin).count();
        }
        result.dequant_ms = dequant_ms / static_cast<double>(iterations);
        result.matvec_ms = matvec_ms / static_cast<double>(iterations);
        result.total_ms = result.dequant_ms + result.matvec_ms;
        results.push_back(result);
    };

    run_backend("cpu", dequantize_int4_cpu, matvec_cpu, true);
    const BackendInfo metal = metal_backend_info();
    run_backend("metal", dequantize_int4_metal, matvec_metal, metal.available);
    return results;
}

#if !defined(NANOQUANT_HAS_METAL)
BackendInfo metal_backend_info() {
    return {"metal", false, false};
}

std::string metal_kernel_source() {
    return {};
}

Tensor dequantize_int4_metal(const Int4Tensor&) {
    throw std::runtime_error("Metal backend is not compiled");
}

std::vector<float> matvec_metal(const Tensor&, std::span<const float>) {
    throw std::runtime_error("Metal backend is not compiled");
}
#endif

}  // namespace nanoquant
