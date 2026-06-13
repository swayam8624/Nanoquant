#include "nanoquant/backend.hpp"

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

#if !defined(NANOQUANT_HAS_METAL)
BackendInfo metal_backend_info() {
    return {"metal", false, false};
}

std::string metal_kernel_source() {
    return {};
}
#endif

}  // namespace nanoquant
