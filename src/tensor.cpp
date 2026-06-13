#include "nanoquant/tensor.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace nanoquant {

Tensor::Tensor(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), values_(rows * cols, 0.0F) {}

Tensor::Tensor(std::size_t rows, std::size_t cols, std::vector<float> values)
    : rows_(rows), cols_(cols), values_(std::move(values)) {
    if (values_.size() != rows_ * cols_) {
        throw std::invalid_argument("tensor value count does not match shape");
    }
}

std::size_t Tensor::rows() const noexcept {
    return rows_;
}

std::size_t Tensor::cols() const noexcept {
    return cols_;
}

std::size_t Tensor::elements() const noexcept {
    return values_.size();
}

std::size_t Tensor::bytes() const noexcept {
    return values_.size() * sizeof(float);
}

float Tensor::at(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("tensor index out of range");
    }
    return values_[row * cols_ + col];
}

void Tensor::set(std::size_t row, std::size_t col, float value) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("tensor index out of range");
    }
    values_[row * cols_ + col] = value;
}

std::span<const float> Tensor::values() const noexcept {
    return values_;
}

std::span<float> Tensor::values() noexcept {
    return values_;
}

Tensor make_deterministic_weights(std::size_t rows, std::size_t cols, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0F, 0.18F);
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);

    Tensor tensor(rows, cols);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const float wave = 0.35F * std::sin(static_cast<float>((row + 1U) * (col + 3U)) * 0.017F);
            const float outlier = (uniform(rng) > 0.985F) ? uniform(rng) * 1.7F : 0.0F;
            tensor.set(row, col, wave + normal(rng) + outlier);
        }
    }
    return tensor;
}

}  // namespace nanoquant
