#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace nanoquant {

class Tensor {
public:
    Tensor() = default;
    Tensor(std::size_t rows, std::size_t cols);
    Tensor(std::size_t rows, std::size_t cols, std::vector<float> values);

    [[nodiscard]] std::size_t rows() const noexcept;
    [[nodiscard]] std::size_t cols() const noexcept;
    [[nodiscard]] std::size_t elements() const noexcept;
    [[nodiscard]] std::size_t bytes() const noexcept;

    [[nodiscard]] float at(std::size_t row, std::size_t col) const;
    void set(std::size_t row, std::size_t col, float value);

    [[nodiscard]] std::span<const float> values() const noexcept;
    [[nodiscard]] std::span<float> values() noexcept;

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<float> values_;
};

Tensor make_deterministic_weights(std::size_t rows, std::size_t cols, std::uint64_t seed);

}  // namespace nanoquant
