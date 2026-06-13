#pragma once

#include "nanoquant/tensor.hpp"
#include "nanoquant/quantization.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>

namespace nanoquant {

struct BinaryTensorInfo {
    std::uint32_t version = 1;
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t data_offset = 0;
    std::uint64_t data_bytes = 0;
};

struct Int4TensorFileInfo {
    std::uint32_t version = 1;
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t group_size = 0;
    std::uint64_t scale_count = 0;
    std::uint64_t packed_bytes = 0;
    std::uint64_t data_offset = 0;
};

void save_binary_tensor(const std::filesystem::path& path, const Tensor& tensor);
Tensor load_binary_tensor(const std::filesystem::path& path);
BinaryTensorInfo inspect_binary_tensor(const std::filesystem::path& path);

void save_int4_tensor(const std::filesystem::path& path, const Int4Tensor& tensor);
Int4Tensor load_int4_tensor(const std::filesystem::path& path);
Int4TensorFileInfo inspect_int4_tensor(const std::filesystem::path& path);

class MappedTensor {
public:
    explicit MappedTensor(const std::filesystem::path& path);
    MappedTensor(const MappedTensor&) = delete;
    MappedTensor& operator=(const MappedTensor&) = delete;
    MappedTensor(MappedTensor&& other) noexcept;
    MappedTensor& operator=(MappedTensor&& other) noexcept;
    ~MappedTensor();

    [[nodiscard]] std::size_t rows() const noexcept;
    [[nodiscard]] std::size_t cols() const noexcept;
    [[nodiscard]] std::size_t elements() const noexcept;
    [[nodiscard]] std::span<const float> values() const noexcept;
    [[nodiscard]] Tensor materialize() const;

private:
    void close() noexcept;

    int fd_ = -1;
    void* mapping_ = nullptr;
    std::size_t mapped_bytes_ = 0;
    BinaryTensorInfo info_;
    const float* data_ = nullptr;
};

}  // namespace nanoquant
