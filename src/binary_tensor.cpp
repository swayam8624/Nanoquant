#include "nanoquant/binary_tensor.hpp"

#include <array>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace nanoquant {
namespace {

constexpr std::array<char, 8> kMagic = {'N', 'Q', 'T', 'N', 'S', 'R', '0', '1'};
constexpr std::array<char, 8> kInt4Magic = {'N', 'Q', 'I', 'N', 'T', '4', '0', '1'};
constexpr std::uint64_t kHeaderBytes = 8 + 4 + 8 + 8;
constexpr std::uint64_t kInt4HeaderBytes = 8 + 4 + 8 + 8 + 8 + 8 + 8;

template <typename T>
void write_scalar(std::ostream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T read_scalar(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("unexpected end of tensor file");
    }
    return value;
}

BinaryTensorInfo read_header(std::istream& in) {
    std::array<char, 8> magic{};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (magic != kMagic) {
        throw std::runtime_error("not a NanoQuant binary tensor");
    }

    BinaryTensorInfo info;
    info.version = read_scalar<std::uint32_t>(in);
    info.rows = read_scalar<std::uint64_t>(in);
    info.cols = read_scalar<std::uint64_t>(in);
    info.data_offset = kHeaderBytes;
    info.data_bytes = info.rows * info.cols * sizeof(float);
    if (info.version != 1U) {
        throw std::runtime_error("unsupported NanoQuant tensor version");
    }
    if (info.rows == 0U || info.cols == 0U) {
        throw std::runtime_error("tensor shape must be non-zero");
    }
    return info;
}

Int4TensorFileInfo read_int4_header(std::istream& in) {
    std::array<char, 8> magic{};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (magic != kInt4Magic) {
        throw std::runtime_error("not a NanoQuant int4 tensor");
    }

    Int4TensorFileInfo info;
    info.version = read_scalar<std::uint32_t>(in);
    info.rows = read_scalar<std::uint64_t>(in);
    info.cols = read_scalar<std::uint64_t>(in);
    info.group_size = read_scalar<std::uint64_t>(in);
    info.scale_count = read_scalar<std::uint64_t>(in);
    info.packed_bytes = read_scalar<std::uint64_t>(in);
    info.data_offset = kInt4HeaderBytes;
    if (info.version != 1U) {
        throw std::runtime_error("unsupported NanoQuant int4 tensor version");
    }
    if (info.rows == 0U || info.cols == 0U || info.group_size == 0U) {
        throw std::runtime_error("int4 tensor shape and group size must be non-zero");
    }
    return info;
}

}  // namespace

void save_binary_tensor(const std::filesystem::path& path, const Tensor& tensor) {
    std::filesystem::create_directories(path.parent_path().empty() ? "." : path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open tensor for writing: " + path.string());
    }

    out.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    write_scalar<std::uint32_t>(out, 1U);
    write_scalar<std::uint64_t>(out, tensor.rows());
    write_scalar<std::uint64_t>(out, tensor.cols());
    out.write(reinterpret_cast<const char*>(tensor.values().data()), static_cast<std::streamsize>(tensor.bytes()));
}

Tensor load_binary_tensor(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open tensor: " + path.string());
    }
    const BinaryTensorInfo info = read_header(in);
    std::vector<float> values(info.rows * info.cols);
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(info.data_bytes));
    if (!in) {
        throw std::runtime_error("tensor data is truncated: " + path.string());
    }
    return Tensor(static_cast<std::size_t>(info.rows), static_cast<std::size_t>(info.cols), std::move(values));
}

BinaryTensorInfo inspect_binary_tensor(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open tensor: " + path.string());
    }
    BinaryTensorInfo info = read_header(in);
    const auto file_bytes = std::filesystem::file_size(path);
    if (file_bytes < info.data_offset + info.data_bytes) {
        throw std::runtime_error("tensor data is truncated: " + path.string());
    }
    return info;
}

void save_int4_tensor(const std::filesystem::path& path, const Int4Tensor& tensor) {
    std::filesystem::create_directories(path.parent_path().empty() ? "." : path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open int4 tensor for writing: " + path.string());
    }

    out.write(kInt4Magic.data(), static_cast<std::streamsize>(kInt4Magic.size()));
    write_scalar<std::uint32_t>(out, 1U);
    write_scalar<std::uint64_t>(out, tensor.rows);
    write_scalar<std::uint64_t>(out, tensor.cols);
    write_scalar<std::uint64_t>(out, tensor.group_size);
    write_scalar<std::uint64_t>(out, tensor.scales.size());
    write_scalar<std::uint64_t>(out, tensor.packed_values.size());
    out.write(reinterpret_cast<const char*>(tensor.scales.data()),
              static_cast<std::streamsize>(tensor.scales.size() * sizeof(float)));
    out.write(reinterpret_cast<const char*>(tensor.packed_values.data()),
              static_cast<std::streamsize>(tensor.packed_values.size()));
}

Int4Tensor load_int4_tensor(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open int4 tensor: " + path.string());
    }
    const Int4TensorFileInfo info = read_int4_header(in);

    Int4Tensor tensor;
    tensor.rows = static_cast<std::size_t>(info.rows);
    tensor.cols = static_cast<std::size_t>(info.cols);
    tensor.group_size = static_cast<std::size_t>(info.group_size);
    tensor.scales.resize(static_cast<std::size_t>(info.scale_count));
    tensor.packed_values.resize(static_cast<std::size_t>(info.packed_bytes));
    in.read(reinterpret_cast<char*>(tensor.scales.data()),
            static_cast<std::streamsize>(tensor.scales.size() * sizeof(float)));
    in.read(reinterpret_cast<char*>(tensor.packed_values.data()),
            static_cast<std::streamsize>(tensor.packed_values.size()));
    if (!in) {
        throw std::runtime_error("int4 tensor data is truncated: " + path.string());
    }
    return tensor;
}

Int4TensorFileInfo inspect_int4_tensor(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open int4 tensor: " + path.string());
    }
    Int4TensorFileInfo info = read_int4_header(in);
    const auto file_bytes = std::filesystem::file_size(path);
    const std::uint64_t expected = info.data_offset + info.scale_count * sizeof(float) + info.packed_bytes;
    if (file_bytes < expected) {
        throw std::runtime_error("int4 tensor data is truncated: " + path.string());
    }
    return info;
}

MappedTensor::MappedTensor(const std::filesystem::path& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("failed to open tensor for mapping: " + path.string());
    }

    struct stat file_stat {};
    if (::fstat(fd_, &file_stat) != 0) {
        close();
        throw std::runtime_error("failed to stat tensor: " + path.string());
    }
    mapped_bytes_ = static_cast<std::size_t>(file_stat.st_size);
    mapping_ = ::mmap(nullptr, mapped_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapping_ == MAP_FAILED) {
        mapping_ = nullptr;
        close();
        throw std::runtime_error("failed to mmap tensor: " + path.string());
    }

    if (mapped_bytes_ < kHeaderBytes) {
        close();
        throw std::runtime_error("mapped tensor is too small");
    }
    std::string header(reinterpret_cast<const char*>(mapping_), kHeaderBytes);
    std::istringstream in(header, std::ios::binary);
    info_ = read_header(in);
    if (mapped_bytes_ < info_.data_offset + info_.data_bytes) {
        close();
        throw std::runtime_error("mapped tensor data is truncated");
    }
    data_ = reinterpret_cast<const float*>(static_cast<const std::byte*>(mapping_) + info_.data_offset);
}

MappedTensor::MappedTensor(MappedTensor&& other) noexcept {
    *this = std::move(other);
}

MappedTensor& MappedTensor::operator=(MappedTensor&& other) noexcept {
    if (this != &other) {
        close();
        fd_ = other.fd_;
        mapping_ = other.mapping_;
        mapped_bytes_ = other.mapped_bytes_;
        info_ = other.info_;
        data_ = other.data_;
        other.fd_ = -1;
        other.mapping_ = nullptr;
        other.mapped_bytes_ = 0;
        other.data_ = nullptr;
    }
    return *this;
}

MappedTensor::~MappedTensor() {
    close();
}

std::size_t MappedTensor::rows() const noexcept {
    return static_cast<std::size_t>(info_.rows);
}

std::size_t MappedTensor::cols() const noexcept {
    return static_cast<std::size_t>(info_.cols);
}

std::size_t MappedTensor::elements() const noexcept {
    return rows() * cols();
}

std::span<const float> MappedTensor::values() const noexcept {
    return {data_, elements()};
}

Tensor MappedTensor::materialize() const {
    return Tensor(rows(), cols(), std::vector<float>(values().begin(), values().end()));
}

void MappedTensor::close() noexcept {
    if (mapping_ != nullptr) {
        ::munmap(mapping_, mapped_bytes_);
        mapping_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

}  // namespace nanoquant
