#include "nanoquant/gguf.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nanoquant {
namespace {

enum class MetadataType : std::uint32_t {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
};

template <typename T>
T read_scalar(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("unexpected end of GGUF file");
    }
    return value;
}

std::string trim_preview(std::string value, std::size_t preview_limit) {
    if (value.size() <= preview_limit) {
        return value;
    }
    if (preview_limit <= 3U) {
        return value.substr(0, preview_limit);
    }
    return value.substr(0, preview_limit - 3U) + "...";
}

std::string read_gguf_string(std::istream& in, std::size_t preview_limit) {
    const std::uint64_t length = read_scalar<std::uint64_t>(in);
    std::string value(length, '\0');
    if (length > 0U) {
        in.read(value.data(), static_cast<std::streamsize>(length));
        if (!in) {
            throw std::runtime_error("unexpected end of GGUF string");
        }
    }
    return trim_preview(std::move(value), preview_limit);
}

std::string metadata_type_name(std::uint32_t type) {
    switch (static_cast<MetadataType>(type)) {
        case MetadataType::Uint8:
            return "uint8";
        case MetadataType::Int8:
            return "int8";
        case MetadataType::Uint16:
            return "uint16";
        case MetadataType::Int16:
            return "int16";
        case MetadataType::Uint32:
            return "uint32";
        case MetadataType::Int32:
            return "int32";
        case MetadataType::Float32:
            return "float32";
        case MetadataType::Bool:
            return "bool";
        case MetadataType::String:
            return "string";
        case MetadataType::Array:
            return "array";
        case MetadataType::Uint64:
            return "uint64";
        case MetadataType::Int64:
            return "int64";
        case MetadataType::Float64:
            return "float64";
    }
    return "unknown(" + std::to_string(type) + ")";
}

std::string scalar_to_string(std::uint8_t value) {
    return std::to_string(static_cast<unsigned int>(value));
}

std::string scalar_to_string(std::int8_t value) {
    return std::to_string(static_cast<int>(value));
}

template <typename T>
std::string scalar_to_string(T value) {
    std::ostringstream out;
    out << value;
    return out.str();
}

std::string read_metadata_value(std::istream& in, std::uint32_t type, std::size_t preview_limit);

std::string read_array(std::istream& in, std::size_t preview_limit) {
    const std::uint32_t element_type = read_scalar<std::uint32_t>(in);
    const std::uint64_t count = read_scalar<std::uint64_t>(in);
    std::ostringstream out;
    out << metadata_type_name(element_type) << "[" << count << "]";
    const std::uint64_t preview_count = std::min<std::uint64_t>(count, 4U);
    if (preview_count > 0U) {
        out << " {";
        for (std::uint64_t index = 0; index < count; ++index) {
            const std::string value = read_metadata_value(in, element_type, preview_limit);
            if (index < preview_count) {
                if (index > 0U) {
                    out << ", ";
                }
                out << value;
            }
        }
        if (count > preview_count) {
            out << ", ...";
        }
        out << "}";
    }
    return trim_preview(out.str(), preview_limit);
}

std::string read_metadata_value(std::istream& in, std::uint32_t type, std::size_t preview_limit) {
    switch (static_cast<MetadataType>(type)) {
        case MetadataType::Uint8:
            return scalar_to_string(read_scalar<std::uint8_t>(in));
        case MetadataType::Int8:
            return scalar_to_string(read_scalar<std::int8_t>(in));
        case MetadataType::Uint16:
            return scalar_to_string(read_scalar<std::uint16_t>(in));
        case MetadataType::Int16:
            return scalar_to_string(read_scalar<std::int16_t>(in));
        case MetadataType::Uint32:
            return scalar_to_string(read_scalar<std::uint32_t>(in));
        case MetadataType::Int32:
            return scalar_to_string(read_scalar<std::int32_t>(in));
        case MetadataType::Float32:
            return scalar_to_string(read_scalar<float>(in));
        case MetadataType::Bool:
            return read_scalar<std::uint8_t>(in) == 0U ? "false" : "true";
        case MetadataType::String:
            return read_gguf_string(in, preview_limit);
        case MetadataType::Array:
            return read_array(in, preview_limit);
        case MetadataType::Uint64:
            return scalar_to_string(read_scalar<std::uint64_t>(in));
        case MetadataType::Int64:
            return scalar_to_string(read_scalar<std::int64_t>(in));
        case MetadataType::Float64:
            return scalar_to_string(read_scalar<double>(in));
    }
    throw std::runtime_error("unsupported GGUF metadata type: " + std::to_string(type));
}

}  // namespace

std::string gguf_type_name(std::uint32_t type) {
    switch (type) {
        case 0:
            return "F32";
        case 1:
            return "F16";
        case 2:
            return "Q4_0";
        case 3:
            return "Q4_1";
        case 6:
            return "Q5_0";
        case 7:
            return "Q5_1";
        case 8:
            return "Q8_0";
        case 9:
            return "Q8_1";
        case 10:
            return "Q2_K";
        case 11:
            return "Q3_K";
        case 12:
            return "Q4_K";
        case 13:
            return "Q5_K";
        case 14:
            return "Q6_K";
        case 15:
            return "Q8_K";
        case 16:
            return "IQ2_XXS";
        case 17:
            return "IQ2_XS";
        case 18:
            return "IQ3_XXS";
        case 19:
            return "IQ1_S";
        case 20:
            return "IQ4_NL";
        case 21:
            return "IQ3_S";
        case 22:
            return "IQ2_S";
        case 23:
            return "IQ4_XS";
        case 24:
            return "I8";
        case 25:
            return "I16";
        case 26:
            return "I32";
        case 27:
            return "I64";
        case 28:
            return "F64";
        default:
            return "type-" + std::to_string(type);
    }
}

GgufInfo inspect_gguf(const std::filesystem::path& path, std::size_t preview_limit) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open GGUF file: " + path.string());
    }

    std::array<char, 4> magic{};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (magic != std::array<char, 4>{'G', 'G', 'U', 'F'}) {
        throw std::runtime_error("not a GGUF file: " + path.string());
    }

    GgufInfo info;
    info.version = read_scalar<std::uint32_t>(in);
    info.tensor_count = read_scalar<std::uint64_t>(in);
    info.metadata_count = read_scalar<std::uint64_t>(in);

    info.metadata.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(info.metadata_count, 1024U)));
    for (std::uint64_t index = 0; index < info.metadata_count; ++index) {
        GgufMetadataEntry entry;
        entry.key = read_gguf_string(in, preview_limit);
        const std::uint32_t type = read_scalar<std::uint32_t>(in);
        entry.type = metadata_type_name(type);
        entry.value_preview = read_metadata_value(in, type, preview_limit);
        info.metadata.push_back(std::move(entry));
    }

    info.tensors.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(info.tensor_count, 1024U)));
    for (std::uint64_t index = 0; index < info.tensor_count; ++index) {
        GgufTensorInfo tensor;
        tensor.name = read_gguf_string(in, preview_limit);
        const std::uint32_t dimensions = read_scalar<std::uint32_t>(in);
        tensor.dimensions.reserve(dimensions);
        for (std::uint32_t dim = 0; dim < dimensions; ++dim) {
            tensor.dimensions.push_back(read_scalar<std::uint64_t>(in));
        }
        tensor.type = read_scalar<std::uint32_t>(in);
        tensor.offset = read_scalar<std::uint64_t>(in);
        info.tensors.push_back(std::move(tensor));
    }

    return info;
}

}  // namespace nanoquant
