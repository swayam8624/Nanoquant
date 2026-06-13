#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace nanoquant {

struct GgufMetadataEntry {
    std::string key;
    std::string type;
    std::string value_preview;
};

struct GgufTensorInfo {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    std::uint32_t type = 0;
    std::uint64_t offset = 0;
};

struct GgufInfo {
    std::uint32_t version = 0;
    std::uint64_t tensor_count = 0;
    std::uint64_t metadata_count = 0;
    std::vector<GgufMetadataEntry> metadata;
    std::vector<GgufTensorInfo> tensors;
};

GgufInfo inspect_gguf(const std::filesystem::path& path, std::size_t preview_limit = 64);
std::string gguf_type_name(std::uint32_t type);

}  // namespace nanoquant
