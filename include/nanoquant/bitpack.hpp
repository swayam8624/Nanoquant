#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nanoquant {

std::vector<std::uint8_t> pack_bits(const std::vector<std::uint8_t>& bits);
std::vector<std::uint8_t> unpack_bits(const std::vector<std::uint8_t>& packed, std::size_t bit_count);

std::vector<std::uint8_t> pack_nibbles(const std::vector<std::uint8_t>& values);
std::vector<std::uint8_t> unpack_nibbles(const std::vector<std::uint8_t>& packed, std::size_t value_count);

}  // namespace nanoquant
