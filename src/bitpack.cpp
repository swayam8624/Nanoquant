#include "nanoquant/bitpack.hpp"

#include <stdexcept>

namespace nanoquant {

std::vector<std::uint8_t> pack_bits(const std::vector<std::uint8_t>& bits) {
    std::vector<std::uint8_t> packed((bits.size() + 7U) / 8U, 0U);
    for (std::size_t index = 0; index < bits.size(); ++index) {
        if (bits[index] > 1U) {
            throw std::invalid_argument("pack_bits expects values in {0, 1}");
        }
        packed[index / 8U] |= static_cast<std::uint8_t>(bits[index] << (index % 8U));
    }
    return packed;
}

std::vector<std::uint8_t> unpack_bits(const std::vector<std::uint8_t>& packed, std::size_t bit_count) {
    std::vector<std::uint8_t> bits(bit_count, 0U);
    for (std::size_t index = 0; index < bit_count; ++index) {
        bits[index] = static_cast<std::uint8_t>((packed[index / 8U] >> (index % 8U)) & 1U);
    }
    return bits;
}

std::vector<std::uint8_t> pack_nibbles(const std::vector<std::uint8_t>& values) {
    std::vector<std::uint8_t> packed((values.size() + 1U) / 2U, 0U);
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (values[index] > 15U) {
            throw std::invalid_argument("pack_nibbles expects values in [0, 15]");
        }
        if ((index & 1U) == 0U) {
            packed[index / 2U] |= values[index];
        } else {
            packed[index / 2U] |= static_cast<std::uint8_t>(values[index] << 4U);
        }
    }
    return packed;
}

std::vector<std::uint8_t> unpack_nibbles(const std::vector<std::uint8_t>& packed, std::size_t value_count) {
    std::vector<std::uint8_t> values(value_count, 0U);
    for (std::size_t index = 0; index < value_count; ++index) {
        const std::uint8_t byte = packed[index / 2U];
        values[index] = ((index & 1U) == 0U) ? static_cast<std::uint8_t>(byte & 0x0FU)
                                             : static_cast<std::uint8_t>((byte >> 4U) & 0x0FU);
    }
    return values;
}

}  // namespace nanoquant
