#ifndef RUNMAT_MATLAB_DATA_ARRAY_STRING_HPP
#define RUNMAT_MATLAB_DATA_ARRAY_STRING_HPP

#include <optional>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace matlab {
namespace data {

using String = std::u16string;
template <typename T> using optional = std::optional<T>;
using MATLABString = optional<String>;

namespace detail {
inline String decodeUTF8(const std::string &value) {
    String result;
    for (std::size_t index = 0; index < value.size();) {
        const auto first = static_cast<unsigned char>(value[index++]);
        std::uint32_t code = 0;
        std::size_t continuationCount = 0;
        if (first <= 0x7f) {
            code = first;
        } else if ((first & 0xe0) == 0xc0) {
            code = first & 0x1f;
            continuationCount = 1;
        } else if ((first & 0xf0) == 0xe0) {
            code = first & 0x0f;
            continuationCount = 2;
        } else if ((first & 0xf8) == 0xf0) {
            code = first & 0x07;
            continuationCount = 3;
        } else {
            throw std::invalid_argument("invalid UTF-8 input");
        }
        if (index + continuationCount > value.size()) {
            throw std::invalid_argument("invalid UTF-8 input");
        }
        for (std::size_t offset = 0; offset < continuationCount; ++offset) {
            const auto continuation = static_cast<unsigned char>(value[index++]);
            if ((continuation & 0xc0) != 0x80) {
                throw std::invalid_argument("invalid UTF-8 input");
            }
            code = (code << 6) | (continuation & 0x3f);
        }
        if (code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff) ||
            (continuationCount == 1 && code < 0x80) ||
            (continuationCount == 2 && code < 0x800) ||
            (continuationCount == 3 && code < 0x10000)) {
            throw std::invalid_argument("invalid UTF-8 input");
        }
        if (code <= 0xffff) {
            result.push_back(static_cast<char16_t>(code));
        } else {
            code -= 0x10000;
            result.push_back(static_cast<char16_t>(0xd800 | (code >> 10)));
            result.push_back(static_cast<char16_t>(0xdc00 | (code & 0x3ff)));
        }
    }
    return result;
}

inline std::string encodeUTF8(const String &value) {
    std::string result;
    for (std::size_t index = 0; index < value.size(); ++index) {
        std::uint32_t code = value[index];
        if (code >= 0xd800 && code <= 0xdbff) {
            if (++index >= value.size()) {
                throw std::invalid_argument("invalid UTF-16 input");
            }
            const std::uint32_t low = value[index];
            if (low < 0xdc00 || low > 0xdfff) {
                throw std::invalid_argument("invalid UTF-16 input");
            }
            code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
        } else if (code >= 0xdc00 && code <= 0xdfff) {
            throw std::invalid_argument("invalid UTF-16 input");
        }
        if (code <= 0x7f) {
            result.push_back(static_cast<char>(code));
        } else if (code <= 0x7ff) {
            result.push_back(static_cast<char>(0xc0 | (code >> 6)));
            result.push_back(static_cast<char>(0x80 | (code & 0x3f)));
        } else if (code <= 0xffff) {
            result.push_back(static_cast<char>(0xe0 | (code >> 12)));
            result.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
            result.push_back(static_cast<char>(0x80 | (code & 0x3f)));
        } else {
            result.push_back(static_cast<char>(0xf0 | (code >> 18)));
            result.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3f)));
            result.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
            result.push_back(static_cast<char>(0x80 | (code & 0x3f)));
        }
    }
    return result;
}
} // namespace detail

} // namespace data
} // namespace matlab

#endif
