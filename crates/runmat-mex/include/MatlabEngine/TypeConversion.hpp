#ifndef RUNMAT_MATLAB_ENGINE_TYPE_CONVERSION_HPP
#define RUNMAT_MATLAB_ENGINE_TYPE_CONVERSION_HPP

#include "MatlabDataArray.hpp"
#include "MatlabEngine/Exception.hpp"

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace matlab {
namespace engine {
namespace detail {

template <typename T> struct is_std_vector : std::false_type {};
template <typename T, typename Allocator>
struct is_std_vector<std::vector<T, Allocator>> : std::true_type {
    using value_type = T;
};

template <typename T> struct is_typed_array : std::false_type {};
template <typename T>
struct is_typed_array<data::TypedArray<T>> : std::true_type {
    using value_type = T;
};

inline data::Array engineInput(const data::Array &value) { return value; }
inline data::Array engineInput(data::Array &&value) { return std::move(value); }

inline data::Array engineInput(const std::u16string &value) {
    return data::ArrayFactory().createCharArray(value);
}

inline data::Array engineInput(const std::string &value) {
    return data::ArrayFactory().createCharArray(value);
}

inline data::Array engineInput(const char16_t *value) {
    if (value == nullptr) throw TypeConversionException("engine input string is null");
    return engineInput(std::u16string(value));
}

inline data::Array engineInput(const char *value) {
    if (value == nullptr) throw TypeConversionException("engine input string is null");
    return engineInput(std::string(value));
}

template <typename T>
std::enable_if_t<!is_std_vector<std::decay_t<T>>::value &&
                     !std::is_base_of_v<data::Array, std::decay_t<T>> &&
                     !std::is_same_v<std::decay_t<T>, std::string> &&
                     !std::is_same_v<std::decay_t<T>, std::u16string>,
                 data::Array>
engineInput(T &&value) {
    return data::ArrayFactory().createScalar<std::decay_t<T>>(
        std::forward<T>(value));
}

template <typename T, typename Allocator>
data::Array engineInput(const std::vector<T, Allocator> &values) {
    return data::ArrayFactory().createArray(
        {1, values.size()}, values.begin(), values.end());
}

template <typename... Inputs>
std::vector<data::Array> engineInputs(Inputs &&...inputs) {
    std::vector<data::Array> result;
    result.reserve(sizeof...(Inputs));
    (result.emplace_back(engineInput(std::forward<Inputs>(inputs))), ...);
    return result;
}

template <typename T> T engineOutput(data::Array value) {
    using Result = std::decay_t<T>;
    if constexpr (std::is_same_v<Result, data::Array>) {
        return value;
    } else if constexpr (is_typed_array<Result>::value) {
        return Result(std::move(value));
    } else if constexpr (std::is_same_v<Result, std::u16string>) {
        const data::CharArray characters(std::move(value));
        return std::u16string(characters.begin(), characters.end());
    } else if constexpr (std::is_same_v<Result, std::string>) {
        const auto text = engineOutput<std::u16string>(std::move(value));
        try {
            return data::detail::encodeUTF8(text);
        } catch (const std::invalid_argument &error) {
            throw TypeConversionException(error.what());
        }
    } else if constexpr (is_std_vector<Result>::value) {
        using Element = typename is_std_vector<Result>::value_type;
        const data::TypedArray<Element> array(std::move(value));
        return Result(array.begin(), array.end());
    } else {
        const data::TypedArray<Result> array(std::move(value));
        if (array.getNumberOfElements() != 1) {
            throw TypeConversionException(
                "engine result cannot be converted to a scalar");
        }
        return array[0];
    }
}

} // namespace detail
} // namespace engine
} // namespace matlab

#endif
