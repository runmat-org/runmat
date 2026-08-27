#ifndef RUNMAT_MATLAB_DATA_ARRAY_ARRAY_FACTORY_HPP
#define RUNMAT_MATLAB_DATA_ARRAY_ARRAY_FACTORY_HPP

#include "TypedArray.hpp"
#include "CellArray.hpp"
#include "StructArray.hpp"
#include "mex.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>

namespace matlab {
namespace data {

enum class InputLayout { COLUMN_MAJOR, ROW_MAJOR };

struct buffer_deleter_t {
    std::size_t elements = 0;
    void operator()(void *pointer) const noexcept { mxFree(pointer); }
};

template <typename T>
using buffer_ptr_t = std::unique_ptr<T, buffer_deleter_t>;

class ArrayFactory {
public:
    ArrayFactory() = default;

    template <typename T>
    TypedArray<T> createArray(ArrayDimensions dimensions) const {
        return TypedArray<T>(Array::adopt(createUninitialized<T>(dimensions)));
    }

    template <typename T>
    TypedArray<T> createArray(ArrayDimensions dimensions,
                              std::initializer_list<T> values) const {
        return createArray<typename std::initializer_list<T>::const_iterator, T>(
            std::move(dimensions), values.begin(), values.end(),
            InputLayout::COLUMN_MAJOR);
    }

    template <typename Iterator,
              typename T = typename std::iterator_traits<Iterator>::value_type>
    TypedArray<T> createArray(ArrayDimensions dimensions, Iterator begin,
                              Iterator end,
                              InputLayout layout = InputLayout::COLUMN_MAJOR) const {
        TypedArray<T> result = createArray<T>(dimensions);
        const auto count = static_cast<std::size_t>(std::distance(begin, end));
        if (count != result.getNumberOfElements()) {
            throw std::invalid_argument("array data count does not match dimensions");
        }
        if (layout == InputLayout::COLUMN_MAJOR || dimensions.size() != 2) {
            std::copy(begin, end, result.begin());
            return result;
        }
        const std::size_t rows = dimensions[0];
        const std::size_t columns = dimensions[1];
        std::vector<T> rowMajor(begin, end);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                result[row + column * rows] =
                    rowMajor[row * columns + column];
            }
        }
        return result;
    }

    template <typename T>
    TypedArray<T> createScalar(const T &value) const {
        return createArray<T>({1, 1}, {value});
    }

    CharArray createCharArray(const std::u16string &value) const {
        return createArray({1, value.size()}, value.begin(), value.end());
    }

    CharArray createCharArray(const std::string &value) const {
        std::u16string converted;
        converted.reserve(value.size());
        for (unsigned char byte : value) {
            if (byte > 0x7f) {
                throw std::invalid_argument("non-ASCII input requires UTF-16");
            }
            converted.push_back(static_cast<char16_t>(byte));
        }
        return createCharArray(converted);
    }

    Array createEmptyArray() const {
        return Array::adopt(mxCreateDoubleMatrix(0, 0, mxREAL));
    }

    CellArray createCellArray(ArrayDimensions dimensions) const {
        (void)elementCount(dimensions);
        mxArray *array = mxCreateCellArray(dimensions.size(), dimensions.data());
        if (array == nullptr) throw std::bad_alloc();
        return CellArray(Array::adopt(array));
    }

    template <typename... Values>
    CellArray createCellArray(ArrayDimensions dimensions, Values &&...values) const {
        CellArray result = createCellArray(std::move(dimensions));
        std::vector<Array> converted{
            toArray(std::forward<Values>(values))...};
        if (converted.size() != result.getNumberOfElements()) {
            throw std::invalid_argument("cell data count does not match dimensions");
        }
        for (std::size_t index = 0; index < converted.size(); ++index) {
            result[index] = std::move(converted[index]);
        }
        return result;
    }

    StructArray createStructArray(ArrayDimensions dimensions,
                                  const std::vector<std::string> &fields) const {
        (void)elementCount(dimensions);
        std::vector<const char *> names;
        names.reserve(fields.size());
        for (const auto &field : fields) names.push_back(field.c_str());
        mxArray *array = mxCreateStructArray(
            dimensions.size(), dimensions.data(), static_cast<int>(names.size()),
            names.data());
        if (array == nullptr) throw std::bad_alloc();
        return StructArray(Array::adopt(array));
    }

    template <typename T>
    buffer_ptr_t<T> createBuffer(std::size_t elements) const {
        static_assert(bufferCompatible<T>(),
                      "this element type requires an explicit representation conversion");
        if (elements > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        auto *pointer = static_cast<T *>(mxMalloc(elements * sizeof(T)));
        if (pointer == nullptr && elements != 0) {
            throw std::bad_alloc();
        }
        return buffer_ptr_t<T>(pointer, buffer_deleter_t{elements});
    }

    template <typename T>
    TypedArray<T> createArrayFromBuffer(
        ArrayDimensions dimensions, buffer_ptr_t<T> buffer,
        MemoryLayout layout = MemoryLayout::COLUMN_MAJOR) const {
        static_assert(bufferCompatible<T>(),
                      "this element type requires an explicit representation conversion");
        const std::size_t required = elementCount(dimensions);
        if (buffer.get_deleter().elements < required) {
            throw std::invalid_argument("buffer is smaller than the requested array");
        }
        TypedArray<T> result = createArray<T>(dimensions);
        if (layout == MemoryLayout::ROW_MAJOR) {
            runmatDataArrayRecordMemoryLayoutCopy(required * sizeof(T));
            for (std::size_t rowMajor = 0; rowMajor < required; ++rowMajor) {
                std::size_t remainder = rowMajor;
                std::vector<std::size_t> coordinates(dimensions.size(), 0);
                for (std::size_t axis = dimensions.size(); axis-- > 0;) {
                    coordinates[axis] =
                        dimensions[axis] == 0 ? 0 : remainder % dimensions[axis];
                    if (dimensions[axis] != 0) remainder /= dimensions[axis];
                }
                std::size_t columnMajor = 0;
                std::size_t columnStride = 1;
                for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
                    columnMajor += coordinates[axis] * columnStride;
                    columnStride *= dimensions[axis];
                }
                result[columnMajor] = buffer.get()[rowMajor];
            }
            return result;
        }
        mxSetData(result.native(), buffer.get());
        (void)buffer.release();
        return result;
    }

private:
    Array toArray(Array value) const { return value; }
    Array toArray(const std::string &value) const { return createCharArray(value); }
    Array toArray(const char *value) const {
        return createCharArray(value == nullptr ? std::string() : std::string(value));
    }
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
    Array toArray(T value) const {
        return createScalar<T>(value);
    }

    static std::size_t elementCount(const ArrayDimensions &dimensions) {
        std::size_t count = 1;
        for (std::size_t dimension : dimensions) {
            if (dimension != 0 &&
                count > std::numeric_limits<std::size_t>::max() / dimension) {
                throw std::bad_array_new_length();
            }
            count *= dimension;
        }
        return count;
    }

    template <typename T> static constexpr bool bufferCompatible() {
        using Traits = detail::ElementTraits<T>;
        return !Traits::complex &&
               std::is_same<typename Traits::Storage, T>::value;
    }

    template <typename T>
    static mxArray *createUninitialized(const ArrayDimensions &dimensions) {
        const std::size_t count = elementCount(dimensions);
        (void)count;
        const mxComplexity complexity =
            detail::ElementTraits<T>::complex ? mxCOMPLEX : mxREAL;
        mxArray *array = nullptr;
        if constexpr (std::is_same<T, bool>::value) {
            array = mxCreateLogicalArray(dimensions.size(), dimensions.data());
        } else if constexpr (std::is_same<T, char16_t>::value) {
            array = mxCreateCharArray(dimensions.size(), dimensions.data());
        } else {
            array = mxCreateUninitNumericArray(
                dimensions.size(), dimensions.data(),
                detail::ElementTraits<T>::classId, complexity);
        }
        if (array == nullptr) {
            throw std::bad_alloc();
        }
        return array;
    }
};

} // namespace data
} // namespace matlab

#endif
