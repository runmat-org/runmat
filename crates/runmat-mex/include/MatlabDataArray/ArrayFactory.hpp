#ifndef RUNMAT_MATLAB_DATA_ARRAY_ARRAY_FACTORY_HPP
#define RUNMAT_MATLAB_DATA_ARRAY_ARRAY_FACTORY_HPP

#include "TypedArray.hpp"
#include "StringArray.hpp"
#include "SparseArray.hpp"
#include "EnumArray.hpp"
#include "CellArray.hpp"
#include "StructArray.hpp"
#include "mex.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <tuple>
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

    StringArray createScalar(const String &value) const {
        return createArray<MATLABString>({1, 1}, {MATLABString(value)});
    }

    StringArray createScalar(const MATLABString &value) const {
        return createArray<MATLABString>({1, 1}, {value});
    }

    StringArray createScalar(const std::string &value) const {
        return createScalar(detail::decodeUTF8(value));
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
    SparseArray<T> createSparseArray(
        ArrayDimensions dimensions, std::size_t nonzeroCount,
        buffer_ptr_t<T> data, buffer_ptr_t<std::size_t> rows,
        buffer_ptr_t<std::size_t> columns) const {
        static_assert(std::is_same<T, double>::value ||
                          std::is_same<T, bool>::value,
                      "this compatibility level supports real double and logical sparse arrays");
        if (dimensions.size() != 2) {
            throw std::invalid_argument("sparse arrays must have two dimensions");
        }
        if (data.get_deleter().elements < nonzeroCount ||
            rows.get_deleter().elements < nonzeroCount ||
            columns.get_deleter().elements < nonzeroCount) {
            throw std::invalid_argument("sparse coordinate buffer is too small");
        }

        std::vector<std::size_t> order(nonzeroCount);
        std::iota(order.begin(), order.end(), 0);
        for (std::size_t index = 0; index < nonzeroCount; ++index) {
            if (rows.get()[index] >= dimensions[0] ||
                columns.get()[index] >= dimensions[1]) {
                throw std::out_of_range("sparse coordinate exceeds dimensions");
            }
        }
        const auto less = [&](std::size_t left, std::size_t right) {
            return std::tie(columns.get()[left], rows.get()[left]) <
                   std::tie(columns.get()[right], rows.get()[right]);
        };
        const bool ordered = std::is_sorted(order.begin(), order.end(), less);
        if (!ordered) {
            std::stable_sort(order.begin(), order.end(), less);
            auto sortedData = createBuffer<T>(nonzeroCount);
            auto sortedRows = createBuffer<std::size_t>(nonzeroCount);
            for (std::size_t destination = 0; destination < nonzeroCount;
                 ++destination) {
                sortedData.get()[destination] = data.get()[order[destination]];
                sortedRows.get()[destination] = rows.get()[order[destination]];
            }
            runmatDataArrayRecordSparseLayoutCopy(
                nonzeroCount * (sizeof(T) + sizeof(std::size_t)));
            data = std::move(sortedData);
            rows = std::move(sortedRows);
        }

        auto columnPointers = createBuffer<std::size_t>(dimensions[1] + 1);
        std::fill(columnPointers.get(), columnPointers.get() + dimensions[1] + 1,
                  0);
        for (std::size_t destination = 0; destination < nonzeroCount;
             ++destination) {
            const std::size_t source = ordered ? destination : order[destination];
            ++columnPointers.get()[columns.get()[source] + 1];
        }
        for (std::size_t column = 0; column < dimensions[1]; ++column) {
            columnPointers.get()[column + 1] += columnPointers.get()[column];
        }
        runmatDataArrayRecordSparseLayoutCopy(nonzeroCount * sizeof(std::size_t));

        mxArray *native = std::is_same<T, bool>::value
                              ? mxCreateSparseLogicalMatrix(
                                    dimensions[0], dimensions[1], nonzeroCount)
                              : mxCreateSparse(dimensions[0], dimensions[1],
                                               nonzeroCount, mxREAL);
        if (native == nullptr) throw std::bad_alloc();
        mxSetData(native, data.get());
        (void)data.release();
        mxSetIr(native, rows.get());
        (void)rows.release();
        mxSetJc(native, columnPointers.get());
        (void)columnPointers.release();
        return SparseArray<T>(Array::adopt(native));
    }

    EnumArray createEnumArray(ArrayDimensions dimensions,
                              const std::string &className,
                              const std::vector<std::string> &members) const {
        if (className.empty()) {
            throw std::invalid_argument("enumeration class name is empty");
        }
        const std::size_t count = elementCount(dimensions);
        if (members.size() != count) {
            throw std::invalid_argument(
                "enumeration member count does not match dimensions");
        }
        const char *property = "__enum_member__";
        mxArray *native = mxCreateStructArray(dimensions.size(), dimensions.data(),
                                              1, &property);
        if (native == nullptr) throw std::bad_alloc();
        if (mxSetClassName(native, className.c_str()) != 0) {
            mxDestroyArray(native);
            throw std::invalid_argument("invalid enumeration class name");
        }
        for (std::size_t index = 0; index < count; ++index) {
            auto member = createScalar(detail::decodeUTF8(members[index]));
            mxSetProperty(native, index, property,
                          detail::ArrayAccess::releaseForOutput(member));
        }
        return EnumArray(Array::adopt(native));
    }

    EnumArray createEnumArray(ArrayDimensions dimensions,
                              const std::string &className) const {
        const std::size_t count = elementCount(dimensions);
        return createEnumArray(std::move(dimensions), className,
                               std::vector<std::string>(count, std::string()));
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
        if constexpr (!std::is_same<T, bool>::value &&
                      !std::is_same<T, std::size_t>::value) {
            if constexpr (detail::ElementTraits<T>::complex) {
                using Storage = typename detail::ElementTraits<T>::Storage;
                static_assert(
                    std::is_trivially_destructible<T>::value,
                    "complex buffer elements must have trivial destruction");
                static_assert(
                    sizeof(T) == sizeof(Storage),
                    "complex buffer element size does not match the host ABI");
                static_assert(
                    alignof(T) == alignof(Storage),
                    "complex buffer element alignment does not match the host ABI");
                for (std::size_t index = 0; index < elements; ++index) {
                    ::new (static_cast<void *>(pointer + index)) T();
                }
            }
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
    Array toArray(const String &value) const { return createScalar(value); }
    Array toArray(const MATLABString &value) const { return createScalar(value); }
    Array toArray(const std::string &value) const { return createScalar(value); }
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
        if constexpr (std::is_same<T, bool>::value ||
                      std::is_same<T, std::size_t>::value) {
            return true;
        } else {
            using Traits = detail::ElementTraits<T>;
            if constexpr (Traits::complex) {
                return std::is_same<T, std::complex<double>>::value ||
                       std::is_same<T, std::complex<float>>::value;
            } else {
                return std::is_same<typename Traits::Storage, T>::value;
            }
        }
    }

    template <typename T>
    static mxArray *createUninitialized(const ArrayDimensions &dimensions) {
        const std::size_t count = elementCount(dimensions);
        (void)count;
        mxArray *array = nullptr;
        if constexpr (std::is_same<T, MATLABString>::value) {
            array = runmatDataArrayCreateStringArray(dimensions.size(),
                                                     dimensions.data());
        } else if constexpr (std::is_same<T, bool>::value) {
            array = mxCreateLogicalArray(dimensions.size(), dimensions.data());
        } else if constexpr (std::is_same<T, char16_t>::value) {
            array = mxCreateCharArray(dimensions.size(), dimensions.data());
        } else {
            const mxComplexity complexity =
                detail::ElementTraits<T>::complex ? mxCOMPLEX : mxREAL;
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
