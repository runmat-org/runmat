#ifndef RUNMAT_MATLAB_DATA_ARRAY_ARRAY_HPP
#define RUNMAT_MATLAB_DATA_ARRAY_ARRAY_HPP

#include "matrix.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

extern "C" RUNMAT_MEX_LOCAL mxArray *runmatDataArrayShare(const mxArray *array);
extern "C" RUNMAT_MEX_LOCAL void
runmatDataArrayRecordMemoryLayoutCopy(std::size_t byteLength);
extern "C" RUNMAT_MEX_LOCAL void
runmatDataArraySetError(const char *identifier, const char *message);

namespace matlab {
namespace data {

using ArrayDimensions = std::vector<std::size_t>;
template <typename T> using Reference = T;
enum class MemoryLayout { COLUMN_MAJOR, ROW_MAJOR };

inline std::size_t getNumElements(const ArrayDimensions &dimensions) {
    std::size_t result = 1;
    for (std::size_t dimension : dimensions) result *= dimension;
    return result;
}

enum class ArrayType {
    UNKNOWN,
    LOGICAL,
    CHAR,
    DOUBLE,
    SINGLE,
    INT8,
    UINT8,
    INT16,
    UINT16,
    INT32,
    UINT32,
    INT64,
    UINT64,
    COMPLEX_DOUBLE,
    COMPLEX_SINGLE,
    COMPLEX_INT8,
    COMPLEX_UINT8,
    COMPLEX_INT16,
    COMPLEX_UINT16,
    COMPLEX_INT32,
    COMPLEX_UINT32,
    COMPLEX_INT64,
    COMPLEX_UINT64,
    CELL,
    STRUCT,
    VALUE_OBJECT,
    HANDLE_OBJECT_REF,
    ENUM,
    SPARSE_LOGICAL,
    SPARSE_DOUBLE,
    SPARSE_COMPLEX_DOUBLE,
    MATLAB_STRING
};

class InvalidArrayTypeException : public std::runtime_error {
public:
    explicit InvalidArrayTypeException(const char *message)
        : std::runtime_error(message) {}
};

namespace detail {
struct ArrayAccess;
struct ArrayControl {
    mxArray *value;
    bool owned;

    ArrayControl(mxArray *value, bool owned) : value(value), owned(owned) {}
    ~ArrayControl() {
        if (owned && value != nullptr) {
            mxDestroyArray(value);
        }
    }
};
} // namespace detail

class Array {
public:
    Array() = default;
    Array(const Array &) = default;
    Array(Array &&) noexcept = default;
    Array &operator=(const Array &) = default;
    Array &operator=(Array &&) noexcept = default;
    virtual ~Array() = default;

    ArrayType getType() const {
        requireValue();
        const bool complex = mxIsComplex(control_->value) != 0;
        if (mxIsSparse(control_->value)) {
            if (mxIsLogical(control_->value)) return ArrayType::SPARSE_LOGICAL;
            return complex ? ArrayType::SPARSE_COMPLEX_DOUBLE
                           : ArrayType::SPARSE_DOUBLE;
        }
        switch (mxGetClassID(control_->value)) {
        case mxLOGICAL_CLASS: return ArrayType::LOGICAL;
        case mxCHAR_CLASS: return ArrayType::CHAR;
        case mxDOUBLE_CLASS:
            return complex ? ArrayType::COMPLEX_DOUBLE : ArrayType::DOUBLE;
        case mxSINGLE_CLASS:
            return complex ? ArrayType::COMPLEX_SINGLE : ArrayType::SINGLE;
        case mxINT8_CLASS: return complex ? ArrayType::COMPLEX_INT8 : ArrayType::INT8;
        case mxUINT8_CLASS: return complex ? ArrayType::COMPLEX_UINT8 : ArrayType::UINT8;
        case mxINT16_CLASS: return complex ? ArrayType::COMPLEX_INT16 : ArrayType::INT16;
        case mxUINT16_CLASS: return complex ? ArrayType::COMPLEX_UINT16 : ArrayType::UINT16;
        case mxINT32_CLASS: return complex ? ArrayType::COMPLEX_INT32 : ArrayType::INT32;
        case mxUINT32_CLASS: return complex ? ArrayType::COMPLEX_UINT32 : ArrayType::UINT32;
        case mxINT64_CLASS: return complex ? ArrayType::COMPLEX_INT64 : ArrayType::INT64;
        case mxUINT64_CLASS: return complex ? ArrayType::COMPLEX_UINT64 : ArrayType::UINT64;
        case mxCELL_CLASS: return ArrayType::CELL;
        case mxSTRUCT_CLASS: return ArrayType::STRUCT;
        default: return ArrayType::UNKNOWN;
        }
    }

    ArrayDimensions getDimensions() const {
        requireValue();
        const auto rank = mxGetNumberOfDimensions(control_->value);
        const auto *dimensions = mxGetDimensions(control_->value);
        return ArrayDimensions(dimensions, dimensions + rank);
    }

    std::size_t getNumberOfElements() const {
        requireValue();
        return mxGetNumberOfElements(control_->value);
    }

    bool isEmpty() const { return getNumberOfElements() == 0; }
    MemoryLayout getMemoryLayout() const { return MemoryLayout::COLUMN_MAJOR; }

    explicit operator bool() const noexcept {
        return control_ && control_->value != nullptr;
    }

    static Array borrow(const mxArray *value) {
        return Array(std::make_shared<detail::ArrayControl>(
            const_cast<mxArray *>(value), false));
    }

    static Array adopt(mxArray *value) {
        return Array(std::make_shared<detail::ArrayControl>(value, true));
    }

    mxArray *releaseForOutput() {
        requireValue();
        if (control_.use_count() == 1 && control_->owned) {
            control_->owned = false;
            return control_->value;
        }
        return runmatDataArrayShare(control_->value);
    }

protected:
    explicit Array(std::shared_ptr<detail::ArrayControl> control)
        : control_(std::move(control)) {}

    mxArray *native() const {
        requireValue();
        return control_->value;
    }

    void ensureWritable() {
        requireValue();
        if (control_.use_count() == 1 && control_->owned) {
            return;
        }
        mxArray *copy = mxDuplicateArray(control_->value);
        if (copy == nullptr) {
            throw std::bad_alloc();
        }
        control_ = std::make_shared<detail::ArrayControl>(copy, true);
    }

private:
    void requireValue() const {
        if (!control_ || control_->value == nullptr) {
            throw InvalidArrayTypeException("array is empty");
        }
    }

    std::shared_ptr<detail::ArrayControl> control_;

    template <typename T> friend class TypedArray;
    friend class ArrayFactory;
    friend struct detail::ArrayAccess;
};

namespace detail {
struct ArrayAccess {
    static mxArray *native(const Array &array) { return array.native(); }
    static void ensureWritable(Array &array) { array.ensureWritable(); }
    static mxArray *releaseForOutput(Array &array) {
        return array.releaseForOutput();
    }
};
} // namespace detail

} // namespace data
} // namespace matlab

#endif
