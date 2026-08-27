#ifndef RUNMAT_MATLAB_DATA_ARRAY_CELL_ARRAY_HPP
#define RUNMAT_MATLAB_DATA_ARRAY_CELL_ARRAY_HPP

#include "TypedArray.hpp"

namespace matlab {
namespace data {

class CellReference {
public:
    CellReference(Array *owner, std::size_t index) : owner_(owner), index_(index) {}

    operator Array() const {
        mxArray *value = mxGetCell(detail::ArrayAccess::native(*owner_), index_);
        return value == nullptr ? Array() : Array::borrow(value);
    }

    CellReference &operator=(Array value) {
        detail::ArrayAccess::ensureContainerWritable(*owner_);
        mxArray *native = value ? detail::ArrayAccess::releaseForOutput(value) : nullptr;
        mxSetCell(detail::ArrayAccess::native(*owner_), index_, native);
        return *this;
    }

private:
    Array *owner_;
    std::size_t index_;
};

template <> class TypedArray<Array> : public Array {
public:
    TypedArray() = default;
    TypedArray(const Array &array) : Array(array) { validate(); }
    TypedArray(Array &&array) : Array(std::move(array)) { validate(); }

    CellReference operator[](std::size_t index) {
        if (index >= getNumberOfElements()) throw std::out_of_range("cell index");
        return CellReference(this, index);
    }

    Array operator[](std::size_t index) const {
        if (index >= getNumberOfElements()) throw std::out_of_range("cell index");
        mxArray *value = mxGetCell(detail::ArrayAccess::native(*this), index);
        return value == nullptr ? Array() : Array::borrow(value);
    }

private:
    void validate() const {
        if (getType() != ArrayType::CELL) {
            throw InvalidArrayTypeException("array is not a cell array");
        }
    }
};

using CellArray = TypedArray<Array>;

} // namespace data
} // namespace matlab

#endif
