#ifndef RUNMAT_MATLAB_DATA_ARRAY_SPARSE_ARRAY_HPP
#define RUNMAT_MATLAB_DATA_ARRAY_SPARSE_ARRAY_HPP

#include "TypedArray.hpp"

#include <utility>

namespace matlab {
namespace data {

using SparseIndex = std::pair<std::size_t, std::size_t>;

template <typename T> class SparseArray : public Array {
public:
    using Traits = detail::ElementTraits<T>;
    using Storage = typename Traits::Storage;
    using iterator = detail::TypedIterator<T, false>;
    using const_iterator = detail::TypedIterator<T, true>;

    SparseArray() = default;
    SparseArray(const Array &array) : Array(array) { validate(); }
    SparseArray(Array &&array) : Array(std::move(array)) { validate(); }

    iterator begin() {
        ensureWritable();
        return iterator(static_cast<Storage *>(mxGetData(native())));
    }
    iterator end() {
        ensureWritable();
        return iterator(static_cast<Storage *>(mxGetData(native())) +
                        getNumberOfNonZeroElements());
    }
    const_iterator begin() const {
        return const_iterator(static_cast<const Storage *>(mxGetData(native())));
    }
    const_iterator end() const {
        return const_iterator(static_cast<const Storage *>(mxGetData(native())) +
                              getNumberOfNonZeroElements());
    }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

    std::size_t getNumberOfNonZeroElements() const {
        const auto dimensions = getDimensions();
        const std::size_t columns = dimensions.size() < 2 ? 1 : dimensions[1];
        const mwIndex *columnPointers = mxGetJc(native());
        return columnPointers == nullptr ? 0 : columnPointers[columns];
    }

    SparseIndex getIndex(const iterator &position) const {
        return indexAt(static_cast<std::size_t>(position.positionFrom(
            static_cast<const Storage *>(mxGetData(native())))));
    }
    SparseIndex getIndex(const const_iterator &position) const {
        return indexAt(static_cast<std::size_t>(position.positionFrom(
            static_cast<const Storage *>(mxGetData(native())))));
    }

private:
    void validate() const {
        const ArrayType type = getType();
        const ArrayType expected =
            std::is_same<T, bool>::value
                ? ArrayType::SPARSE_LOGICAL
                : (std::is_same<T, std::complex<double>>::value
                       ? ArrayType::SPARSE_COMPLEX_DOUBLE
                       : ArrayType::SPARSE_DOUBLE);
        if (type != expected) {
            throw InvalidArrayTypeException("array has the wrong sparse element type");
        }
    }

    SparseIndex indexAt(std::size_t offset) const {
        if (offset >= getNumberOfNonZeroElements()) {
            throw std::out_of_range("sparse iterator");
        }
        const mwIndex *rows = mxGetIr(native());
        const mwIndex *columns = mxGetJc(native());
        const auto dimensions = getDimensions();
        const std::size_t columnCount = dimensions.size() < 2 ? 1 : dimensions[1];
        std::size_t column = 0;
        while (column < columnCount && columns[column + 1] <= offset) ++column;
        return {rows[offset], column};
    }
};

} // namespace data
} // namespace matlab

#endif
