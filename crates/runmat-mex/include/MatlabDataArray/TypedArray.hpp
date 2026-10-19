#ifndef RUNMAT_MATLAB_DATA_ARRAY_TYPED_ARRAY_HPP
#define RUNMAT_MATLAB_DATA_ARRAY_TYPED_ARRAY_HPP

#include "Array.hpp"

#include <complex>
#include <cstdint>
#include <iterator>
#include <type_traits>

namespace matlab {
namespace data {
namespace detail {

template <typename T> struct ElementTraits;

#define RUNMAT_DATA_TRAIT(cpp_type, storage_type, class_id, array_type)          \
    template <> struct ElementTraits<cpp_type> {                                \
        using Storage = storage_type;                                            \
        static constexpr mxClassID classId = class_id;                           \
        static constexpr ArrayType arrayType = ArrayType::array_type;             \
        static constexpr bool complex = false;                                   \
        static cpp_type get(const Storage &value) { return value; }               \
        static void set(Storage &slot, const cpp_type &value) { slot = value; }   \
    }

RUNMAT_DATA_TRAIT(double, double, mxDOUBLE_CLASS, DOUBLE);
RUNMAT_DATA_TRAIT(float, float, mxSINGLE_CLASS, SINGLE);
RUNMAT_DATA_TRAIT(std::int8_t, std::int8_t, mxINT8_CLASS, INT8);
RUNMAT_DATA_TRAIT(std::uint8_t, std::uint8_t, mxUINT8_CLASS, UINT8);
RUNMAT_DATA_TRAIT(std::int16_t, std::int16_t, mxINT16_CLASS, INT16);
RUNMAT_DATA_TRAIT(std::uint16_t, std::uint16_t, mxUINT16_CLASS, UINT16);
RUNMAT_DATA_TRAIT(std::int32_t, std::int32_t, mxINT32_CLASS, INT32);
RUNMAT_DATA_TRAIT(std::uint32_t, std::uint32_t, mxUINT32_CLASS, UINT32);
RUNMAT_DATA_TRAIT(std::int64_t, std::int64_t, mxINT64_CLASS, INT64);
RUNMAT_DATA_TRAIT(std::uint64_t, std::uint64_t, mxUINT64_CLASS, UINT64);
RUNMAT_DATA_TRAIT(char16_t, mxChar, mxCHAR_CLASS, CHAR);

#undef RUNMAT_DATA_TRAIT

template <> struct ElementTraits<bool> {
    using Storage = mxLogical;
    static constexpr mxClassID classId = mxLOGICAL_CLASS;
    static constexpr ArrayType arrayType = ArrayType::LOGICAL;
    static constexpr bool complex = false;
    static bool get(const Storage &value) { return value != 0; }
    static void set(Storage &slot, bool value) { slot = value ? 1 : 0; }
};

template <> struct ElementTraits<std::complex<double>> {
    using Storage = mxComplexDouble;
    static constexpr mxClassID classId = mxDOUBLE_CLASS;
    static constexpr ArrayType arrayType = ArrayType::COMPLEX_DOUBLE;
    static constexpr bool complex = true;
    static std::complex<double> get(const Storage &value) {
        return {value.real, value.imag};
    }
    static void set(Storage &slot, const std::complex<double> &value) {
        slot.real = value.real();
        slot.imag = value.imag();
    }
};

template <> struct ElementTraits<std::complex<float>> {
    using Storage = mxComplexSingle;
    static constexpr mxClassID classId = mxSINGLE_CLASS;
    static constexpr ArrayType arrayType = ArrayType::COMPLEX_SINGLE;
    static constexpr bool complex = true;
    static std::complex<float> get(const Storage &value) {
        return {value.real, value.imag};
    }
    static void set(Storage &slot, const std::complex<float> &value) {
        slot.real = value.real();
        slot.imag = value.imag();
    }
};

#define RUNMAT_COMPLEX_DATA_TRAIT(cpp_type, storage_type, class_id, array_type) \
    template <> struct ElementTraits<std::complex<cpp_type>> {                  \
        using Storage = storage_type;                                            \
        static constexpr mxClassID classId = class_id;                           \
        static constexpr ArrayType arrayType = ArrayType::array_type;             \
        static constexpr bool complex = true;                                    \
        static std::complex<cpp_type> get(const Storage &value) {                 \
            return {value.real, value.imag};                                      \
        }                                                                         \
        static void set(Storage &slot, const std::complex<cpp_type> &value) {      \
            slot.real = value.real();                                              \
            slot.imag = value.imag();                                              \
        }                                                                         \
    }

RUNMAT_COMPLEX_DATA_TRAIT(std::int8_t, mxComplexInt8, mxINT8_CLASS, COMPLEX_INT8);
RUNMAT_COMPLEX_DATA_TRAIT(std::uint8_t, mxComplexUint8, mxUINT8_CLASS, COMPLEX_UINT8);
RUNMAT_COMPLEX_DATA_TRAIT(std::int16_t, mxComplexInt16, mxINT16_CLASS, COMPLEX_INT16);
RUNMAT_COMPLEX_DATA_TRAIT(std::uint16_t, mxComplexUint16, mxUINT16_CLASS, COMPLEX_UINT16);
RUNMAT_COMPLEX_DATA_TRAIT(std::int32_t, mxComplexInt32, mxINT32_CLASS, COMPLEX_INT32);
RUNMAT_COMPLEX_DATA_TRAIT(std::uint32_t, mxComplexUint32, mxUINT32_CLASS, COMPLEX_UINT32);
RUNMAT_COMPLEX_DATA_TRAIT(std::int64_t, mxComplexInt64, mxINT64_CLASS, COMPLEX_INT64);
RUNMAT_COMPLEX_DATA_TRAIT(std::uint64_t, mxComplexUint64, mxUINT64_CLASS, COMPLEX_UINT64);

#undef RUNMAT_COMPLEX_DATA_TRAIT

template <typename T> class ElementReference {
public:
    using Traits = ElementTraits<T>;
    using Storage = typename Traits::Storage;

    ElementReference() : slot_(nullptr) {}
    explicit ElementReference(Storage *slot) : slot_(slot) {}
    void reset(Storage *slot) { slot_ = slot; }
    operator T() const { return Traits::get(*slot_); }
    ElementReference &operator=(const T &value) {
        Traits::set(*slot_, value);
        return *this;
    }
    ElementReference &operator=(const ElementReference &value) {
        return *this = static_cast<T>(value);
    }

private:
    Storage *slot_;
};

template <typename T, bool Constant> class TypedIterator {
public:
    using Traits = ElementTraits<T>;
    using Storage = typename Traits::Storage;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = void;
    using iterator_category = std::random_access_iterator_tag;
    using reference = typename std::conditional<Constant, T, ElementReference<T> &>::type;

    TypedIterator() : pointer_(nullptr) {}
    explicit TypedIterator(typename std::conditional<Constant, const Storage *, Storage *>::type pointer)
        : pointer_(pointer) {}

    reference operator*() const {
        if constexpr (Constant) {
            return Traits::get(*pointer_);
        } else {
            reference_.reset(pointer_);
            return reference_;
        }
    }
    TypedIterator &operator++() { ++pointer_; return *this; }
    TypedIterator operator++(int) { auto copy = *this; ++*this; return copy; }
    TypedIterator &operator--() { --pointer_; return *this; }
    TypedIterator &operator+=(difference_type amount) { pointer_ += amount; return *this; }
    TypedIterator &operator-=(difference_type amount) { pointer_ -= amount; return *this; }
    TypedIterator operator+(difference_type amount) const { auto copy = *this; return copy += amount; }
    TypedIterator operator-(difference_type amount) const { auto copy = *this; return copy -= amount; }
    difference_type operator-(const TypedIterator &other) const { return pointer_ - other.pointer_; }
    bool operator==(const TypedIterator &other) const { return pointer_ == other.pointer_; }
    bool operator!=(const TypedIterator &other) const { return !(*this == other); }
    bool operator<(const TypedIterator &other) const { return pointer_ < other.pointer_; }
    bool operator>(const TypedIterator &other) const { return other < *this; }
    bool operator<=(const TypedIterator &other) const { return !(other < *this); }
    bool operator>=(const TypedIterator &other) const { return !(*this < other); }
    difference_type positionFrom(const Storage *base) const {
        return pointer_ - base;
    }

private:
    typename std::conditional<Constant, const Storage *, Storage *>::type pointer_;
    mutable ElementReference<T> reference_;
};

} // namespace detail

template <typename T> class TypedArray : public Array {
public:
    using iterator = detail::TypedIterator<T, false>;
    using const_iterator = detail::TypedIterator<T, true>;

    TypedArray() = default;
    TypedArray(const Array &array) : Array(array) { validate(); }
    TypedArray(Array &&array) : Array(std::move(array)) { validate(); }

    iterator begin() { return iterator(mutableData()); }
    iterator end() { return iterator(mutableData() + getNumberOfElements()); }
    const_iterator begin() const { return const_iterator(data()); }
    const_iterator end() const { return const_iterator(data() + getNumberOfElements()); }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

    detail::ElementReference<T> operator[](std::size_t index) {
        if (index >= getNumberOfElements()) throw std::out_of_range("array index");
        return detail::ElementReference<T>(mutableData() + index);
    }
    T operator[](std::size_t index) const {
        if (index >= getNumberOfElements()) throw std::out_of_range("array index");
        return detail::ElementTraits<T>::get(data()[index]);
    }

private:
    using Traits = detail::ElementTraits<T>;
    using Storage = typename Traits::Storage;

    void validate() const {
        if (getType() != Traits::arrayType) {
            throw InvalidArrayTypeException("array element type does not match TypedArray");
        }
    }
    const Storage *data() const {
        return static_cast<const Storage *>(mxGetData(native()));
    }
    Storage *mutableData() {
        ensureWritable();
        return static_cast<Storage *>(mxGetData(native()));
    }
};

using CharArray = TypedArray<char16_t>;

} // namespace data
} // namespace matlab

#endif
