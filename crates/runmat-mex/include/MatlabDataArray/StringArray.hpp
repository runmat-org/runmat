#ifndef RUNMAT_MATLAB_DATA_ARRAY_STRING_ARRAY_HPP
#define RUNMAT_MATLAB_DATA_ARRAY_STRING_ARRAY_HPP

#include "String.hpp"
#include "TypedArray.hpp"

#include <iterator>
#include <limits>

namespace matlab {
namespace data {

namespace detail {
inline MATLABString readString(const Array &array, std::size_t index) {
    const mxArray *native = ArrayAccess::native(array);
    const std::size_t length = runmatDataArrayStringLength(native, index);
    if (length == std::numeric_limits<std::size_t>::max()) {
        throw InvalidArrayTypeException("could not read string array element");
    }
    String value(length, u'\0');
    if (runmatDataArrayCopyString(native, index,
                                  reinterpret_cast<unsigned short *>(value.data()),
                                  length) != 0) {
        throw InvalidArrayTypeException("could not read string array element");
    }
    if (value == u"<missing>") return MATLABString();
    return MATLABString(std::move(value));
}

inline void writeString(Array &array, std::size_t index,
                        const MATLABString &value) {
    static const String missing = u"<missing>";
    const String &stored = value ? *value : missing;
    ArrayAccess::ensureWritable(array);
    if (runmatDataArraySetString(
            ArrayAccess::native(array), index,
            reinterpret_cast<const unsigned short *>(stored.data()),
            stored.size()) != 0) {
        throw InvalidArrayTypeException("could not write string array element");
    }
}
} // namespace detail

class StringReference {
public:
    StringReference(Array *owner, std::size_t index)
        : owner_(owner), index_(index) {}
    operator MATLABString() const { return detail::readString(*owner_, index_); }
    StringReference &operator=(const MATLABString &value) {
        detail::writeString(*owner_, index_, value);
        return *this;
    }
    StringReference &operator=(const String &value) {
        return *this = MATLABString(value);
    }

private:
    Array *owner_;
    std::size_t index_;
};

template <bool Constant> class StringIterator {
public:
    using difference_type = std::ptrdiff_t;
    using value_type = MATLABString;
    using pointer = void;
    using iterator_category = std::random_access_iterator_tag;
    using reference = typename std::conditional<Constant, MATLABString,
                                                StringReference>::type;

    StringIterator() : owner_(nullptr), index_(0) {}
    StringIterator(typename std::conditional<Constant, const Array *, Array *>::type owner,
                   std::size_t index)
        : owner_(owner), index_(index) {}
    reference operator*() const {
        if constexpr (Constant) {
            return detail::readString(*owner_, index_);
        } else {
            return StringReference(owner_, index_);
        }
    }
    StringIterator &operator++() { ++index_; return *this; }
    StringIterator operator++(int) { auto copy = *this; ++*this; return copy; }
    StringIterator &operator--() { --index_; return *this; }
    StringIterator &operator+=(difference_type amount) { index_ += amount; return *this; }
    StringIterator &operator-=(difference_type amount) { index_ -= amount; return *this; }
    StringIterator operator+(difference_type amount) const { auto copy = *this; return copy += amount; }
    StringIterator operator-(difference_type amount) const { auto copy = *this; return copy -= amount; }
    reference operator[](difference_type amount) const { return *(*this + amount); }
    difference_type operator-(const StringIterator &other) const {
        return static_cast<difference_type>(index_) -
               static_cast<difference_type>(other.index_);
    }
    bool operator==(const StringIterator &other) const {
        return owner_ == other.owner_ && index_ == other.index_;
    }
    bool operator!=(const StringIterator &other) const { return !(*this == other); }
    bool operator<(const StringIterator &other) const { return index_ < other.index_; }
    bool operator>(const StringIterator &other) const { return other < *this; }
    bool operator<=(const StringIterator &other) const { return !(other < *this); }
    bool operator>=(const StringIterator &other) const { return !(*this < other); }

private:
    typename std::conditional<Constant, const Array *, Array *>::type owner_;
    std::size_t index_;
};

template <> class TypedArray<MATLABString> : public Array {
public:
    using iterator = StringIterator<false>;
    using const_iterator = StringIterator<true>;

    TypedArray() = default;
    TypedArray(const Array &array) : Array(array) { validate(); }
    TypedArray(Array &&array) : Array(std::move(array)) { validate(); }

    iterator begin() { return iterator(this, 0); }
    iterator end() { return iterator(this, getNumberOfElements()); }
    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const { return const_iterator(this, getNumberOfElements()); }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }
    StringReference operator[](std::size_t index) {
        if (index >= getNumberOfElements()) throw std::out_of_range("string index");
        return StringReference(this, index);
    }
    MATLABString operator[](std::size_t index) const {
        if (index >= getNumberOfElements()) throw std::out_of_range("string index");
        return detail::readString(*this, index);
    }

private:
    void validate() const {
        if (getType() != ArrayType::MATLAB_STRING) {
            throw InvalidArrayTypeException("array is not a string array");
        }
    }
};

using StringArray = TypedArray<MATLABString>;

} // namespace data
} // namespace matlab

#endif
