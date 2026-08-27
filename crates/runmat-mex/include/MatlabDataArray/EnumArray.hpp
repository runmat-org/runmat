#ifndef RUNMAT_MATLAB_DATA_ARRAY_ENUM_ARRAY_HPP
#define RUNMAT_MATLAB_DATA_ARRAY_ENUM_ARRAY_HPP

#include "StringArray.hpp"

#include <string>

namespace matlab {
namespace data {

class Enumeration {
public:
    Enumeration() = default;
    explicit Enumeration(std::string name) : name_(std::move(name)) {}
    operator std::string() const { return name_; }
    const std::string &name() const noexcept { return name_; }

private:
    std::string name_;
};

namespace detail {
inline std::string enumName(const Array &array, std::size_t index) {
    mxArray *member = mxGetProperty(ArrayAccess::native(array), index,
                                    "__enum_member__");
    if (member == nullptr) {
        throw InvalidArrayTypeException("enumeration element has no member name");
    }
    const std::size_t length = runmatDataArrayStringLength(member, 0);
    String text(length, u'\0');
    if (runmatDataArrayCopyString(
            member, 0, reinterpret_cast<unsigned short *>(text.data()), length) != 0) {
        throw InvalidArrayTypeException("could not read enumeration member name");
    }
    try {
        return encodeUTF8(text);
    } catch (const std::invalid_argument &error) {
        throw InvalidArrayTypeException(error.what());
    }
}
} // namespace detail

class EnumReference {
public:
    EnumReference(Array *owner, std::size_t index)
        : owner_(owner), index_(index) {}
    operator Enumeration() const {
        return Enumeration(detail::enumName(*owner_, index_));
    }
    operator std::string() const { return detail::enumName(*owner_, index_); }

private:
    Array *owner_;
    std::size_t index_;
};

class EnumIterator {
public:
    using difference_type = std::ptrdiff_t;
    using value_type = Enumeration;
    using pointer = void;
    using reference = Enumeration;
    using iterator_category = std::random_access_iterator_tag;

    EnumIterator() : owner_(nullptr), index_(0) {}
    EnumIterator(const Array *owner, std::size_t index)
        : owner_(owner), index_(index) {}
    Enumeration operator*() const {
        return Enumeration(detail::enumName(*owner_, index_));
    }
    EnumIterator &operator++() {
        ++index_;
        return *this;
    }
    EnumIterator operator++(int) {
        auto copy = *this;
        ++*this;
        return copy;
    }
    EnumIterator &operator--() {
        --index_;
        return *this;
    }
    EnumIterator &operator+=(difference_type amount) {
        index_ += amount;
        return *this;
    }
    EnumIterator &operator-=(difference_type amount) {
        index_ -= amount;
        return *this;
    }
    EnumIterator operator+(difference_type amount) const {
        auto copy = *this;
        return copy += amount;
    }
    EnumIterator operator-(difference_type amount) const {
        auto copy = *this;
        return copy -= amount;
    }
    Enumeration operator[](difference_type amount) const {
        return *(*this + amount);
    }
    difference_type operator-(const EnumIterator &other) const {
        return static_cast<difference_type>(index_) -
               static_cast<difference_type>(other.index_);
    }
    bool operator==(const EnumIterator &other) const {
        return owner_ == other.owner_ && index_ == other.index_;
    }
    bool operator!=(const EnumIterator &other) const { return !(*this == other); }
    bool operator<(const EnumIterator &other) const { return index_ < other.index_; }
    bool operator>(const EnumIterator &other) const { return other < *this; }
    bool operator<=(const EnumIterator &other) const { return !(other < *this); }
    bool operator>=(const EnumIterator &other) const { return !(*this < other); }

private:
    const Array *owner_;
    std::size_t index_;
};

template <> class TypedArray<Enumeration> : public Array {
public:
    using iterator = EnumIterator;
    using const_iterator = EnumIterator;

    TypedArray() = default;
    TypedArray(const Array &array) : Array(array) { validate(); }
    TypedArray(Array &&array) : Array(std::move(array)) { validate(); }
    iterator begin() { return iterator(this, 0); }
    iterator end() { return iterator(this, getNumberOfElements()); }
    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const {
        return const_iterator(this, getNumberOfElements());
    }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }
    EnumReference operator[](std::size_t index) {
        if (index >= getNumberOfElements()) throw std::out_of_range("enum index");
        return EnumReference(this, index);
    }
    Enumeration operator[](std::size_t index) const {
        if (index >= getNumberOfElements()) throw std::out_of_range("enum index");
        return Enumeration(detail::enumName(*this, index));
    }
    std::string getClassName() const {
        const char *name = mxGetClassName(native());
        return name == nullptr ? std::string() : std::string(name);
    }

private:
    void validate() const {
        if (getType() != ArrayType::ENUM) {
            throw InvalidArrayTypeException("array is not an enumeration array");
        }
    }
};

using EnumArray = TypedArray<Enumeration>;

} // namespace data
} // namespace matlab

#endif
