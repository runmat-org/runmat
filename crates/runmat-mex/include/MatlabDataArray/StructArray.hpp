#ifndef RUNMAT_MATLAB_DATA_ARRAY_STRUCT_ARRAY_HPP
#define RUNMAT_MATLAB_DATA_ARRAY_STRUCT_ARRAY_HPP

#include "Array.hpp"

#include <string>

namespace matlab {
namespace data {

class StructFieldReference {
public:
    StructFieldReference(Array *owner, std::size_t index, std::string field)
        : owner_(owner), index_(index), field_(std::move(field)) {}

    operator Array() const {
        mxArray *value = mxGetField(detail::ArrayAccess::native(*owner_), index_,
                                    field_.c_str());
        return value == nullptr ? Array() : Array::borrow(value);
    }

    StructFieldReference &operator=(Array value) {
        detail::ArrayAccess::ensureContainerWritable(*owner_);
        mxArray *native = value ? detail::ArrayAccess::releaseForOutput(value) : nullptr;
        mxSetField(detail::ArrayAccess::native(*owner_), index_, field_.c_str(), native);
        return *this;
    }

private:
    Array *owner_;
    std::size_t index_;
    std::string field_;
};

class StructReference {
public:
    StructReference(Array *owner, std::size_t index) : owner_(owner), index_(index) {}
    StructFieldReference operator[](const std::string &field) {
        return StructFieldReference(owner_, index_, field);
    }

private:
    Array *owner_;
    std::size_t index_;
};

class ConstStructReference {
public:
    ConstStructReference(const Array *owner, std::size_t index)
        : owner_(owner), index_(index) {}
    Array operator[](const std::string &field) const {
        mxArray *value = mxGetField(detail::ArrayAccess::native(*owner_), index_,
                                    field.c_str());
        return value == nullptr ? Array() : Array::borrow(value);
    }

private:
    const Array *owner_;
    std::size_t index_;
};

class StructArray : public Array {
public:
    StructArray() = default;
    StructArray(const Array &array) : Array(array) { validate(); }
    StructArray(Array &&array) : Array(std::move(array)) { validate(); }

    StructReference operator[](std::size_t index) {
        if (index >= getNumberOfElements()) throw std::out_of_range("struct index");
        return StructReference(this, index);
    }
    ConstStructReference operator[](std::size_t index) const {
        if (index >= getNumberOfElements()) throw std::out_of_range("struct index");
        return ConstStructReference(this, index);
    }

    std::vector<std::string> getFieldNames() const {
        std::vector<std::string> fields;
        const int count = mxGetNumberOfFields(detail::ArrayAccess::native(*this));
        fields.reserve(static_cast<std::size_t>(count));
        for (int index = 0; index < count; ++index) {
            const char *name =
                mxGetFieldNameByNumber(detail::ArrayAccess::native(*this), index);
            fields.emplace_back(name == nullptr ? "" : name);
        }
        return fields;
    }

private:
    void validate() const {
        if (getType() != ArrayType::STRUCT) {
            throw InvalidArrayTypeException("array is not a struct array");
        }
    }
};

} // namespace data
} // namespace matlab

#endif
