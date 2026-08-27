#ifndef RUNMAT_CPP_MEX_HPP
#define RUNMAT_CPP_MEX_HPP

#include "MatlabDataArray.hpp"
#include "mex.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace matlab {

class Exception : public std::runtime_error {
public:
    explicit Exception(const std::string &message) : std::runtime_error(message) {}
};

namespace engine {
namespace detail {
inline std::string utf8(const std::u16string &value) {
    std::string result;
    for (std::size_t index = 0; index < value.size(); ++index) {
        std::uint32_t code = value[index];
        if (code >= 0xd800 && code <= 0xdbff) {
            if (++index >= value.size()) throw Exception("invalid UTF-16 input");
            const std::uint32_t low = value[index];
            if (low < 0xdc00 || low > 0xdfff) throw Exception("invalid UTF-16 input");
            code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
        } else if (code >= 0xdc00 && code <= 0xdfff) {
            throw Exception("invalid UTF-16 input");
        }
        if (code <= 0x7f) {
            result.push_back(static_cast<char>(code));
        } else if (code <= 0x7ff) {
            result.push_back(static_cast<char>(0xc0 | (code >> 6)));
            result.push_back(static_cast<char>(0x80 | (code & 0x3f)));
        } else if (code <= 0xffff) {
            result.push_back(static_cast<char>(0xe0 | (code >> 12)));
            result.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
            result.push_back(static_cast<char>(0x80 | (code & 0x3f)));
        } else {
            result.push_back(static_cast<char>(0xf0 | (code >> 18)));
            result.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3f)));
            result.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
            result.push_back(static_cast<char>(0x80 | (code & 0x3f)));
        }
    }
    return result;
}
} // namespace detail

class MATLABEngine {
public:
    std::vector<data::Array>
    feval(const std::u16string &functionName, std::size_t outputCount,
          const std::vector<data::Array> &inputs) {
        std::vector<mxArray *> nativeInputs;
        nativeInputs.reserve(inputs.size());
        for (const auto &input : inputs) {
            nativeInputs.push_back(data::detail::ArrayAccess::native(input));
        }
        std::vector<mxArray *> nativeOutputs(outputCount, nullptr);
        const std::string name = detail::utf8(functionName);
        if (mexCallMATLAB(static_cast<int>(nativeOutputs.size()),
                          nativeOutputs.data(),
                          static_cast<int>(nativeInputs.size()),
                          nativeInputs.data(), name.c_str()) != 0) {
            for (mxArray *output : nativeOutputs) {
                if (output != nullptr) mxDestroyArray(output);
            }
            throw Exception("RunMat callback failed: " + name);
        }
        std::vector<data::Array> outputs;
        outputs.reserve(nativeOutputs.size());
        for (mxArray *output : nativeOutputs) {
            outputs.push_back(data::Array::adopt(output));
        }
        return outputs;
    }

    data::Array feval(const std::u16string &functionName,
                      const std::vector<data::Array> &inputs) {
        auto outputs = feval(functionName, 1, inputs);
        return std::move(outputs.front());
    }

    void eval(const std::u16string &command) {
        const std::string encoded = detail::utf8(command);
        if (mexEvalString(encoded.c_str()) != 0) {
            throw Exception("RunMat evaluation failed");
        }
    }
};
} // namespace engine

namespace mex {

class ArgumentList {
public:
    using iterator = std::vector<data::Array>::iterator;
    using const_iterator = std::vector<data::Array>::const_iterator;

    ArgumentList() : values_(std::make_shared<std::vector<data::Array>>()) {}

    data::Array &operator[](std::size_t index) { return values_->at(index); }
    const data::Array &operator[](std::size_t index) const { return values_->at(index); }
    iterator begin() { return values_->begin(); }
    iterator end() { return values_->end(); }
    const_iterator begin() const { return values_->begin(); }
    const_iterator end() const { return values_->end(); }
    std::size_t size() const noexcept { return values_->size(); }
    bool empty() const noexcept { return values_->empty(); }

    static ArgumentList inputs(int count, const mxArray *const *values) {
        ArgumentList result;
        result.values_->reserve(static_cast<std::size_t>(count));
        for (int index = 0; index < count; ++index) {
            result.values_->push_back(data::Array::borrow(values[index]));
        }
        return result;
    }

    static ArgumentList outputs(int count) {
        ArgumentList result;
        result.values_->resize(static_cast<std::size_t>(count));
        return result;
    }

    void releaseOutputs(mxArray **outputs) {
        for (std::size_t index = 0; index < values_->size(); ++index) {
            if ((*values_)[index]) {
                outputs[index] =
                    data::detail::ArrayAccess::releaseForOutput((*values_)[index]);
            }
        }
    }

private:
    std::shared_ptr<std::vector<data::Array>> values_;
};

class Function {
public:
    virtual ~Function() = default;
    virtual void operator()(ArgumentList outputs, ArgumentList inputs) = 0;

protected:
    std::shared_ptr<engine::MATLABEngine> getEngine() {
        static auto instance = std::make_shared<engine::MATLABEngine>();
        return instance;
    }
    void mexLock() { ::mexLock(); }
    void mexUnlock() { ::mexUnlock(); }
    std::u16string getFunctionName() const {
        const char *name = ::mexFunctionName();
        std::u16string result;
        if (name != nullptr) {
            while (*name != '\0') result.push_back(static_cast<unsigned char>(*name++));
        }
        return result;
    }
};

} // namespace mex
} // namespace matlab

#endif
