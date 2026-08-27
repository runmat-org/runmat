#ifndef RUNMAT_CPP_MEX_HPP
#define RUNMAT_CPP_MEX_HPP

#include "MatlabDataArray.hpp"
#include "mex.h"

#include <cstddef>
#include <memory>
#include <streambuf>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace matlab {
namespace engine {
class Exception : public std::runtime_error {
public:
    explicit Exception(const std::string &message) : std::runtime_error(message) {}
};

class MATLABException : public Exception {
public:
    explicit MATLABException(const std::string &message) : Exception(message) {}
};

class MATLABExecutionException : public MATLABException {
public:
    explicit MATLABExecutionException(const std::string &message)
        : MATLABException(message) {}
};

class ThreadAffinityException : public Exception {
public:
    ThreadAffinityException()
        : Exception("C++ MEX engine operations require the originating MEX thread") {}
};

using StreamBuffer = std::basic_streambuf<char16_t>;

enum class WorkspaceType { BASE = 0, GLOBAL = 1 };

namespace detail {
inline std::string utf8(const std::u16string &value) {
    try {
        return data::detail::encodeUTF8(value);
    } catch (const std::invalid_argument &error) {
        throw Exception(error.what());
    }
}

inline std::u16string utf16(const std::string &value) {
    try {
        return data::detail::decodeUTF8(value);
    } catch (const std::invalid_argument &error) {
        throw Exception(error.what());
    }
}

inline const char *workspaceName(WorkspaceType workspace) {
    return workspace == WorkspaceType::GLOBAL ? "global" : "base";
}

inline void requireDefaultStreams(const std::shared_ptr<StreamBuffer> &output,
                                  const std::shared_ptr<StreamBuffer> &error) {
    if (output || error) {
        throw Exception("redirected C++ engine streams are not supported by this host");
    }
}
} // namespace detail

inline std::u16string convertUTF8StringToUTF16String(const std::string &value) {
    return detail::utf16(value);
}

inline std::string convertUTF16StringToUTF8String(const std::u16string &value) {
    return detail::utf8(value);
}

class MATLABEngine {
public:
    std::vector<data::Array>
    feval(const std::u16string &functionName, int outputCount,
          const std::vector<data::Array> &inputs,
          const std::shared_ptr<StreamBuffer> &output = {},
          const std::shared_ptr<StreamBuffer> &error = {}) {
        ensureOwner();
        detail::requireDefaultStreams(output, error);
        if (outputCount < 0) throw Exception("output count cannot be negative");
        std::vector<mxArray *> nativeInputs;
        nativeInputs.reserve(inputs.size());
        for (const auto &input : inputs) {
            nativeInputs.push_back(data::detail::ArrayAccess::native(input));
        }
        std::vector<mxArray *> nativeOutputs(static_cast<std::size_t>(outputCount),
                                             nullptr);
        const std::string name = detail::utf8(functionName);
        if (mexCallMATLAB(static_cast<int>(nativeOutputs.size()),
                          nativeOutputs.data(),
                          static_cast<int>(nativeInputs.size()),
                          nativeInputs.data(), name.c_str()) != 0) {
            for (mxArray *output : nativeOutputs) {
                if (output != nullptr) mxDestroyArray(output);
            }
            throw MATLABExecutionException("RunMat callback failed: " + name);
        }
        std::vector<data::Array> outputs;
        outputs.reserve(nativeOutputs.size());
        for (mxArray *output : nativeOutputs) {
            outputs.push_back(data::Array::adopt(output));
        }
        return outputs;
    }

    data::Array feval(const std::u16string &functionName,
                      const std::vector<data::Array> &inputs,
                      const std::shared_ptr<StreamBuffer> &output = {},
                      const std::shared_ptr<StreamBuffer> &error = {}) {
        auto outputs = feval(functionName, 1, inputs, output, error);
        return std::move(outputs.front());
    }

    data::Array feval(const std::u16string &functionName,
                      const data::Array &input,
                      const std::shared_ptr<StreamBuffer> &output = {},
                      const std::shared_ptr<StreamBuffer> &error = {}) {
        return feval(functionName, std::vector<data::Array>{input}, output, error);
    }

    std::vector<data::Array>
    feval(const std::string &functionName, int outputCount,
          const std::vector<data::Array> &inputs,
          const std::shared_ptr<StreamBuffer> &output = {},
          const std::shared_ptr<StreamBuffer> &error = {}) {
        return feval(detail::utf16(functionName), outputCount, inputs, output,
                     error);
    }

    data::Array feval(const std::string &functionName,
                      const std::vector<data::Array> &inputs,
                      const std::shared_ptr<StreamBuffer> &output = {},
                      const std::shared_ptr<StreamBuffer> &error = {}) {
        return feval(detail::utf16(functionName), inputs, output, error);
    }

    data::Array feval(const std::string &functionName, const data::Array &input,
                      const std::shared_ptr<StreamBuffer> &output = {},
                      const std::shared_ptr<StreamBuffer> &error = {}) {
        return feval(detail::utf16(functionName), input, output, error);
    }

    void eval(const std::u16string &command,
              const std::shared_ptr<StreamBuffer> &output = {},
              const std::shared_ptr<StreamBuffer> &error = {}) {
        ensureOwner();
        detail::requireDefaultStreams(output, error);
        const std::string encoded = detail::utf8(command);
        if (mexEvalString(encoded.c_str()) != 0) {
            throw MATLABExecutionException("RunMat evaluation failed");
        }
    }

    data::Array getVariable(
        const std::u16string &variableName,
        WorkspaceType workspace = WorkspaceType::BASE) {
        ensureOwner();
        const std::string name = detail::utf8(variableName);
        mxArray *value = mexGetVariable(detail::workspaceName(workspace),
                                        name.c_str());
        if (value == nullptr) {
            throw Exception("RunMat workspace variable was not found: " + name);
        }
        return data::Array::adopt(value);
    }

    data::Array getVariable(
        const std::string &variableName,
        WorkspaceType workspace = WorkspaceType::BASE) {
        return getVariable(detail::utf16(variableName), workspace);
    }

    void setVariable(const std::u16string &variableName,
                     const data::Array &value,
                     WorkspaceType workspace = WorkspaceType::BASE) {
        ensureOwner();
        const std::string name = detail::utf8(variableName);
        if (mexPutVariable(detail::workspaceName(workspace), name.c_str(),
                           data::detail::ArrayAccess::native(value)) != 0) {
            throw Exception("RunMat workspace assignment failed: " + name);
        }
    }

    void setVariable(const std::string &variableName, const data::Array &value,
                     WorkspaceType workspace = WorkspaceType::BASE) {
        setVariable(detail::utf16(variableName), value, workspace);
    }

    data::Array getProperty(const data::Array &object,
                            const std::u16string &propertyName) {
        return getProperty(object, 0, propertyName);
    }

    data::Array getProperty(const data::Array &object,
                            const std::string &propertyName) {
        return getProperty(object, 0, detail::utf16(propertyName));
    }

    data::Array getProperty(const data::Array &object, std::size_t index,
                            const std::u16string &propertyName) {
        const std::string name = detail::utf8(propertyName);
        mxArray *property = mxGetProperty(
            data::detail::ArrayAccess::native(object), index, name.c_str());
        if (property == nullptr) {
            throw Exception("RunMat object property was not found: " + name);
        }
        return data::Array::adopt(runmatDataArrayShare(property));
    }

    data::Array getProperty(const data::Array &object, std::size_t index,
                            const std::string &propertyName) {
        return getProperty(object, index, detail::utf16(propertyName));
    }

    void setProperty(data::Array &object, const std::u16string &propertyName,
                     const data::Array &propertyValue) {
        setProperty(object, 0, propertyName, propertyValue);
    }

    void setProperty(data::Array &object, const std::string &propertyName,
                     const data::Array &propertyValue) {
        setProperty(object, 0, detail::utf16(propertyName), propertyValue);
    }

    void setProperty(data::Array &object, std::size_t index,
                     const std::u16string &propertyName,
                     const data::Array &propertyValue) {
        data::detail::ArrayAccess::ensureContainerWritable(object);
        setPropertyOnSharedControl(object, index, propertyName, propertyValue);
    }

    void setProperty(data::Array &object, std::size_t index,
                     const std::string &propertyName,
                     const data::Array &propertyValue) {
        setProperty(object, index, detail::utf16(propertyName), propertyValue);
    }

private:
    void setPropertyOnSharedControl(data::Array &object, std::size_t index,
                                    const std::u16string &propertyName,
                                    const data::Array &propertyValue) {
        const std::string name = detail::utf8(propertyName);
        data::Array sharedValue = propertyValue;
        mxSetProperty(data::detail::ArrayAccess::native(object), index,
                      name.c_str(),
                      data::detail::ArrayAccess::releaseForOutput(sharedValue));
    }

    void ensureOwner() const {
        if (std::this_thread::get_id() != owner_) {
            throw ThreadAffinityException();
        }
    }

    std::thread::id owner_ = std::this_thread::get_id();
};
} // namespace engine

using Exception = engine::Exception;

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
        thread_local auto instance = std::make_shared<engine::MATLABEngine>();
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
