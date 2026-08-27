#ifndef RUNMAT_CPP_MEX_HPP
#define RUNMAT_CPP_MEX_HPP

#include "MatlabDataArray.hpp"
#include "MatlabEngine/Exception.hpp"
#include "MatlabEngine/FutureResult.hpp"
#include "MatlabEngine/StreamBuffer.hpp"
#include "MatlabEngine/TypeConversion.hpp"
#include "mex.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

extern "C" RUNMAT_MEX_LOCAL std::uint64_t
runmatEngineContextCreate();
extern "C" RUNMAT_MEX_LOCAL void
runmatEngineContextRelease(std::uint64_t engineContext);
extern "C" RUNMAT_MEX_LOCAL std::uint64_t
runmatAsyncSubmitEval(std::uint64_t engineContext, const char *command,
                      int captureStdout, int captureStderr);
extern "C" RUNMAT_MEX_LOCAL std::uint64_t
runmatAsyncSubmitCall(std::uint64_t engineContext, const char *functionName,
                      std::size_t outputCount,
                      std::size_t inputCount,
                      const mxArray *const *inputs, int captureStdout,
                      int captureStderr);
extern "C" RUNMAT_MEX_LOCAL std::uint64_t runmatAsyncSubmitGetVariable(
    std::uint64_t engineContext, const char *workspace, const char *name);
extern "C" RUNMAT_MEX_LOCAL std::uint64_t runmatAsyncSubmitPutVariable(
    std::uint64_t engineContext, const char *workspace, const char *name,
    const mxArray *value);
extern "C" RUNMAT_MEX_LOCAL std::uint64_t runmatAsyncSubmitGetProperty(
    std::uint64_t engineContext, const mxArray *object, std::size_t index,
    const char *name);
extern "C" RUNMAT_MEX_LOCAL std::uint64_t runmatAsyncSubmitSetProperty(
    std::uint64_t engineContext, mxArray *object, std::size_t index,
    const char *name, const mxArray *value);

namespace matlab {
namespace engine {
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

} // namespace detail

inline std::u16string convertUTF8StringToUTF16String(const std::string &value) {
    return detail::utf16(value);
}

inline std::string convertUTF16StringToUTF8String(const std::u16string &value) {
    return detail::utf8(value);
}

class RunMatEngine {
public:
    RunMatEngine() : engineContext_(runmatEngineContextCreate()) {
        if (engineContext_ == 0) {
            throw Exception("could not acquire the MEX engine context");
        }
    }

    ~RunMatEngine() { runmatEngineContextRelease(engineContext_); }

    RunMatEngine(const RunMatEngine &) = delete;
    RunMatEngine &operator=(const RunMatEngine &) = delete;
    RunMatEngine(RunMatEngine &&) = delete;
    RunMatEngine &operator=(RunMatEngine &&) = delete;

    std::vector<data::Array>
    feval(const std::u16string &functionName, int outputCount,
          const std::vector<data::Array> &inputs,
          const std::shared_ptr<StreamBuffer> &output = {},
          const std::shared_ptr<StreamBuffer> &error = {}) {
        ensureOwner();
        if (outputCount < 0) throw Exception("output count cannot be negative");
        if (output || error) {
            return fevalAsync(functionName,
                              static_cast<std::size_t>(outputCount), inputs,
                              output, error)
                .get();
        }
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

    FutureResult<std::vector<data::Array>>
    fevalAsync(const std::u16string &functionName, std::size_t outputCount,
               const std::vector<data::Array> &inputs,
               const std::shared_ptr<StreamBuffer> &output = {},
               const std::shared_ptr<StreamBuffer> &error = {}) {
        std::vector<const mxArray *> nativeInputs;
        nativeInputs.reserve(inputs.size());
        for (const auto &input : inputs) {
            nativeInputs.push_back(data::detail::ArrayAccess::native(input));
        }
        const std::string name = detail::utf8(functionName);
        const auto request = runmatAsyncSubmitCall(
            engineContext_, name.c_str(), outputCount, nativeInputs.size(),
            nativeInputs.data(), output ? 1 : 0, error ? 1 : 0);
        if (request == 0) {
            throw Exception("could not submit asynchronous engine call: " + name);
        }
        return FutureResult<std::vector<data::Array>>(
            std::make_shared<detail::AsyncControl<std::vector<data::Array>>>(
                request, outputCount, output, error));
    }

    FutureResult<data::Array>
    fevalAsync(const std::u16string &functionName,
               const std::vector<data::Array> &inputs,
               const std::shared_ptr<StreamBuffer> &output = {},
               const std::shared_ptr<StreamBuffer> &error = {}) {
        std::vector<const mxArray *> nativeInputs;
        nativeInputs.reserve(inputs.size());
        for (const auto &input : inputs) {
            nativeInputs.push_back(data::detail::ArrayAccess::native(input));
        }
        const std::string name = detail::utf8(functionName);
        const auto request = runmatAsyncSubmitCall(
            engineContext_, name.c_str(), 1, nativeInputs.size(),
            nativeInputs.data(), output ? 1 : 0, error ? 1 : 0);
        if (request == 0) {
            throw Exception("could not submit asynchronous engine call: " + name);
        }
        return FutureResult<data::Array>(
            std::make_shared<detail::AsyncControl<data::Array>>(
                request, 1, output, error));
    }

    FutureResult<data::Array>
    fevalAsync(const std::u16string &functionName, const data::Array &input,
               const std::shared_ptr<StreamBuffer> &output = {},
               const std::shared_ptr<StreamBuffer> &error = {}) {
        return fevalAsync(functionName, std::vector<data::Array>{input}, output,
                          error);
    }

    FutureResult<data::Array>
    fevalAsync(const std::string &functionName,
               const std::vector<data::Array> &inputs,
               const std::shared_ptr<StreamBuffer> &output = {},
               const std::shared_ptr<StreamBuffer> &error = {}) {
        return fevalAsync(detail::utf16(functionName), inputs, output, error);
    }

    FutureResult<data::Array>
    fevalAsync(const std::string &functionName, const data::Array &input,
               const std::shared_ptr<StreamBuffer> &output = {},
               const std::shared_ptr<StreamBuffer> &error = {}) {
        return fevalAsync(detail::utf16(functionName), input, output, error);
    }

    template <typename ReturnType, typename... Inputs>
    ReturnType feval(const std::u16string &functionName, Inputs &&...inputs) {
        static_assert(
            !std::is_same_v<std::decay_t<ReturnType>,
                            std::vector<data::Array>>,
            "use the output-count overload for multiple engine outputs");
        auto arguments = detail::engineInputs(std::forward<Inputs>(inputs)...);
        if constexpr (std::is_void_v<ReturnType>) {
            (void)feval(functionName, 0, arguments);
        } else {
            auto outputs = feval(functionName, 1, arguments);
            return detail::engineOutput<ReturnType>(std::move(outputs.front()));
        }
    }

    template <typename ReturnType, typename... Inputs>
    ReturnType feval(const std::string &functionName, Inputs &&...inputs) {
        return feval<ReturnType>(detail::utf16(functionName),
                                 std::forward<Inputs>(inputs)...);
    }

    template <typename ReturnType, typename... Inputs>
    FutureResult<ReturnType>
    fevalAsync(const std::u16string &functionName, Inputs &&...inputs) {
        static_assert(
            !std::is_same_v<std::decay_t<ReturnType>,
                            std::vector<data::Array>>,
            "use the output-count overload for multiple engine outputs");
        auto arguments = detail::engineInputs(std::forward<Inputs>(inputs)...);
        std::vector<const mxArray *> nativeInputs;
        nativeInputs.reserve(arguments.size());
        for (const auto &input : arguments) {
            nativeInputs.push_back(data::detail::ArrayAccess::native(input));
        }
        const std::string name = detail::utf8(functionName);
        constexpr std::size_t outputCount = std::is_void_v<ReturnType> ? 0 : 1;
        const auto request = runmatAsyncSubmitCall(
            engineContext_, name.c_str(), outputCount, nativeInputs.size(),
            nativeInputs.data(), 0, 0);
        if (request == 0) {
            throw Exception("could not submit asynchronous engine call: " + name);
        }
        return FutureResult<ReturnType>(
            std::make_shared<detail::AsyncControl<ReturnType>>(request,
                                                               outputCount));
    }

    template <typename ReturnType, typename... Inputs>
    FutureResult<ReturnType>
    fevalAsync(const std::string &functionName, Inputs &&...inputs) {
        return fevalAsync<ReturnType>(detail::utf16(functionName),
                                      std::forward<Inputs>(inputs)...);
    }

    void eval(const std::u16string &command,
              const std::shared_ptr<StreamBuffer> &output = {},
              const std::shared_ptr<StreamBuffer> &error = {}) {
        ensureOwner();
        if (output || error) {
            evalAsync(command, output, error).get();
            return;
        }
        const std::string encoded = detail::utf8(command);
        if (mexEvalString(encoded.c_str()) != 0) {
            throw MATLABExecutionException("RunMat evaluation failed");
        }
    }

    FutureResult<void>
    evalAsync(const std::u16string &command,
              const std::shared_ptr<StreamBuffer> &output = {},
              const std::shared_ptr<StreamBuffer> &error = {}) {
        const std::string encoded = detail::utf8(command);
        const auto request = runmatAsyncSubmitEval(
            engineContext_, encoded.c_str(), output ? 1 : 0, error ? 1 : 0);
        if (request == 0) {
            throw Exception("could not submit asynchronous engine evaluation");
        }
        return FutureResult<void>(
            std::make_shared<detail::AsyncControl<void>>(
                request, 0, output, error));
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

    FutureResult<data::Array> getVariableAsync(
        const std::u16string &variableName,
        WorkspaceType workspace = WorkspaceType::BASE) {
        const std::string name = detail::utf8(variableName);
        const auto request = runmatAsyncSubmitGetVariable(
            engineContext_, detail::workspaceName(workspace), name.c_str());
        if (request == 0) {
            throw Exception("could not submit asynchronous workspace read: " + name);
        }
        return FutureResult<data::Array>(
            std::make_shared<detail::AsyncControl<data::Array>>(request, 1));
    }

    FutureResult<data::Array> getVariableAsync(
        const std::string &variableName,
        WorkspaceType workspace = WorkspaceType::BASE) {
        return getVariableAsync(detail::utf16(variableName), workspace);
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

    FutureResult<void> setVariableAsync(
        const std::u16string &variableName, const data::Array &value,
        WorkspaceType workspace = WorkspaceType::BASE) {
        const std::string name = detail::utf8(variableName);
        const auto request = runmatAsyncSubmitPutVariable(
            engineContext_, detail::workspaceName(workspace), name.c_str(),
            data::detail::ArrayAccess::native(value));
        if (request == 0) {
            throw Exception("could not submit asynchronous workspace write: " + name);
        }
        return FutureResult<void>(
            std::make_shared<detail::AsyncControl<void>>(request, 0));
    }

    FutureResult<void> setVariableAsync(
        const std::string &variableName, const data::Array &value,
        WorkspaceType workspace = WorkspaceType::BASE) {
        return setVariableAsync(detail::utf16(variableName), value, workspace);
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

    FutureResult<data::Array>
    getPropertyAsync(const data::Array &object,
                     const std::u16string &propertyName) {
        return getPropertyAsync(object, 0, propertyName);
    }

    FutureResult<data::Array>
    getPropertyAsync(const data::Array &object,
                     const std::string &propertyName) {
        return getPropertyAsync(object, 0, detail::utf16(propertyName));
    }

    FutureResult<data::Array>
    getPropertyAsync(const data::Array &object, std::size_t index,
                     const std::u16string &propertyName) {
        const std::string name = detail::utf8(propertyName);
        const auto request = runmatAsyncSubmitGetProperty(
            engineContext_, data::detail::ArrayAccess::native(object), index,
            name.c_str());
        if (request == 0) {
            throw Exception("could not submit asynchronous property read: " + name);
        }
        return FutureResult<data::Array>(
            std::make_shared<detail::AsyncControl<data::Array>>(request, 1));
    }

    FutureResult<data::Array>
    getPropertyAsync(const data::Array &object, std::size_t index,
                     const std::string &propertyName) {
        return getPropertyAsync(object, index, detail::utf16(propertyName));
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

    FutureResult<void>
    setPropertyAsync(data::Array &object, const std::u16string &propertyName,
                     const data::Array &propertyValue) {
        return setPropertyAsync(object, 0, propertyName, propertyValue);
    }

    FutureResult<void>
    setPropertyAsync(data::Array &object, const std::string &propertyName,
                     const data::Array &propertyValue) {
        return setPropertyAsync(object, 0, detail::utf16(propertyName),
                                propertyValue);
    }

    FutureResult<void>
    setPropertyAsync(data::Array &object, std::size_t index,
                     const std::u16string &propertyName,
                     const data::Array &propertyValue) {
        const std::string name = detail::utf8(propertyName);
        const auto request = runmatAsyncSubmitSetProperty(
            engineContext_, data::detail::ArrayAccess::native(object), index,
            name.c_str(), data::detail::ArrayAccess::native(propertyValue));
        if (request == 0) {
            throw Exception("could not submit asynchronous property write: " + name);
        }
        data::Array keepAlive = object;
        return FutureResult<void>(
            std::make_shared<detail::AsyncControl<void>>(
                request, 0, std::move(keepAlive)));
    }

    FutureResult<void>
    setPropertyAsync(data::Array &object, std::size_t index,
                     const std::string &propertyName,
                     const data::Array &propertyValue) {
        return setPropertyAsync(object, index, detail::utf16(propertyName),
                                propertyValue);
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
    std::uint64_t engineContext_;
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
        if (!engine_) engine_ = std::make_shared<engine::RunMatEngine>();
        return engine_;
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

private:
    std::shared_ptr<engine::RunMatEngine> engine_;
};

} // namespace mex
} // namespace matlab

#endif
