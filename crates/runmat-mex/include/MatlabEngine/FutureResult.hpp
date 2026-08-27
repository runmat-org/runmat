#ifndef RUNMAT_MATLAB_ENGINE_FUTURE_RESULT_HPP
#define RUNMAT_MATLAB_ENGINE_FUTURE_RESULT_HPP

#include "MatlabDataArray/Array.hpp"
#include "MatlabDataArray/String.hpp"
#include "MatlabEngine/Exception.hpp"
#include "MatlabEngine/StreamBuffer.hpp"
#include "MatlabEngine/TypeConversion.hpp"

#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <type_traits>
#include <vector>

extern "C" RUNMAT_MEX_LOCAL int runmatAsyncIsReady(std::uint64_t request);
extern "C" RUNMAT_MEX_LOCAL int
runmatAsyncWait(std::uint64_t request, std::int64_t timeoutMillis);
extern "C" RUNMAT_MEX_LOCAL int
runmatAsyncCancel(std::uint64_t request, int allowInterrupt);
extern "C" RUNMAT_MEX_LOCAL int
runmatAsyncCopyResult(std::uint64_t request, std::size_t outputCapacity,
                      mxArray **outputs);
extern "C" RUNMAT_MEX_LOCAL std::size_t
runmatAsyncCopyText(std::uint64_t request, unsigned int field, char *output,
                    std::size_t outputCapacity);
extern "C" RUNMAT_MEX_LOCAL void runmatAsyncRelease(std::uint64_t request);

namespace matlab {
namespace engine {

class RunMatEngine;
using MATLABEngine = RunMatEngine;
template <typename T> class FutureResult;
template <typename T> class SharedFutureResult;

namespace detail {
inline std::string asyncText(std::uint64_t request, unsigned int field) {
    const auto length = runmatAsyncCopyText(request, field, nullptr, 0);
    std::string value(length, '\0');
    if (length != 0) {
        const auto required =
            runmatAsyncCopyText(request, field, value.data(), value.size());
        if (required != length) {
            throw Exception("asynchronous engine text changed while being read");
        }
    }
    return value;
}

inline void throwAsyncFailure(std::uint64_t request, int status) {
    if (status == 0) return;
    if (status == 2) throw CancelException();
    const auto identifier = asyncText(request, 2);
    const auto message = asyncText(request, 3);
    if (!identifier.empty() && !message.empty()) {
        throw MATLABExecutionException(identifier + ": " + message);
    }
    if (!message.empty()) throw MATLABExecutionException(message);
    if (!identifier.empty()) throw MATLABExecutionException(identifier);
    throw MATLABExecutionException("asynchronous engine operation failed");
}

inline void writeAsyncStream(std::uint64_t request, unsigned int field,
                             const std::shared_ptr<StreamBuffer> &stream) {
    if (!stream) return;
    const auto encoded = asyncText(request, field);
    if (encoded.empty()) return;
    std::u16string text;
    try {
        text = data::detail::decodeUTF8(encoded);
    } catch (const std::invalid_argument &error) {
        throw Exception(error.what());
    }
    const auto written = stream->sputn(text.data(),
                                       static_cast<std::streamsize>(text.size()));
    if (written != static_cast<std::streamsize>(text.size())) {
        throw Exception("C++ engine stream rejected captured output");
    }
    (void)stream->pubsync();
}

template <typename T> struct AsyncValue {
    static T read(std::uint64_t request, std::size_t) {
        mxArray *output = nullptr;
        throwAsyncFailure(request, runmatAsyncCopyResult(request, 1, &output));
        if (output == nullptr)
            throw MATLABExecutionException("asynchronous engine returned no value");
        return engineOutput<T>(data::Array::adopt(output));
    }
};

template <> struct AsyncValue<void> {
    static void read(std::uint64_t request, std::size_t) {
        throwAsyncFailure(request, runmatAsyncCopyResult(request, 0, nullptr));
    }
};

template <> struct AsyncValue<data::Array> {
    static data::Array read(std::uint64_t request, std::size_t) {
        mxArray *output = nullptr;
        throwAsyncFailure(request, runmatAsyncCopyResult(request, 1, &output));
        if (output == nullptr)
            throw MATLABExecutionException("asynchronous engine returned no value");
        return data::Array::adopt(output);
    }
};

template <> struct AsyncValue<std::vector<data::Array>> {
    static std::vector<data::Array> read(std::uint64_t request,
                                         std::size_t outputCount) {
        std::vector<mxArray *> native(outputCount, nullptr);
        throwAsyncFailure(
            request,
            runmatAsyncCopyResult(request, native.size(), native.data()));
        std::vector<data::Array> values;
        values.reserve(native.size());
        for (mxArray *value : native) values.push_back(data::Array::adopt(value));
        return values;
    }
};

template <typename T> class AsyncControl {
public:
    AsyncControl(std::uint64_t request, std::size_t outputCount,
                 std::shared_ptr<StreamBuffer> output = {},
                 std::shared_ptr<StreamBuffer> error = {})
        : request_(request), outputCount_(outputCount),
          output_(std::move(output)), error_(std::move(error)) {}

    ~AsyncControl() { runmatAsyncRelease(request_); }

    bool cancel(bool allowInterrupt) {
        return runmatAsyncCancel(request_, allowInterrupt ? 1 : 0) != 0;
    }

    bool ready() {
        const bool ready = runmatAsyncIsReady(request_) != 0;
        if (ready) flushStreams();
        return ready;
    }

    void wait() {
        (void)runmatAsyncWait(request_, -1);
        flushStreams();
    }

    template <typename Rep, typename Period>
    std::future_status waitFor(const std::chrono::duration<Rep, Period> &duration) {
        const auto millis = duration <= duration.zero()
                                ? std::chrono::milliseconds(0)
                                : std::chrono::duration_cast<std::chrono::milliseconds>(duration);
        if (runmatAsyncWait(request_, millis.count()) == 0)
            return std::future_status::timeout;
        flushStreams();
        return std::future_status::ready;
    }

    const T &sharedValue() {
        std::lock_guard<std::mutex> lock(mutex_);
        flushStreamsLocked();
        if (!value_) value_.emplace(AsyncValue<T>::read(request_, outputCount_));
        return *value_;
    }

    T takeValue() {
        std::lock_guard<std::mutex> lock(mutex_);
        flushStreamsLocked();
        if (!value_) value_.emplace(AsyncValue<T>::read(request_, outputCount_));
        return std::move(*value_);
    }

private:
    std::uint64_t request_;
    std::size_t outputCount_;
    std::mutex mutex_;
    std::optional<T> value_;
    std::shared_ptr<StreamBuffer> output_;
    std::shared_ptr<StreamBuffer> error_;
    bool streamsFlushed_ = false;

    void flushStreams() {
        std::lock_guard<std::mutex> lock(mutex_);
        flushStreamsLocked();
    }
    void flushStreamsLocked() {
        if (streamsFlushed_) return;
        writeAsyncStream(request_, 0, output_);
        writeAsyncStream(request_, 1, error_);
        streamsFlushed_ = true;
    }
};

template <> class AsyncControl<void> {
public:
    AsyncControl(std::uint64_t request, std::size_t,
                 std::shared_ptr<StreamBuffer> output = {},
                 std::shared_ptr<StreamBuffer> error = {})
        : request_(request), output_(std::move(output)), error_(std::move(error)) {}
    AsyncControl(std::uint64_t request, std::size_t, data::Array keepAlive)
        : request_(request), keepAlive_(std::move(keepAlive)) {}
    ~AsyncControl() { runmatAsyncRelease(request_); }
    bool cancel(bool allowInterrupt) {
        return runmatAsyncCancel(request_, allowInterrupt ? 1 : 0) != 0;
    }
    bool ready() {
        const bool ready = runmatAsyncIsReady(request_) != 0;
        if (ready) flushStreams();
        return ready;
    }
    void wait() {
        (void)runmatAsyncWait(request_, -1);
        flushStreams();
    }
    template <typename Rep, typename Period>
    std::future_status waitFor(const std::chrono::duration<Rep, Period> &duration) {
        const auto millis = duration <= duration.zero()
                                ? std::chrono::milliseconds(0)
                                : std::chrono::duration_cast<std::chrono::milliseconds>(duration);
        if (runmatAsyncWait(request_, millis.count()) == 0)
            return std::future_status::timeout;
        flushStreams();
        return std::future_status::ready;
    }
    void sharedValue() {
        std::lock_guard<std::mutex> lock(mutex_);
        flushStreamsLocked();
        if (!read_) {
            AsyncValue<void>::read(request_, 0);
            read_ = true;
        }
    }
    void takeValue() { sharedValue(); }

private:
    std::uint64_t request_;
    std::optional<data::Array> keepAlive_;
    std::mutex mutex_;
    bool read_ = false;
    std::shared_ptr<StreamBuffer> output_;
    std::shared_ptr<StreamBuffer> error_;
    bool streamsFlushed_ = false;

    void flushStreams() {
        std::lock_guard<std::mutex> lock(mutex_);
        flushStreamsLocked();
    }
    void flushStreamsLocked() {
        if (streamsFlushed_) return;
        writeAsyncStream(request_, 0, output_);
        writeAsyncStream(request_, 1, error_);
        streamsFlushed_ = true;
    }
};
} // namespace detail

template <typename T> class FutureResult {
public:
    FutureResult() = default;
    FutureResult(FutureResult &&) noexcept = default;
    FutureResult &operator=(FutureResult &&) noexcept = default;
    FutureResult(const FutureResult &) = delete;
    FutureResult &operator=(const FutureResult &) = delete;

    bool valid() const noexcept { return static_cast<bool>(control_); }
    bool cancel(bool allowInterrupt = true) {
        return control_ && control_->cancel(allowInterrupt);
    }
    void wait() const { require().wait(); }
    template <typename Rep, typename Period>
    std::future_status wait_for(const std::chrono::duration<Rep, Period> &duration) const {
        return require().waitFor(duration);
    }
    template <typename Clock, typename Duration>
    std::future_status
    wait_until(const std::chrono::time_point<Clock, Duration> &deadline) const {
        return wait_for(deadline - Clock::now());
    }
    T get() {
        auto control = release();
        return control->takeValue();
    }
    SharedFutureResult<T> share();

private:
    explicit FutureResult(std::shared_ptr<detail::AsyncControl<T>> control)
        : control_(std::move(control)) {}
    detail::AsyncControl<T> &require() const {
        if (!control_) throw Exception("future has no shared state");
        return *control_;
    }
    std::shared_ptr<detail::AsyncControl<T>> release() {
        auto control = std::move(control_);
        if (!control) throw Exception("future has no shared state");
        return control;
    }
    std::shared_ptr<detail::AsyncControl<T>> control_;
    friend class RunMatEngine;
    friend class SharedFutureResult<T>;
};

template <typename T> class SharedFutureResult {
public:
    SharedFutureResult() = default;
    bool valid() const noexcept { return static_cast<bool>(control_); }
    bool cancel(bool allowInterrupt = true) {
        return control_ && control_->cancel(allowInterrupt);
    }
    void wait() const { require().wait(); }
    template <typename Rep, typename Period>
    std::future_status wait_for(const std::chrono::duration<Rep, Period> &duration) const {
        return require().waitFor(duration);
    }
    template <typename Clock, typename Duration>
    std::future_status
    wait_until(const std::chrono::time_point<Clock, Duration> &deadline) const {
        return wait_for(deadline - Clock::now());
    }
    decltype(auto) get() const {
        if constexpr (std::is_void_v<T>) {
            require().sharedValue();
            return;
        } else {
            return require().sharedValue();
        }
    }

private:
    explicit SharedFutureResult(std::shared_ptr<detail::AsyncControl<T>> control)
        : control_(std::move(control)) {}
    detail::AsyncControl<T> &require() const {
        if (!control_) throw Exception("future has no shared state");
        return *control_;
    }
    std::shared_ptr<detail::AsyncControl<T>> control_;
    friend class FutureResult<T>;
};

template <typename T> SharedFutureResult<T> FutureResult<T>::share() {
    return SharedFutureResult<T>(release());
}

} // namespace engine
} // namespace matlab

#endif
