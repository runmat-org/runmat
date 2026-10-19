#ifndef RUNMAT_MATLAB_ENGINE_EXCEPTION_HPP
#define RUNMAT_MATLAB_ENGINE_EXCEPTION_HPP

#include <stdexcept>
#include <string>

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

class TypeConversionException : public Exception {
public:
    explicit TypeConversionException(const std::string &message)
        : Exception(message) {}
};

class CancelException : public Exception {
public:
    CancelException() : Exception("asynchronous operation was cancelled") {}
};

class ThreadAffinityException : public Exception {
public:
    ThreadAffinityException()
        : Exception("synchronous C++ MEX engine operations require the gateway thread") {}
};

} // namespace engine
} // namespace matlab

#endif
