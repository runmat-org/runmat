#ifndef RUNMAT_CPP_MEX_ADAPTER_HPP
#define RUNMAT_CPP_MEX_ADAPTER_HPP

#define RUNMAT_CPP_MEX_ADAPTER 1
#include "mex.hpp"

#include <memory>

#if defined(_WIN32)
#define RUNMAT_MEX_CPP_EXPORT __declspec(dllexport)
#else
#define RUNMAT_MEX_CPP_EXPORT __attribute__((visibility("default")))
#endif

#if !defined(RUNMAT_MX_INTERLEAVED_COMPLEX)
#error "The modern C++ MEX/Data API requires the R2018a interleaved-complex contract"
#endif

class MexFunction;

namespace runmat_mex_detail {
template <typename Gateway>
std::unique_ptr<Gateway> &cppGateway() {
    static std::unique_ptr<Gateway> gateway;
    return gateway;
}

template <typename Gateway>
void invokeCppGateway(int outputCount, mxArray **outputValues, int inputCount,
                      const mxArray **inputValues) {
    auto &gateway = cppGateway<Gateway>();
    if (!gateway) gateway = std::make_unique<Gateway>();
    auto outputs = matlab::mex::ArgumentList::outputs(outputCount);
    auto inputs = matlab::mex::ArgumentList::inputs(inputCount, inputValues);
    (*gateway)(outputs, inputs);
    outputs.releaseOutputs(outputValues);
}
} // namespace runmat_mex_detail

extern "C" RUNMAT_MEX_CPP_EXPORT void runmatMexCppUnload(void) {
    runmat_mex_detail::cppGateway<MexFunction>().reset();
}

#undef RUNMAT_MEX_CPP_EXPORT

extern "C" RUNMAT_MEX_LOCAL void
mexFunction(int outputCount, mxArray **outputValues, int inputCount,
            const mxArray **inputValues) {
    try {
        runmat_mex_detail::invokeCppGateway<MexFunction>(
            outputCount, outputValues, inputCount, inputValues);
    } catch (const std::exception &error) {
        runmatDataArraySetError("RunMat:mex:cppException", error.what());
    } catch (...) {
        runmatDataArraySetError(
            "RunMat:mex:cppException",
            "C++ MEX gateway threw an unknown exception");
    }
}

#endif
