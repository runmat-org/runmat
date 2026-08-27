#ifndef RUNMAT_CPP_MEX_ADAPTER_HPP
#define RUNMAT_CPP_MEX_ADAPTER_HPP

#define RUNMAT_CPP_MEX_ADAPTER 1
#include "mex.hpp"

#if !defined(RUNMAT_MX_INTERLEAVED_COMPLEX)
#error "The modern C++ MEX/Data API requires the R2018a interleaved-complex contract"
#endif

class MexFunction;

namespace runmat_mex_detail {
template <typename Gateway>
void invokeCppGateway(int outputCount, mxArray **outputValues, int inputCount,
                      const mxArray **inputValues) {
    Gateway gateway;
    auto outputs = matlab::mex::ArgumentList::outputs(outputCount);
    auto inputs = matlab::mex::ArgumentList::inputs(inputCount, inputValues);
    gateway(outputs, inputs);
    outputs.releaseOutputs(outputValues);
}
} // namespace runmat_mex_detail

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
