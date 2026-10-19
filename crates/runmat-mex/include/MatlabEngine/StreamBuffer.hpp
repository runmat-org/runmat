#ifndef RUNMAT_MATLAB_ENGINE_STREAM_BUFFER_HPP
#define RUNMAT_MATLAB_ENGINE_STREAM_BUFFER_HPP

#include <streambuf>

namespace matlab {
namespace engine {

using StreamBuffer = std::basic_streambuf<char16_t>;

} // namespace engine
} // namespace matlab

#endif
