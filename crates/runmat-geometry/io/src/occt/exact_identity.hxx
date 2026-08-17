#pragma once

#include "rust/cxx.h"

#include <array>
#include <cstdint>
#include <unordered_map>

class TopoDS_Shape;

namespace runmat_geometry_io {
namespace occt_backend {

struct OcctImportOptions;

class ExactIdentityContext {
public:
  void append(rust::Vec<std::uint8_t>& destination,
              const TopoDS_Shape& shape,
              std::uint64_t shape_key,
              const OcctImportOptions& options);

private:
  std::unordered_map<std::uint64_t, std::array<std::uint8_t, 32>> digests_;
  std::uint64_t serialized_byte_work_ = 0;
};

} // namespace occt_backend
} // namespace runmat_geometry_io
