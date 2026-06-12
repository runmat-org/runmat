#pragma once

#include "rust/cxx.h"

#include <cstdint>

namespace runmat_geometry_io {
namespace occt_backend {

enum class OcctCadFormat : std::uint8_t;
struct OcctImportOptions;
struct OcctImportPayload;

OcctImportPayload import_cad_bytes(rust::Str path,
                                   rust::Slice<const std::uint8_t> bytes,
                                   OcctCadFormat format,
                                   OcctImportOptions options);

} // namespace occt_backend
} // namespace runmat_geometry_io
