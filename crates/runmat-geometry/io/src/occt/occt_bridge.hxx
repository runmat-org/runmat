#pragma once

#include "rust/cxx.h"

#include <cstdint>

namespace runmat_geometry_io {
namespace occt_backend {

enum class OcctCadFormat : std::uint8_t;
struct OcctImportOptions;
struct OcctImportPayload;
struct OcctPreviewSessionChunkOptions;
struct OcctPreviewSessionChunkPayload;
struct OcctPreviewSessionStartPayload;

OcctImportPayload import_cad_bytes(rust::Str path,
                                   rust::Slice<const std::uint8_t> bytes,
                                   OcctCadFormat format,
                                   OcctImportOptions options);
OcctPreviewSessionStartPayload start_cad_preview_session(
    rust::Str path,
    rust::Slice<const std::uint8_t> bytes,
    OcctCadFormat format,
    OcctImportOptions options);
OcctPreviewSessionChunkPayload read_cad_preview_session_chunk(
    std::uint64_t session_id,
    OcctPreviewSessionChunkOptions options);
void close_cad_preview_session(std::uint64_t session_id);

} // namespace occt_backend
} // namespace runmat_geometry_io
