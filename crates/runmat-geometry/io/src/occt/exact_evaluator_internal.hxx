#pragma once

#include <TopoDS_Face.hxx>

#include <cstdint>

namespace runmat_geometry_io {
namespace occt_backend {
namespace detail {

struct ExactFaceContext {
  TopoDS_Face face;
  double meters_per_source_unit;
};

ExactFaceContext exact_face_context(std::uint64_t session_id,
                                    std::uint64_t face_key);

} // namespace detail
} // namespace occt_backend
} // namespace runmat_geometry_io
