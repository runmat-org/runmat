#pragma once

#include <cstdint>

class TopoDS_Shape;
class BRepTools_ShapeSet;

namespace runmat_geometry_io {
namespace occt_backend {

struct OcctExactShapePayload;
struct OcctImportOptions;
class ExactIdentityContext;

void append_exact_topology(OcctExactShapePayload& result,
                           const TopoDS_Shape& root,
                           const BRepTools_ShapeSet& shape_set,
                           ExactIdentityContext& identity_context,
                           std::uint64_t occurrence_index,
                           const OcctImportOptions& options);

} // namespace occt_backend
} // namespace runmat_geometry_io
