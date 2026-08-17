#pragma once

class TopoDS_Shape;

namespace runmat_geometry_io {
namespace occt_backend {

struct OcctExactShapePayload;
struct OcctImportOptions;

void append_exact_topology(OcctExactShapePayload& result,
                           const TopoDS_Shape& root,
                           const OcctImportOptions& options);

} // namespace occt_backend
} // namespace runmat_geometry_io
