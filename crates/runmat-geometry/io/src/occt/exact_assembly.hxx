#pragma once

#include <XCAFDoc_ShapeTool.hxx>

#include <string>

class TopoDS_Shape;

namespace runmat_geometry_io {
namespace occt_backend {

struct OcctExactShapePayload;
struct OcctImportOptions;

std::string serialize_exact_shape(const TopoDS_Shape& shape,
                                  const OcctImportOptions& options);
void append_exact_occurrences(OcctExactShapePayload& result,
                              const TopoDS_Shape& root,
                              const Handle(XCAFDoc_ShapeTool)& shape_tool,
                              bool has_xcaf,
                              const std::string& representation,
                              const OcctImportOptions& options);

} // namespace occt_backend
} // namespace runmat_geometry_io
