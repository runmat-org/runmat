#pragma once

#include "exact_names.hxx"

#include <TopoDS_Shape.hxx>
#include <XCAFDoc_ShapeTool.hxx>

namespace runmat_geometry_io {
namespace occt_backend {

struct OcctImportOptions;

struct ExactSharedInterfaces {
  // Compound of every source solid, placed, with each coincident face between two solids
  // replaced by one face that both solids use.
  TopoDS_Shape shape;
  ExactNameTable names;
};

// Joins touching solids so coincident faces become shared interfaces, the way a conformal
// multi-body mesh needs them (part in air, coil and core in air, fluid against a wall).
//
// Every solid of every placed part enters one General Fuse. Bodies may touch but must not
// interpenetrate: a solid that the fuse splits, or a fragment owned by two solids, is rejected
// with the names of the bodies involved. Solids keep their own names or their part's product
// name. A shared face carries the names authored on every face it replaces, ordered by the
// volume of the owning solid (smallest first) and then by name, so the part's name precedes
// the name of the medium around it.
ExactSharedInterfaces share_exact_interfaces(const TopoDS_Shape& root,
                                             const Handle(XCAFDoc_ShapeTool)& shape_tool,
                                             bool has_xcaf,
                                             const OcctImportOptions& options);

} // namespace occt_backend
} // namespace runmat_geometry_io
