#pragma once

#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Wire.hxx>

#include <vector>

namespace runmat_geometry_io {
namespace occt_backend {

// Returns every direct wire edge use exactly once. OCCT's face-local traversal supplies canonical
// order and orientation where available; direct children complete repaired Boolean wires for
// which the traversal helper stops early.
std::vector<TopoDS_Edge> exact_wire_edge_uses(const TopoDS_Wire& wire,
                                              const TopoDS_Face& face);

} // namespace occt_backend
} // namespace runmat_geometry_io
