#include "exact_wire.hxx"

#include <BRepTools_WireExplorer.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Iterator.hxx>

#include <stdexcept>

namespace runmat_geometry_io {
namespace occt_backend {

std::vector<TopoDS_Edge> exact_wire_edge_uses(const TopoDS_Wire& wire,
                                              const TopoDS_Face& face) {
  std::vector<TopoDS_Edge> direct_edges;
  for (TopoDS_Iterator children(wire, Standard_False, Standard_True);
       children.More(); children.Next()) {
    if (children.Value().ShapeType() != TopAbs_EDGE) {
      throw std::runtime_error("OCCT exact wire contains a non-edge child");
    }
    direct_edges.push_back(TopoDS::Edge(children.Value()));
  }

  std::vector<bool> direct_used(direct_edges.size(), false);
  std::vector<TopoDS_Edge> ordered_edges;
  for (BRepTools_WireExplorer explorer(wire, face); explorer.More(); explorer.Next()) {
    const TopoDS_Edge edge = explorer.Current();
    ordered_edges.push_back(edge);
    std::size_t match = direct_edges.size();
    for (std::size_t index = 0; index < direct_edges.size(); ++index) {
      if (!direct_used[index] && edge.IsEqual(direct_edges[index])) {
        match = index;
        break;
      }
    }
    if (match == direct_edges.size()) {
      for (std::size_t index = 0; index < direct_edges.size(); ++index) {
        if (!direct_used[index] && edge.IsSame(direct_edges[index])) {
          match = index;
          break;
        }
      }
    }
    if (match != direct_edges.size()) {
      direct_used[match] = true;
    } else {
      throw std::runtime_error(
          "OCCT wire traversal returned an edge absent from the direct incidence inventory");
    }
  }
  for (std::size_t index = 0; index < direct_edges.size(); ++index) {
    if (!direct_used[index]) {
      ordered_edges.push_back(direct_edges[index]);
    }
  }
  return ordered_edges;
}

} // namespace occt_backend
} // namespace runmat_geometry_io
