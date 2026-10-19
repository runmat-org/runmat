#include "exact_wire.hxx"

#include <BRepTools_WireExplorer.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Iterator.hxx>
#include <TopoDS_Vertex.hxx>

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace runmat_geometry_io {
namespace occt_backend {
namespace {

void require_connected_cycle(const std::vector<TopoDS_Edge>& edges) {
  if (edges.empty()) {
    throw std::runtime_error("OCCT exact wire has no edge uses to traverse");
  }
  for (std::size_t index = 0; index < edges.size(); ++index) {
    TopoDS_Vertex current_start;
    TopoDS_Vertex current_end;
    TopoDS_Vertex next_start;
    TopoDS_Vertex next_end;
    TopExp::Vertices(edges[index], current_start, current_end, Standard_True);
    TopExp::Vertices(
        edges[(index + 1) % edges.size()], next_start, next_end, Standard_True);
    if (current_start.IsNull() || current_end.IsNull() || next_start.IsNull() ||
        next_end.IsNull() || !current_end.IsSame(next_start)) {
      throw std::runtime_error(
          "OCCT wire edge uses do not form one oriented connected cycle");
    }
  }
}

std::vector<TopoDS_Edge> reconstruct_connected_traversal(
    const std::vector<TopoDS_Edge>& edges) {
  if (edges.empty()) {
    throw std::runtime_error("OCCT exact wire has no edge uses to traverse");
  }

  TopoDS_Vertex first;
  TopoDS_Vertex ignored_end;
  TopExp::Vertices(edges.front(), first, ignored_end, Standard_True);
  if (first.IsNull()) {
    throw std::runtime_error("OCCT wire traversal has no oriented start vertex");
  }

  std::vector<bool> used(edges.size(), false);
  std::vector<TopoDS_Vertex> vertex_stack{first};
  std::vector<std::size_t> edge_stack;
  std::vector<std::size_t> reverse_circuit;
  while (!vertex_stack.empty()) {
    const TopoDS_Vertex current = vertex_stack.back();
    std::size_t next_index = edges.size();
    TopoDS_Vertex next_vertex;
    for (std::size_t index = 0; index < edges.size(); ++index) {
      if (used[index]) {
        continue;
      }
      TopoDS_Vertex start;
      TopoDS_Vertex end;
      TopExp::Vertices(edges[index], start, end, Standard_True);
      if (!start.IsNull() && !end.IsNull() && start.IsSame(current)) {
        next_index = index;
        next_vertex = end;
        break;
      }
    }

    if (next_index != edges.size()) {
      used[next_index] = true;
      edge_stack.push_back(next_index);
      vertex_stack.push_back(next_vertex);
      continue;
    }

    vertex_stack.pop_back();
    if (!edge_stack.empty()) {
      reverse_circuit.push_back(edge_stack.back());
      edge_stack.pop_back();
    }
  }

  if (reverse_circuit.size() != edges.size()) {
    throw std::runtime_error(
        "OCCT wire edge uses cannot be reconstructed as one oriented connected cycle");
  }
  std::reverse(reverse_circuit.begin(), reverse_circuit.end());
  std::vector<TopoDS_Edge> ordered;
  ordered.reserve(edges.size());
  for (const std::size_t index : reverse_circuit) {
    ordered.push_back(edges[index]);
  }
  require_connected_cycle(ordered);
  return ordered;
}

} // namespace

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
  if (std::all_of(direct_used.begin(), direct_used.end(),
                  [](bool used) { return used; })) {
    require_connected_cycle(ordered_edges);
    return ordered_edges;
  }
  // Boolean and healing results can contain a valid wire for which WireExplorer stops before
  // visiting every stored edge use. Reconstruct the directed Euler cycle from authoritative
  // oriented vertex incidence instead of appending the omitted uses in storage order.
  return reconstruct_connected_traversal(direct_edges);
}

} // namespace occt_backend
} // namespace runmat_geometry_io
