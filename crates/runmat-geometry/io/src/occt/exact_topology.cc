#include "runmat-geometry-io/src/occt/ffi.rs.h"
#include "exact_identity.hxx"
#include "exact_nesting.hxx"
#include "exact_topology.hxx"

#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepClass3d.hxx>
#include <BRepTools.hxx>
#include <BRepTools_ShapeSet.hxx>
#include <BRepTools_WireExplorer.hxx>
#include <BRep_Tool.hxx>
#include <Geom2d_Curve.hxx>
#include <TopAbs_Orientation.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Iterator.hxx>
#include <TopoDS_CompSolid.hxx>
#include <TopoDS_Shell.hxx>
#include <TopoDS_Solid.hxx>
#include <TopoDS_Vertex.hxx>
#include <TopoDS_Wire.hxx>
#include <gp_Pnt.hxx>

#include <algorithm>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace runmat_geometry_io {
namespace occt_backend {
namespace {

void check_cancelled(const OcctImportOptions& options) {
  if (options.cancel_token_id != 0 && occt_import_cancelled(options.cancel_token_id)) {
    throw std::runtime_error("OCCT CAD import cancelled");
  }
}

bool reversed_orientation(const TopoDS_Shape& shape, const char* role) {
  switch (shape.Orientation()) {
    case TopAbs_FORWARD:
      return false;
    case TopAbs_REVERSED:
      return true;
    default:
      throw std::runtime_error(std::string("OCCT exact topology contains unsupported ") +
                               role + " orientation");
  }
}

std::uint64_t shape_key(const BRepTools_ShapeSet& shapes,
                        const TopoDS_Shape& shape,
                        const char* role) {
  const Standard_Integer index = shapes.Index(shape);
  if (index <= 0) {
    throw std::runtime_error(std::string("OCCT exact topology omitted ") + role +
                             " from the canonical B-rep shape table");
  }
  return static_cast<std::uint64_t>(index);
}

} // namespace

void append_exact_topology(OcctExactShapePayload& result,
                           const TopoDS_Shape& root,
                           const BRepTools_ShapeSet& shape_set,
                           ExactIdentityContext& identity_context,
                           std::uint64_t occurrence_index,
                           const OcctImportOptions& options) {
  std::set<std::uint64_t> vertex_keys;
  for (TopExp_Explorer explorer(root, TopAbs_VERTEX); explorer.More(); explorer.Next()) {
    check_cancelled(options);
    const TopoDS_Vertex vertex = TopoDS::Vertex(explorer.Current());
    const gp_Pnt point = BRep_Tool::Pnt(vertex);
    OcctExactVertexPayload payload;
    payload.occurrence_index = occurrence_index;
    payload.shape_key = shape_key(shape_set, vertex, "vertex");
    if (!vertex_keys.insert(payload.shape_key).second) {
      continue;
    }
    identity_context.append(payload.identity_digest,
                            shape_set.Shape(static_cast<Standard_Integer>(payload.shape_key)),
                            payload.shape_key,
                            options);
    payload.point_x = point.X();
    payload.point_y = point.Y();
    payload.point_z = point.Z();
    payload.tolerance = BRep_Tool::Tolerance(vertex);
    result.vertices.push_back(payload);
  }
  std::sort(result.vertices.begin(), result.vertices.end(),
            [](const auto& left, const auto& right) { return left.shape_key < right.shape_key; });

  std::set<std::uint64_t> edge_keys;
  for (TopExp_Explorer explorer(root, TopAbs_EDGE); explorer.More(); explorer.Next()) {
    check_cancelled(options);
    const TopoDS_Edge edge_use = TopoDS::Edge(explorer.Current());
    const std::uint64_t edge_key = shape_key(shape_set, edge_use, "edge");
    const TopoDS_Edge edge = TopoDS::Edge(shape_set.Shape(static_cast<Standard_Integer>(edge_key)));
    TopoDS_Vertex start;
    TopoDS_Vertex end;
    // ExactEdge endpoints are tied to increasing evaluator parameter, independent of any
    // oriented coedge use. FORWARD/REVERSED child roles provide that canonical ordering.
    TopExp::Vertices(edge, start, end, Standard_False);
    const bool closed = BRep_Tool::IsClosed(edge);
    BRepAdaptor_Curve curve(edge);
    OcctExactEdgePayload payload;
    payload.occurrence_index = occurrence_index;
    payload.shape_key = edge_key;
    if (!edge_keys.insert(payload.shape_key).second) {
      continue;
    }
    identity_context.append(payload.identity_digest, edge, payload.shape_key, options);
    payload.start_vertex_key = start.IsNull() ? 0 : shape_key(shape_set, start, "edge vertex");
    payload.end_vertex_key = end.IsNull() ? 0 : shape_key(shape_set, end, "edge vertex");
    payload.closed = closed;
    payload.periodic = closed && curve.IsPeriodic();
    payload.degenerate = BRep_Tool::Degenerated(edge);
    result.edges.push_back(payload);
  }
  std::sort(result.edges.begin(), result.edges.end(),
            [](const auto& left, const auto& right) { return left.shape_key < right.shape_key; });

  std::set<std::uint64_t> face_keys;
  for (TopExp_Explorer explorer(root, TopAbs_FACE); explorer.More(); explorer.Next()) {
    check_cancelled(options);
    const TopoDS_Face face = TopoDS::Face(explorer.Current());
    const std::uint64_t face_key = shape_key(shape_set, face, "face");
    if (!face_keys.insert(face_key).second) {
      continue;
    }
    BRepAdaptor_Surface surface(face, Standard_False);
    const bool periodic_u = surface.IsUPeriodic();
    const bool periodic_v = surface.IsVPeriodic();
    const TopoDS_Wire outer_wire = BRepTools::OuterWire(face);
    if (outer_wire.IsNull()) {
      throw std::runtime_error("OCCT exact face has no outer wire");
    }

    OcctExactFacePayload face_payload;
    face_payload.occurrence_index = occurrence_index;
    face_payload.shape_key = face_key;
    identity_context.append(face_payload.identity_digest,
                            shape_set.Shape(static_cast<Standard_Integer>(face_key)),
                            face_key,
                            options);
    face_payload.reversed = reversed_orientation(face, "face");
    face_payload.outer_wire_key = shape_key(shape_set, outer_wire, "outer wire");
    face_payload.periodic_u = periodic_u;
    face_payload.periodic_v = periodic_v;
    face_payload.singular = false;

    for (TopoDS_Iterator wires(face, Standard_False, Standard_True); wires.More(); wires.Next()) {
      if (wires.Value().ShapeType() != TopAbs_WIRE) {
        continue;
      }
      const TopoDS_Wire wire = TopoDS::Wire(wires.Value());
      const std::uint64_t wire_key = shape_key(shape_set, wire, "wire");
      if (!wire.IsSame(outer_wire)) {
        face_payload.inner_wire_keys.push_back(wire_key);
      }

      OcctExactWirePayload wire_payload;
      wire_payload.occurrence_index = occurrence_index;
      wire_payload.shape_key = wire_key;
      identity_context.append(wire_payload.identity_digest,
                              shape_set.Shape(static_cast<Standard_Integer>(wire_key)),
                              wire_key,
                              options);
      wire_payload.face_key = face_key;
      wire_payload.reversed = reversed_orientation(wire, "wire");
      std::unordered_map<std::uint64_t, std::uint8_t> seam_occurrences;
      BRepTools_WireExplorer edge_uses(wire, face);
      for (; edge_uses.More(); edge_uses.Next()) {
        const TopoDS_Edge edge = edge_uses.Current();
        const std::uint64_t edge_key = shape_key(shape_set, edge, "coedge edge");
        Standard_Real first = 0.0;
        Standard_Real last = 0.0;
        const Handle(Geom2d_Curve) pcurve = BRep_Tool::CurveOnSurface(edge, face, first, last);

        OcctExactCoedgePayload coedge_payload;
        coedge_payload.occurrence_index = occurrence_index;
        coedge_payload.coedge_key =
            static_cast<std::uint64_t>(wire_payload.coedge_keys.size()) + 1;
        coedge_payload.face_key = face_key;
        coedge_payload.wire_key = wire_key;
        coedge_payload.edge_key = edge_key;
        coedge_payload.reversed = reversed_orientation(edge, "coedge");
        coedge_payload.has_pcurve = !pcurve.IsNull();
        coedge_payload.seam_image = -1;
        if ((periodic_u || periodic_v) && BRep_Tool::IsClosed(edge, face)) {
          const std::uint8_t occurrence = seam_occurrences[edge_key]++;
          if (occurrence > 1) {
            throw std::runtime_error("OCCT exact seam edge has more than two face uses");
          }
          TopoDS_Edge local_edge = edge;
          if (face.Orientation() == TopAbs_REVERSED) {
            local_edge.Reverse();
          }
          coedge_payload.seam_image =
              local_edge.Orientation() == TopAbs_REVERSED ? 1 : 0;
        }
        if (BRep_Tool::Degenerated(edge)) {
          face_payload.singular = true;
        }
        wire_payload.coedge_keys.push_back(coedge_payload.coedge_key);
        result.coedges.push_back(coedge_payload);
      }
      if (wire_payload.coedge_keys.empty()) {
        throw std::runtime_error("OCCT exact wire has no connected edge uses");
      }
      result.wires.push_back(wire_payload);
    }
    std::sort(face_payload.inner_wire_keys.begin(), face_payload.inner_wire_keys.end());
    result.faces.push_back(face_payload);
  }
  std::sort(result.faces.begin(), result.faces.end(),
            [](const auto& left, const auto& right) { return left.shape_key < right.shape_key; });
  std::sort(result.wires.begin(), result.wires.end(),
            [](const auto& left, const auto& right) { return left.shape_key < right.shape_key; });

  std::set<std::uint64_t> shell_keys;
  for (TopExp_Explorer explorer(root, TopAbs_SHELL); explorer.More(); explorer.Next()) {
    check_cancelled(options);
    const TopoDS_Shell shell = TopoDS::Shell(explorer.Current());
    OcctExactShellPayload payload;
    payload.occurrence_index = occurrence_index;
    payload.shape_key = shape_key(shape_set, shell, "shell");
    if (!shell_keys.insert(payload.shape_key).second) {
      continue;
    }
    identity_context.append(payload.identity_digest,
                            shape_set.Shape(static_cast<Standard_Integer>(payload.shape_key)),
                            payload.shape_key,
                            options);
    payload.reversed = reversed_orientation(shell, "shell");
    for (TopoDS_Iterator faces(shell, Standard_False, Standard_True); faces.More(); faces.Next()) {
      if (faces.Value().ShapeType() != TopAbs_FACE) {
        continue;
      }
      payload.face_keys.push_back(shape_key(shape_set, faces.Value(), "shell face"));
      payload.face_reversed.push_back(reversed_orientation(faces.Value(), "shell face use"));
    }
    result.shells.push_back(payload);
  }
  std::sort(result.shells.begin(), result.shells.end(),
            [](const auto& left, const auto& right) { return left.shape_key < right.shape_key; });

  std::set<std::uint64_t> solid_keys;
  for (TopExp_Explorer explorer(root, TopAbs_SOLID); explorer.More(); explorer.Next()) {
    check_cancelled(options);
    const TopoDS_Solid solid = TopoDS::Solid(explorer.Current());
    const TopoDS_Shell outer_shell = BRepClass3d::OuterShell(solid);
    if (outer_shell.IsNull()) {
      throw std::runtime_error("OCCT exact solid has no outer shell");
    }
    validate_exact_solid_nesting(solid, outer_shell, options);
    OcctExactSolidPayload payload;
    payload.occurrence_index = occurrence_index;
    payload.shape_key = shape_key(shape_set, solid, "solid");
    if (!solid_keys.insert(payload.shape_key).second) {
      continue;
    }
    identity_context.append(payload.identity_digest,
                            shape_set.Shape(static_cast<Standard_Integer>(payload.shape_key)),
                            payload.shape_key,
                            options);
    payload.outer_shell_key = shape_key(shape_set, outer_shell, "solid outer shell");
    for (TopoDS_Iterator shells(solid, Standard_False, Standard_True); shells.More(); shells.Next()) {
      if (shells.Value().ShapeType() != TopAbs_SHELL || shells.Value().IsSame(outer_shell)) {
        continue;
      }
      payload.void_shell_keys.push_back(shape_key(shape_set, shells.Value(), "solid void shell"));
    }
    std::sort(payload.void_shell_keys.begin(), payload.void_shell_keys.end());
    result.solids.push_back(payload);
  }
  std::sort(result.solids.begin(), result.solids.end(),
            [](const auto& left, const auto& right) { return left.shape_key < right.shape_key; });

  std::set<std::uint64_t> claimed_solids;
  std::set<std::uint64_t> compsolid_keys;
  for (TopExp_Explorer explorer(root, TopAbs_COMPSOLID); explorer.More(); explorer.Next()) {
    check_cancelled(options);
    const TopoDS_CompSolid compsolid = TopoDS::CompSolid(explorer.Current());
    OcctExactLumpPayload payload;
    payload.occurrence_index = occurrence_index;
    payload.shape_key = shape_key(shape_set, compsolid, "compsolid");
    payload.from_compsolid = true;
    if (!compsolid_keys.insert(payload.shape_key).second) {
      continue;
    }
    identity_context.append(payload.identity_digest,
                            shape_set.Shape(static_cast<Standard_Integer>(payload.shape_key)),
                            payload.shape_key,
                            options);
    for (TopExp_Explorer solids(compsolid, TopAbs_SOLID); solids.More(); solids.Next()) {
      const std::uint64_t solid_key = shape_key(shape_set, solids.Current(), "compsolid solid");
      if (!claimed_solids.insert(solid_key).second) {
        throw std::runtime_error("OCCT exact solid belongs to multiple compsolids");
      }
      payload.solid_keys.push_back(solid_key);
    }
    std::sort(payload.solid_keys.begin(), payload.solid_keys.end());
    if (payload.solid_keys.empty()) {
      throw std::runtime_error("OCCT exact compsolid contains no solids");
    }
    result.lumps.push_back(payload);
  }
  for (const auto& solid : result.solids) {
    if (solid.occurrence_index != occurrence_index) {
      continue;
    }
    if (claimed_solids.insert(solid.shape_key).second) {
      OcctExactLumpPayload payload;
      payload.occurrence_index = occurrence_index;
      payload.shape_key = solid.shape_key;
      payload.from_compsolid = false;
      for (const std::uint8_t byte : solid.identity_digest) {
        payload.identity_digest.push_back(byte);
      }
      payload.solid_keys.push_back(solid.shape_key);
      result.lumps.push_back(payload);
    }
  }
  std::sort(result.lumps.begin(), result.lumps.end(), [](const auto& left, const auto& right) {
    if (left.from_compsolid != right.from_compsolid) {
      return left.from_compsolid < right.from_compsolid;
    }
    return left.shape_key < right.shape_key;
  });
  if (claimed_solids.size() != solid_keys.size()) {
    throw std::runtime_error("OCCT exact lump extraction did not cover every solid");
  }
}

} // namespace occt_backend
} // namespace runmat_geometry_io
