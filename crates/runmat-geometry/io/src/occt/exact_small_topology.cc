#include "runmat-geometry-io/src/occt/ffi.rs.h"
#include "exact_healing.hxx"

#include <ShapeExtend_Status.hxx>
#include <ShapeFix_FixSmallFace.hxx>
#include <ShapeFix_Wireframe.hxx>
#include <Precision.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp.hxx>
#include <TopTools_IndexedMapOfShape.hxx>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace runmat_geometry_io {
namespace occt_backend {
namespace {

Standard_Integer entity_count(const TopoDS_Shape& shape,
                              const TopAbs_ShapeEnum kind) {
  TopTools_IndexedMapOfShape entities;
  TopExp::MapShapes(shape, kind, entities);
  return entities.Extent();
}

void check_small_topology_cancelled(const OcctImportOptions& options) {
  if (options.cancel_token_id != 0 && occt_import_cancelled(options.cancel_token_id)) {
    throw std::runtime_error("OCCT CAD import cancelled");
  }
}

} // namespace

ExactHealingMutation simplify_exact_small_topology(
    const TopoDS_Shape& shape,
    const OcctImportOptions& options,
    const double tolerance,
    const std::uint64_t initial_identity_work,
    bool& short_edges_simplified,
    bool& sliver_faces_simplified) {
  if (!std::isfinite(tolerance) || tolerance <= 0.0) {
    throw std::runtime_error(
        "OCCT small-topology repair requires a positive finite tolerance");
  }
  Handle(ShapeBuild_ReShape) context = new ShapeBuild_ReShape();
  ShapeFix_Wireframe wireframe(shape);
  wireframe.SetContext(context);
  wireframe.SetPrecision(tolerance);
  wireframe.SetMinTolerance(std::min(tolerance, Precision::Confusion()));
  wireframe.SetMaxTolerance(tolerance);
  wireframe.ModeDropSmallEdges() = Standard_False;
  wireframe.SetLimitAngle(-1.0);
  wireframe.FixSmallEdges();
  if (wireframe.StatusSmallEdges(ShapeExtend_FAIL1)) {
    throw std::runtime_error("OCCT failed to simplify one or more short edges");
  }
  const TopoDS_Shape after_edges = wireframe.Shape();
  if (after_edges.IsNull()) {
    throw std::runtime_error("OCCT short-edge simplification produced a null shape");
  }
  short_edges_simplified =
      entity_count(after_edges, TopAbs_EDGE) < entity_count(shape, TopAbs_EDGE);
  check_small_topology_cancelled(options);

  ShapeFix_FixSmallFace face_fixer;
  face_fixer.SetContext(context);
  face_fixer.SetPrecision(tolerance);
  face_fixer.SetMinTolerance(std::min(tolerance, Precision::Confusion()));
  face_fixer.SetMaxTolerance(tolerance);
  face_fixer.Init(after_edges);
  const TopoDS_Shape without_spot_faces = face_fixer.FixSpotFace();
  if (without_spot_faces.IsNull()) {
    throw std::runtime_error("OCCT spot-face simplification produced a null shape");
  }
  check_small_topology_cancelled(options);
  face_fixer.Init(without_spot_faces);
  const TopoDS_Shape result = face_fixer.FixStripFace();
  if (result.IsNull()) {
    throw std::runtime_error("OCCT strip-face simplification produced a null shape");
  }
  sliver_faces_simplified =
      entity_count(result, TopAbs_FACE) < entity_count(after_edges, TopAbs_FACE);
  check_small_topology_cancelled(options);

  ExactHealingMutation mutation;
  mutation.shape = result;
  mutation.identity_work_bytes = initial_identity_work;
  mutation.changed = short_edges_simplified || sliver_faces_simplified;
  if (mutation.changed) {
    append_small_topology_relations(shape, result, context, options, mutation);
    measure_healing_vertex_displacement(shape, result, mutation);
  }
  return mutation;
}

} // namespace occt_backend
} // namespace runmat_geometry_io
