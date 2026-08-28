#include "runmat-geometry-io/src/occt/ffi.rs.h"
#include "occt_bridge.hxx"
#include "exact_evaluator_internal.hxx"

#include <BRepAdaptor_Curve.hxx>
#include <BRepClass_FaceClassifier.hxx>
#include <BRepGProp.hxx>
#include <BRepTools_WireExplorer.hxx>
#include <BRep_Tool.hxx>
#include <BRepTools.hxx>
#include <BRepTools_ShapeSet.hxx>
#include <BRep_Builder.hxx>
#include <Extrema_ExtPC.hxx>
#include <GCPnts_AbscissaPoint.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopAbs_Orientation.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Iterator.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Wire.hxx>
#include <Geom2d_Curve.hxx>
#include <GProp_GProps.hxx>
#include <gp_Pnt2d.hxx>
#include <gp_Mat.hxx>
#include <gp_Vec2d.hxx>
#include <gp_Pnt.hxx>
#include <gp_Vec.hxx>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace runmat_geometry_io {
namespace occt_backend {
namespace {

struct ExactEvaluatorSession {
  TopoDS_Shape root;
  BRepTools_ShapeSet shapes{Standard_False};
  double meters_per_source_unit = 1.0;
};

std::mutex& sessions_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<std::uint64_t, std::shared_ptr<const ExactEvaluatorSession>>& sessions() {
  static std::unordered_map<std::uint64_t, std::shared_ptr<const ExactEvaluatorSession>> values;
  return values;
}

std::atomic<std::uint64_t>& next_session_id() {
  static std::atomic<std::uint64_t> value{1};
  return value;
}

std::shared_ptr<const ExactEvaluatorSession> session(std::uint64_t id) {
  std::lock_guard<std::mutex> lock(sessions_mutex());
  const auto found = sessions().find(id);
  if (found == sessions().end()) {
    throw std::runtime_error("OCCT exact evaluator session does not exist");
  }
  return found->second;
}

TopoDS_Edge edge(const ExactEvaluatorSession& session, std::uint64_t shape_key) {
  if (shape_key == 0 || shape_key > static_cast<std::uint64_t>(session.shapes.NbShapes())) {
    throw std::runtime_error("OCCT exact curve token is outside the B-rep shape table");
  }
  const TopoDS_Shape& shape = session.shapes.Shape(static_cast<Standard_Integer>(shape_key));
  if (shape.ShapeType() != TopAbs_EDGE) {
    throw std::runtime_error("OCCT exact curve token does not identify an edge");
  }
  return TopoDS::Edge(shape);
}

TopoDS_Face face(const ExactEvaluatorSession& session, std::uint64_t shape_key) {
  if (shape_key == 0 || shape_key > static_cast<std::uint64_t>(session.shapes.NbShapes())) {
    throw std::runtime_error("OCCT exact face token is outside the B-rep shape table");
  }
  const TopoDS_Shape& shape =
      session.shapes.Shape(static_cast<Standard_Integer>(shape_key));
  if (shape.ShapeType() == TopAbs_FACE) {
    return TopoDS::Face(shape);
  }
  throw std::runtime_error("OCCT exact face token does not identify a face");
}

void require_finite(double value, const char* role) {
  if (!std::isfinite(value)) {
    throw std::runtime_error(std::string("OCCT exact curve ") + role + " must be finite");
  }
}

void require_range(const BRepAdaptor_Curve& curve, double start, double end) {
  require_finite(start, "range start");
  require_finite(end, "range end");
  if (start > end || start < curve.FirstParameter() || end > curve.LastParameter()) {
    throw std::runtime_error("OCCT exact curve range is outside the edge domain");
  }
}

struct PcurveData {
  Handle(Geom2d_Curve) curve;
  double start = 0.0;
  double end = 0.0;
};

PcurveData pcurve(const ExactEvaluatorSession& session,
                  std::uint64_t face_key,
                  std::uint64_t wire_key,
                  std::uint64_t coedge_position,
                  std::int8_t seam_image) {
  if (face_key == 0 || wire_key == 0 || coedge_position == 0 ||
      face_key > static_cast<std::uint64_t>(session.shapes.NbShapes()) ||
      wire_key > static_cast<std::uint64_t>(session.shapes.NbShapes()) ||
      seam_image < -1 || seam_image > 1) {
    throw std::runtime_error("OCCT exact pcurve token is outside the B-rep topology");
  }

  const TopoDS_Face face_use = face(session, face_key);

  TopoDS_Wire wire;
  for (TopoDS_Iterator wires(face_use, Standard_False, Standard_True); wires.More(); wires.Next()) {
    if (wires.Value().ShapeType() == TopAbs_WIRE &&
        static_cast<std::uint64_t>(session.shapes.Index(wires.Value())) == wire_key) {
      wire = TopoDS::Wire(wires.Value());
      break;
    }
  }
  if (wire.IsNull()) {
    throw std::runtime_error("OCCT exact pcurve wire is not owned by its face");
  }

  BRepTools_WireExplorer edge_uses(wire, face_use);
  std::uint64_t position = 1;
  while (edge_uses.More() && position < coedge_position) {
    edge_uses.Next();
    ++position;
  }
  if (!edge_uses.More() || position != coedge_position) {
    throw std::runtime_error("OCCT exact pcurve coedge position is outside its wire");
  }
  TopoDS_Edge edge_use = edge_uses.Current();
  if (seam_image >= 0) {
    TopoDS_Edge local_edge = edge_use;
    if (face_use.Orientation() == TopAbs_REVERSED) {
      local_edge.Reverse();
    }
    const std::int8_t resolved_image =
        local_edge.Orientation() == TopAbs_REVERSED ? 1 : 0;
    if (resolved_image != seam_image || !BRep_Tool::IsClosed(edge_use, face_use)) {
      throw std::runtime_error("OCCT exact pcurve seam image does not match its oriented use");
    }
  } else if (BRep_Tool::IsClosed(edge_use, face_use)) {
    throw std::runtime_error("OCCT exact pcurve omitted a required seam image");
  }

  PcurveData result;
  result.curve = BRep_Tool::CurveOnSurface(edge_use, face_use, result.start, result.end);
  if (result.curve.IsNull()) {
    throw std::runtime_error("OCCT exact pcurve is unavailable for its face use");
  }
  require_finite(result.start, "pcurve range start");
  require_finite(result.end, "pcurve range end");
  if (result.start > result.end) {
    throw std::runtime_error("OCCT exact pcurve has a reversed parameter domain");
  }
  return result;
}

} // namespace

detail::ExactFaceContext detail::exact_face_context(std::uint64_t session_id,
                                                    std::uint64_t face_key) {
  const auto value = session(session_id);
  return {face(*value, face_key), value->meters_per_source_unit};
}

std::uint64_t start_exact_evaluator_session(
    rust::Slice<const std::uint8_t> representation,
    double meters_per_source_unit) {
  if (representation.empty()) {
    throw std::runtime_error("OCCT exact evaluator representation is empty");
  }
  if (!std::isfinite(meters_per_source_unit) || meters_per_source_unit <= 0.0) {
    throw std::runtime_error("OCCT exact evaluator unit scale must be finite and positive");
  }
  std::string bytes(reinterpret_cast<const char*>(representation.data()), representation.size());
  std::istringstream input(bytes);
  BRep_Builder builder;
  auto value = std::make_shared<ExactEvaluatorSession>();
  BRepTools::Read(value->root, input, builder);
  if (value->root.IsNull()) {
    throw std::runtime_error("OCCT exact evaluator could not read the B-rep representation");
  }
  value->shapes.Add(value->root);
  value->meters_per_source_unit = meters_per_source_unit;

  const std::uint64_t id = next_session_id().fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(sessions_mutex());
  sessions().emplace(id, std::move(value));
  return id;
}

OcctCurveRangePayload exact_curve_range(std::uint64_t session_id,
                                        std::uint64_t shape_key) {
  const auto value = session(session_id);
  BRepAdaptor_Curve curve(edge(*value, shape_key));
  OcctCurveRangePayload result;
  result.start = curve.FirstParameter();
  result.end = curve.LastParameter();
  require_finite(result.start, "range start");
  require_finite(result.end, "range end");
  if (result.start > result.end) {
    throw std::runtime_error("OCCT exact curve has a reversed parameter domain");
  }
  return result;
}

OcctCurveDerivativesPayload exact_curve_derivatives(std::uint64_t session_id,
                                                    std::uint64_t shape_key,
                                                    double parameter) {
  const auto value = session(session_id);
  BRepAdaptor_Curve curve(edge(*value, shape_key));
  require_range(curve, parameter, parameter);
  gp_Pnt point;
  gp_Vec first;
  gp_Vec second;
  curve.D2(parameter, point, first, second);
  const double scale = value->meters_per_source_unit;
  OcctCurveDerivativesPayload result;
  result.point_x = point.X() * scale;
  result.point_y = point.Y() * scale;
  result.point_z = point.Z() * scale;
  result.first_x = first.X() * scale;
  result.first_y = first.Y() * scale;
  result.first_z = first.Z() * scale;
  result.second_x = second.X() * scale;
  result.second_y = second.Y() * scale;
  result.second_z = second.Z() * scale;
  return result;
}

double exact_curve_arc_length(std::uint64_t session_id,
                              std::uint64_t shape_key,
                              double start,
                              double end,
                              double absolute_error_m) {
  const auto value = session(session_id);
  BRepAdaptor_Curve curve(edge(*value, shape_key));
  require_range(curve, start, end);
  require_finite(absolute_error_m, "arc-length tolerance");
  if (absolute_error_m <= 0.0) {
    throw std::runtime_error("OCCT exact curve arc-length tolerance must be positive");
  }
  const double source_tolerance = absolute_error_m / value->meters_per_source_unit;
  const double length = GCPnts_AbscissaPoint::Length(curve, start, end, source_tolerance) *
                        value->meters_per_source_unit;
  require_finite(length, "arc length");
  return length;
}

OcctCurveProjectionPayload exact_curve_inverse_project(
    std::uint64_t session_id,
    std::uint64_t shape_key,
    rust::Slice<const double> point_m,
    double absolute_error_m) {
  if (point_m.size() != 3) {
    throw std::runtime_error("OCCT exact curve projection requires one 3D point");
  }
  const auto value = session(session_id);
  for (double coordinate : point_m) {
    require_finite(coordinate, "projection point");
  }
  require_finite(absolute_error_m, "projection tolerance");
  if (absolute_error_m <= 0.0) {
    throw std::runtime_error("OCCT exact curve projection tolerance must be positive");
  }

  BRepAdaptor_Curve curve(edge(*value, shape_key));
  const double scale = value->meters_per_source_unit;
  const gp_Pnt query(point_m[0] / scale, point_m[1] / scale, point_m[2] / scale);
  const double tolerance = absolute_error_m / scale;
  Extrema_ExtPC extrema(query, curve, curve.FirstParameter(), curve.LastParameter(), tolerance);
  if (!extrema.IsDone()) {
    throw std::runtime_error("OCCT exact curve projection did not converge");
  }

  double best_parameter = curve.FirstParameter();
  gp_Pnt best_point;
  curve.D0(best_parameter, best_point);
  double best_squared_distance = query.SquareDistance(best_point);
  gp_Pnt end_point;
  curve.D0(curve.LastParameter(), end_point);
  const double end_squared_distance = query.SquareDistance(end_point);
  if (end_squared_distance < best_squared_distance) {
    best_parameter = curve.LastParameter();
    best_point = end_point;
    best_squared_distance = end_squared_distance;
  }
  for (Standard_Integer index = 1; index <= extrema.NbExt(); ++index) {
    if (!extrema.IsMin(index)) {
      continue;
    }
    const double squared_distance = extrema.SquareDistance(index);
    const double parameter = extrema.Point(index).Parameter();
    if (squared_distance < best_squared_distance ||
        (squared_distance == best_squared_distance && parameter < best_parameter)) {
      best_parameter = parameter;
      best_point = extrema.Point(index).Value();
      best_squared_distance = squared_distance;
    }
  }

  OcctCurveProjectionPayload result;
  result.parameter = best_parameter;
  result.point_x = best_point.X() * scale;
  result.point_y = best_point.Y() * scale;
  result.point_z = best_point.Z() * scale;
  result.distance = std::sqrt(best_squared_distance) * scale;
  require_finite(result.distance, "projection distance");
  return result;
}

OcctCurveRangePayload exact_pcurve_range(std::uint64_t session_id,
                                         std::uint64_t face_key,
                                         std::uint64_t wire_key,
                                         std::uint64_t coedge_position,
                                         std::int8_t seam_image) {
  const auto value = session(session_id);
  const PcurveData pcurve_value =
      pcurve(*value, face_key, wire_key, coedge_position, seam_image);
  OcctCurveRangePayload result;
  result.start = pcurve_value.start;
  result.end = pcurve_value.end;
  return result;
}

OcctPcurveDerivativesPayload exact_pcurve_derivatives(
    std::uint64_t session_id,
    std::uint64_t face_key,
    std::uint64_t wire_key,
    std::uint64_t coedge_position,
    std::int8_t seam_image,
    double parameter) {
  const auto value = session(session_id);
  const PcurveData pcurve_value =
      pcurve(*value, face_key, wire_key, coedge_position, seam_image);
  require_finite(parameter, "pcurve parameter");
  if (parameter < pcurve_value.start || parameter > pcurve_value.end) {
    throw std::runtime_error("OCCT exact pcurve parameter is outside the edge domain");
  }
  gp_Pnt2d point;
  gp_Vec2d first;
  gp_Vec2d second;
  pcurve_value.curve->D2(parameter, point, first, second);
  OcctPcurveDerivativesPayload result;
  result.range_start = pcurve_value.start;
  result.range_end = pcurve_value.end;
  result.point_u = point.X();
  result.point_v = point.Y();
  result.first_u = first.X();
  result.first_v = first.Y();
  result.second_u = second.X();
  result.second_v = second.Y();
  return result;
}

std::int8_t exact_trim_classify(std::uint64_t session_id,
                                std::uint64_t face_key,
                                double u,
                                double v,
                                double boundary_tolerance_uv) {
  require_finite(u, "trim point U");
  require_finite(v, "trim point V");
  require_finite(boundary_tolerance_uv, "trim boundary tolerance");
  if (boundary_tolerance_uv < 0.0) {
    throw std::runtime_error("OCCT exact trim boundary tolerance must be non-negative");
  }
  const auto value = session(session_id);
  const BRepClass_FaceClassifier classifier(
      face(*value, face_key), gp_Pnt2d(u, v), boundary_tolerance_uv, Standard_True);
  switch (classifier.State()) {
    case TopAbs_IN:
      return 1;
    case TopAbs_ON:
      return 0;
    case TopAbs_OUT:
      return -1;
    default:
      throw std::runtime_error("OCCT exact trim classification did not converge");
  }
}

OcctMassPropertiesPayload exact_mass_properties(std::uint64_t session_id,
                                                rust::Slice<const std::uint64_t> shape_keys,
                                                bool is_sheet_body) {
  const auto value = session(session_id);
  if (shape_keys.empty()) {
    throw std::runtime_error("OCCT exact body token contains no shapes");
  }
  GProp_GProps surface_properties;
  GProp_GProps volume_properties;
  for (const std::uint64_t shape_key : shape_keys) {
    if (shape_key == 0 || shape_key > static_cast<std::uint64_t>(value->shapes.NbShapes())) {
      throw std::runtime_error("OCCT exact body token is outside the B-rep shape table");
    }
    const TopoDS_Shape& body =
        value->shapes.Shape(static_cast<Standard_Integer>(shape_key));
    GProp_GProps surface_component;
    BRepGProp::SurfaceProperties(body, surface_component);
    surface_properties.Add(surface_component);
    if (is_sheet_body) {
      if (body.ShapeType() != TopAbs_SHELL) {
        throw std::runtime_error("OCCT exact sheet body token does not identify a shell");
      }
    } else {
      if (body.ShapeType() != TopAbs_SOLID) {
        throw std::runtime_error("OCCT exact solid body token does not identify a solid");
      }
      GProp_GProps volume_component;
      BRepGProp::VolumeProperties(body, volume_component);
      volume_properties.Add(volume_component);
    }
  }
  const double surface_area = surface_properties.Mass();
  require_finite(surface_area, "surface area");
  if (surface_area <= 0.0) {
    throw std::runtime_error("OCCT exact body surface area must be positive");
  }

  if (!is_sheet_body) {
    require_finite(volume_properties.Mass(), "volume");
    if (volume_properties.Mass() <= 0.0) {
      throw std::runtime_error("OCCT exact solid volume must be positive");
    }
  }
  const GProp_GProps& centroid_properties =
      is_sheet_body ? surface_properties : volume_properties;
  const gp_Pnt centroid = centroid_properties.CentreOfMass();
  const gp_Mat inertia = is_sheet_body ? gp_Mat() : volume_properties.MatrixOfInertia();
  const double scale = value->meters_per_source_unit;
  const double scale2 = scale * scale;
  const double scale3 = scale2 * scale;
  const double scale5 = scale3 * scale2;
  OcctMassPropertiesPayload result;
  result.volume = is_sheet_body ? 0.0 : volume_properties.Mass() * scale3;
  result.surface_area = surface_area * scale2;
  result.centroid_x = centroid.X() * scale;
  result.centroid_y = centroid.Y() * scale;
  result.centroid_z = centroid.Z() * scale;
  result.inertia_xx = is_sheet_body ? 0.0 : inertia.Value(1, 1) * scale5;
  result.inertia_yy = is_sheet_body ? 0.0 : inertia.Value(2, 2) * scale5;
  result.inertia_zz = is_sheet_body ? 0.0 : inertia.Value(3, 3) * scale5;
  result.inertia_xy = is_sheet_body ? 0.0 : inertia.Value(1, 2) * scale5;
  result.inertia_xz = is_sheet_body ? 0.0 : inertia.Value(1, 3) * scale5;
  result.inertia_yz = is_sheet_body ? 0.0 : inertia.Value(2, 3) * scale5;
  return result;
}

void close_exact_evaluator_session(std::uint64_t session_id) {
  std::lock_guard<std::mutex> lock(sessions_mutex());
  sessions().erase(session_id);
}

} // namespace occt_backend
} // namespace runmat_geometry_io
