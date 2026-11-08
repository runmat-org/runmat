#include "runmat-geometry-io/src/occt/ffi.rs.h"
#include "occt_bridge.hxx"
#include "exact_evaluator_internal.hxx"

#include <BRepAdaptor_Surface.hxx>
#include <BRep_Tool.hxx>
#include <Extrema_ExtPS.hxx>
#include <GeomAPI_ProjectPointOnCurve.hxx>
#include <Geom_Surface.hxx>
#include <Precision.hxx>
#include <gp_Pnt.hxx>
#include <gp_Vec.hxx>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace runmat_geometry_io {
namespace occt_backend {
namespace {

void require_finite(double value, const char* role) {
  if (!std::isfinite(value)) {
    throw std::runtime_error(std::string("OCCT exact surface ") + role + " must be finite");
  }
}

void require_bounds(const BRepAdaptor_Surface& surface) {
  require_finite(surface.FirstUParameter(), "U range start");
  require_finite(surface.LastUParameter(), "U range end");
  require_finite(surface.FirstVParameter(), "V range start");
  require_finite(surface.LastVParameter(), "V range end");
  if (surface.FirstUParameter() > surface.LastUParameter() ||
      surface.FirstVParameter() > surface.LastVParameter()) {
    throw std::runtime_error("OCCT exact surface has a reversed parameter domain");
  }
}

double admitted_parameter(double value,
                          double start,
                          double end,
                          bool periodic,
                          double period,
                          const char* role) {
  require_finite(value, role);
  if (periodic && (!std::isfinite(period) || period <= 0.0)) {
    throw std::runtime_error("OCCT exact surface has an invalid period");
  }
  const double tolerance = Precision::PConfusion() *
                           std::max({1.0, std::abs(start), std::abs(end), std::abs(value)});
  if ((value < start - tolerance || value > end + tolerance) && periodic) {
    double offset = std::fmod(value - start, period);
    if (offset < 0.0) {
      offset += period;
    }
    value = start + offset;
  }
  if (value < start - tolerance || value > end + tolerance) {
    std::ostringstream message;
    message.precision(17);
    message << "OCCT exact surface parameter is outside the face domain: " << role << " "
            << value << " outside [" << start << ", " << end << "]"
            << (periodic ? " (periodic)" : "");
    throw std::runtime_error(message.str());
  }
  return std::clamp(value, start, end);
}

// The cached bounds pad a face's trimmed domain by UResolution of its largest edge tolerance.
// UResolution is a global estimate from the surface's largest parametric speed, so near a slow
// corner a pcurve can run past that padding while its surface point stays inside the edge's
// tolerance tube (an exported EV battery pack: u = 1 + 6e-9 against 1 + 1.1e-9 padding). Clamp a
// non-periodic parameter onto the domain when the surface point it names lies within the face's
// largest edge tolerance (times the 1.1 sampling allowance of incidence validation) of the
// clamped point. Import admits each edge's measured curve/pcurve gap into that tolerance, so it
// is the one bound. Clamping only keeps evaluation on the trimmed surface: incidence validation
// still holds the clamped point to the edge tolerance. Larger overshoots stay outside and are
// rejected by admitted_parameter.
void admit_tolerant_overshoot(const BRepAdaptor_Surface& surface,
                              const TopoDS_Face& face,
                              const std::array<double, 4>& bounds,
                              double& u,
                              double& v) {
  const bool u_out = !surface.IsUPeriodic() && std::isfinite(u) && (u < bounds[0] || u > bounds[1]);
  const bool v_out = !surface.IsVPeriodic() && std::isfinite(v) && (v < bounds[2] || v > bounds[3]);
  if (!u_out && !v_out) {
    return;
  }
  const double tolerance = 1.1 * BRep_Tool::MaxTolerance(face, TopAbs_EDGE);
  if (!std::isfinite(tolerance) || tolerance <= 0.0) {
    return;
  }
  const double clamped_u = u_out ? std::clamp(u, bounds[0], bounds[1]) : u;
  const double clamped_v = v_out ? std::clamp(v, bounds[2], bounds[3]) : v;
  const gp_Pnt requested = surface.Value(u, v);
  const gp_Pnt clamped = surface.Value(clamped_u, clamped_v);
  if (requested.Distance(clamped) <= tolerance) {
    u = clamped_u;
    v = clamped_v;
  }
}

double parameter_tolerance(double resolution) {
  require_finite(resolution, "projection parameter tolerance");
  if (resolution <= 0.0) {
    throw std::runtime_error("OCCT exact surface projection parameter tolerance must be positive");
  }
  return std::max(resolution, Precision::PConfusion());
}

struct ProjectionCandidate {
  bool present = false;
  double squared_distance = std::numeric_limits<double>::infinity();
  double u = 0.0;
  double v = 0.0;
  gp_Pnt point;
};

void consider(ProjectionCandidate& best,
              const gp_Pnt& query,
              double u,
              double v,
              const gp_Pnt& point) {
  const double squared_distance = query.SquareDistance(point);
  if (!std::isfinite(u) || !std::isfinite(v) || !std::isfinite(squared_distance)) {
    return;
  }
  if (!best.present || squared_distance < best.squared_distance ||
      (squared_distance == best.squared_distance &&
       (u < best.u || (u == best.u && v < best.v)))) {
    best = {true, squared_distance, u, v, point};
  }
}

void consider_u_boundary(ProjectionCandidate& best,
                         const gp_Pnt& query,
                         const Handle(Geom_Surface)& surface,
                         double u,
                         double v_start,
                         double v_end) {
  GeomAPI_ProjectPointOnCurve projection(query, surface->UIso(u), v_start, v_end);
  for (Standard_Integer index = 1; index <= projection.NbPoints(); ++index) {
    consider(best, query, u, projection.Parameter(index), projection.Point(index));
  }
}

void consider_v_boundary(ProjectionCandidate& best,
                         const gp_Pnt& query,
                         const Handle(Geom_Surface)& surface,
                         double v,
                         double u_start,
                         double u_end) {
  GeomAPI_ProjectPointOnCurve projection(query, surface->VIso(v), u_start, u_end);
  for (Standard_Integer index = 1; index <= projection.NbPoints(); ++index) {
    consider(best, query, projection.Parameter(index), v, projection.Point(index));
  }
}

} // namespace

OcctSurfacePropertiesPayload exact_surface_properties(std::uint64_t session_id,
                                                      std::uint64_t face_key) {
  const detail::ExactFaceContext context = detail::exact_face_context(session_id, face_key);
  const BRepAdaptor_Surface surface(context.face, Standard_True);
  require_bounds(surface);
  OcctSurfacePropertiesPayload result;
  result.u_start = surface.FirstUParameter();
  result.u_end = surface.LastUParameter();
  result.v_start = surface.FirstVParameter();
  result.v_end = surface.LastVParameter();
  result.u_periodic = surface.IsUPeriodic();
  result.u_period = result.u_periodic ? surface.UPeriod() : 0.0;
  result.v_periodic = surface.IsVPeriodic();
  result.v_period = result.v_periodic ? surface.VPeriod() : 0.0;
  if ((result.u_periodic && (!std::isfinite(result.u_period) || result.u_period <= 0.0)) ||
      (result.v_periodic && (!std::isfinite(result.v_period) || result.v_period <= 0.0))) {
    throw std::runtime_error("OCCT exact surface has an invalid period");
  }
  return result;
}

OcctSurfaceDerivativesPayload exact_surface_derivatives(std::uint64_t session_id,
                                                        std::uint64_t face_key,
                                                        double u,
                                                        double v) {
  const detail::ExactFaceContext context = detail::exact_face_context(session_id, face_key);
  // Unrestricted adaptation avoids recomputing the trimmed wire domain. Admit
  // parameters against the identical bounds captured with this session face.
  const BRepAdaptor_Surface surface(context.face, Standard_False);
  const auto& bounds = context.parameter_bounds;
  for (const double bound : bounds) require_finite(bound, "cached parameter bound");
  if (bounds[0] > bounds[1] || bounds[2] > bounds[3]) {
    throw std::runtime_error("OCCT exact surface has a reversed parameter domain");
  }
  admit_tolerant_overshoot(surface, context.face, bounds, u, v);
  u = admitted_parameter(u, bounds[0], bounds[1], surface.IsUPeriodic(),
                         surface.IsUPeriodic() ? surface.UPeriod() : 0.0, "U parameter");
  v = admitted_parameter(v, bounds[2], bounds[3], surface.IsVPeriodic(),
                         surface.IsVPeriodic() ? surface.VPeriod() : 0.0, "V parameter");
  gp_Pnt point;
  gp_Vec du;
  gp_Vec dv;
  gp_Vec duu;
  gp_Vec dvv;
  gp_Vec duv;
  surface.D2(u, v, point, du, dv, duu, dvv, duv);
  const double scale = context.meters_per_source_unit;
  OcctSurfaceDerivativesPayload result;
  result.parameter_scale = scale;
  result.u_parameter_unit = 0;
  result.v_parameter_unit = 0;
  switch (surface.GetType()) {
    case GeomAbs_Plane: result.u_parameter_unit = 1; result.v_parameter_unit = 1; break;
    case GeomAbs_Cylinder: result.u_parameter_unit = 2; result.v_parameter_unit = 1; break;
    case GeomAbs_Sphere:
    case GeomAbs_Torus: result.u_parameter_unit = 2; result.v_parameter_unit = 2; break;
    default: break; // Retain other parameterizations as named native axes.
  }
  result.point_x = point.X() * scale;
  result.point_y = point.Y() * scale;
  result.point_z = point.Z() * scale;
  result.du_x = du.X() * scale;
  result.du_y = du.Y() * scale;
  result.du_z = du.Z() * scale;
  result.dv_x = dv.X() * scale;
  result.dv_y = dv.Y() * scale;
  result.dv_z = dv.Z() * scale;
  result.duu_x = duu.X() * scale;
  result.duu_y = duu.Y() * scale;
  result.duu_z = duu.Z() * scale;
  result.duv_x = duv.X() * scale;
  result.duv_y = duv.Y() * scale;
  result.duv_z = duv.Z() * scale;
  result.dvv_x = dvv.X() * scale;
  result.dvv_y = dvv.Y() * scale;
  result.dvv_z = dvv.Z() * scale;
  return result;
}

OcctSurfaceProjectionPayload exact_surface_closest_point(
    std::uint64_t session_id,
    std::uint64_t face_key,
    rust::Slice<const double> point_m,
    double absolute_error_m) {
  if (point_m.size() != 3) {
    throw std::runtime_error("OCCT exact surface projection requires one 3D point");
  }
  for (double coordinate : point_m) {
    require_finite(coordinate, "projection point");
  }
  require_finite(absolute_error_m, "projection tolerance");
  if (absolute_error_m <= 0.0) {
    throw std::runtime_error("OCCT exact surface projection tolerance must be positive");
  }

  const detail::ExactFaceContext context = detail::exact_face_context(session_id, face_key);
  const BRepAdaptor_Surface surface(context.face, Standard_True);
  require_bounds(surface);
  const double scale = context.meters_per_source_unit;
  const gp_Pnt query(point_m[0] / scale, point_m[1] / scale, point_m[2] / scale);
  const double source_tolerance = absolute_error_m / scale;
  require_finite(source_tolerance, "projection source tolerance");
  const double tolerance_u = parameter_tolerance(surface.UResolution(source_tolerance));
  const double tolerance_v = parameter_tolerance(surface.VResolution(source_tolerance));
  ProjectionCandidate best;

  const Extrema_ExtPS extrema(query,
                              surface,
                              surface.FirstUParameter(),
                              surface.LastUParameter(),
                              surface.FirstVParameter(),
                              surface.LastVParameter(),
                              tolerance_u,
                              tolerance_v,
                              Extrema_ExtFlag_MIN);
  if (extrema.IsDone()) {
    for (Standard_Integer index = 1; index <= extrema.NbExt(); ++index) {
      double u = 0.0;
      double v = 0.0;
      extrema.Point(index).Parameter(u, v);
      consider(best, query, u, v, extrema.Point(index).Value());
    }
  }

  const Handle(Geom_Surface) geometry = BRep_Tool::Surface(context.face);
  if (geometry.IsNull()) {
    throw std::runtime_error("OCCT exact surface geometry is unavailable");
  }
  const double u_start = surface.FirstUParameter();
  const double u_end = surface.LastUParameter();
  const double v_start = surface.FirstVParameter();
  const double v_end = surface.LastVParameter();
  consider_u_boundary(best, query, geometry, u_start, v_start, v_end);
  consider_u_boundary(best, query, geometry, u_end, v_start, v_end);
  consider_v_boundary(best, query, geometry, v_start, u_start, u_end);
  consider_v_boundary(best, query, geometry, v_end, u_start, u_end);
  consider(best, query, u_start, v_start, surface.Value(u_start, v_start));
  consider(best, query, u_start, v_end, surface.Value(u_start, v_end));
  consider(best, query, u_end, v_start, surface.Value(u_end, v_start));
  consider(best, query, u_end, v_end, surface.Value(u_end, v_end));
  if (!best.present) {
    throw std::runtime_error("OCCT exact surface projection did not converge");
  }

  OcctSurfaceProjectionPayload result;
  result.parameter_scale = scale;
  result.u_parameter_unit = 0;
  result.v_parameter_unit = 0;
  switch (surface.GetType()) {
    case GeomAbs_Plane: result.u_parameter_unit = 1; result.v_parameter_unit = 1; break;
    case GeomAbs_Cylinder: result.u_parameter_unit = 2; result.v_parameter_unit = 1; break;
    case GeomAbs_Sphere:
    case GeomAbs_Torus: result.u_parameter_unit = 2; result.v_parameter_unit = 2; break;
    default: break;
  }
  result.u = best.u;
  result.v = best.v;
  result.point_x = best.point.X() * scale;
  result.point_y = best.point.Y() * scale;
  result.point_z = best.point.Z() * scale;
  result.distance = std::sqrt(best.squared_distance) * scale;
  return result;
}

} // namespace occt_backend
} // namespace runmat_geometry_io
