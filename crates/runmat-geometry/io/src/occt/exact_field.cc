// Values over a body's faces at given points (kernel_analyze kind 9), on exact OCCT geometry:
// what thickness, curvature and zebra color the model by.
//
// Reals in: quantity (0 thickness, 1 curvature, 2 normal only), then for each face: the face's
// index in the body summary, the point count n, and n points (x, y, z; metres). Points are near
// the face (a display tessellation's vertices); each is projected onto the face's surface.
// Reals out: eight per point, in order: the projected point (3), the outward unit normal there
// (3), and two values:
//   thickness: the distance from the point into the body along the inward normal to the next
//     face (m), and NaN;
//   curvature: the two principal curvatures (1/m), smaller first, positive where the face is
//     convex (bends away from its outward normal);
//   normal: NaN, NaN.
// A point off the trimmed face (a mesh chord across a hole's edge) is drawn at the nearest point
// of the face's boundary and measured just inside it. A point whose projection fails gives NaN
// for all eight.

#include "runmat-geometry-io/src/occt/kernel_ffi.rs.h"
#include "exact_kernel.hxx"

#include <BRepAdaptor_Surface.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepClass_FaceClassifier.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepBndLib.hxx>
#include <BRepLProp_SLProps.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <GeomAPI_ProjectPointOnSurf.hxx>
#include <Geom_Surface.hxx>
#include <IntCurvesFace_ShapeIntersector.hxx>
#include <gp_Lin.hxx>
#include <gp_Pnt2d.hxx>
#include <gp_Vec.hxx>

#include <algorithm>
#include <cmath>
#include <limits>

namespace runmat_geometry_io {
namespace occt_kernel {

rust::Vec<double> surface_field(const Body& b, rust::Slice<const double> r) {
  if (r.empty()) fail("a surface field needs a quantity");
  const int quantity = static_cast<int>(r[0]);
  if (quantity < 0 || quantity > 2) fail("unknown surface field quantity");
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Bnd_Box box;
  BRepBndLib::Add(b.shape, box);
  const double size = box.IsVoid() ? 1.0 : std::sqrt(box.SquareExtent());
  // Hits closer than this are the face the ray starts on.
  const double eps = size * 1e-7;
  IntCurvesFace_ShapeIntersector rays;
  if (quantity == 0) rays.Load(b.shape, 1e-9);
  rust::Vec<double> out;
  std::size_t at = 1;
  while (at < r.size()) {
    if (at + 2 > r.size()) fail("a surface field's face list is short");
    const long f = static_cast<long>(r[at]);
    const std::size_t n = static_cast<std::size_t>(r[at + 1]);
    at += 2;
    if (f < 0 || static_cast<std::size_t>(f) >= b.faces.size()) fail("the face isn't on the body");
    if (at + 3 * n > r.size()) fail("a surface field's point list is short");
    const TopoDS_Face& face = b.faces[static_cast<std::size_t>(f)];
    const bool reversed = face.Orientation() == TopAbs_REVERSED;
    Handle(Geom_Surface) surface = BRep_Tool::Surface(face);
    BRepAdaptor_Surface adaptor(face, Standard_True);
    const bool plane = adaptor.GetType() == GeomAbs_Plane;
    for (std::size_t i = 0; i < n; ++i, at += 3) {
      const gp_Pnt p(r[at], r[at + 1], r[at + 2]);
      double u = 0, v = 0;
      bool found = false;
      if (!surface.IsNull()) {
        GeomAPI_ProjectPointOnSurf proj(p, surface);
        if (proj.NbPoints() > 0) {
          proj.LowerDistanceParameters(u, v);
          found = true;
        }
      }
      if (!found) {
        for (int k = 0; k < 8; ++k) out.push_back(nan);
        continue;
      }
      // A display mesh's chord across a curved boundary (a hole's edge) leaves points off the
      // trimmed face. Such a point is drawn on the face's boundary, nearest it, and measured a
      // step further onto the face, where the face's own value is.
      bool on_boundary = false;
      gp_Pnt boundary;
      {
        BRepClass_FaceClassifier cls(face, gp_Pnt2d(u, v), 1e-7);
        if (cls.State() == TopAbs_OUT) {
          BRepExtrema_DistShapeShape near(BRepBuilderAPI_MakeVertex(p).Vertex(), face);
          if (near.IsDone() && near.NbSolution() > 0) {
            boundary = near.PointOnShape2(1);
            gp_Pnt off;
            surface->D0(u, v, off);
            gp_Vec d(off, boundary);
            if (d.Magnitude() > 1e-12) {
              const gp_Pnt inner = boundary.Translated(d.Normalized() * (size * 1e-4));
              GeomAPI_ProjectPointOnSurf again(inner, surface);
              if (again.NbPoints() > 0) {
                again.LowerDistanceParameters(u, v);
                on_boundary = true;
              }
            }
          }
        }
      }
      BRepLProp_SLProps props(adaptor, u, v, quantity == 1 ? 2 : 1, 1e-9);
      if (!props.IsNormalDefined()) {
        for (int k = 0; k < 8; ++k) out.push_back(nan);
        continue;
      }
      const gp_Pnt q = props.Value();
      gp_Dir normal = props.Normal();
      if (reversed) normal.Reverse();
      const gp_Pnt drawn = on_boundary ? boundary : q;
      out.push_back(drawn.X()); out.push_back(drawn.Y()); out.push_back(drawn.Z());
      out.push_back(normal.X()); out.push_back(normal.Y()); out.push_back(normal.Z());
      double a = nan, c = nan;
      if (quantity == 0) {
        rays.Perform(gp_Lin(q, normal.Reversed()), eps, 1e12);
        if (rays.IsDone()) {
          double best = std::numeric_limits<double>::infinity();
          for (int k = 1; k <= rays.NbPnt(); ++k) best = std::min(best, rays.WParameter(k));
          if (std::isfinite(best)) a = best;
        }
      } else if (quantity == 1) {
        if (plane) {
          a = c = 0;
        } else if (props.IsCurvatureDefined()) {
          // OCCT's curvatures are positive where the surface bends toward its own normal
          // (D1U ^ D1V); toward the outward normal is concave, so convex reads positive after
          // the sign flips (and flips back on a reversed face).
          const double s = reversed ? 1.0 : -1.0;
          const double k1 = s * props.MinCurvature(), k2 = s * props.MaxCurvature();
          a = std::min(k1, k2);
          c = std::max(k1, k2);
        }
      }
      out.push_back(a);
      out.push_back(c);
    }
  }
  return out;
}

}  // namespace occt_kernel
}  // namespace runmat_geometry_io
