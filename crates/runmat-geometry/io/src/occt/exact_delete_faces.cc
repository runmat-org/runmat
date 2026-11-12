// Delete faces, healed, for K1 (direct.delete_faces with heal).
//
// The picked faces are removed and their neighbours extended to close the body
// (BRepAlgoAPI_Defeaturing). A curved surface split into several faces counts as one: a hole
// whose wall is two half cylinders (as STEP files often write it, and as a cut through two arcs
// builds it) is removed whole when one of its faces is picked. Defeaturing can't lift a part of
// a curved surface alone, and when it can't remove a feature it warns and returns the body
// unchanged; that is reported as a failure, never as a delete that did nothing.

#include "runmat-geometry-io/src/occt/kernel_ffi.rs.h"
#include "exact_kernel.hxx"

#include <BOPAlgo_Alerts.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepAlgoAPI_Defeaturing.hxx>
#include <GeomAbs_SurfaceType.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopTools_ListIteratorOfListOfShape.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopoDS.hxx>
#include <gp_Ax1.hxx>
#include <gp_Cone.hxx>
#include <gp_Cylinder.hxx>
#include <gp_Lin.hxx>
#include <gp_Sphere.hxx>
#include <gp_Torus.hxx>

#include <cmath>
#include <sstream>
#include <vector>

namespace runmat_geometry_io {
namespace occt_kernel {

namespace {

constexpr double kLength = 1e-6;  // model units (m): a micrometre
constexpr double kAngle = 1e-6;   // radians

bool same_axis(const gp_Ax1& a, const gp_Ax1& b) {
  if (!a.Direction().IsParallel(b.Direction(), kAngle)) return false;
  // b's location on a's line.
  return gp_Lin(a).Distance(b.Location()) < kLength;
}

// Whether two faces lie on one curved surface: the same cylinder, cone, sphere or torus.
// Planes and free-form surfaces are never merged: a split plane is two faces to the user too.
bool same_curved_surface(const TopoDS_Face& f, const TopoDS_Face& g) {
  BRepAdaptor_Surface a(f, Standard_False);
  BRepAdaptor_Surface b(g, Standard_False);
  if (a.GetType() != b.GetType()) return false;
  switch (a.GetType()) {
    case GeomAbs_Cylinder: {
      const gp_Cylinder p = a.Cylinder(), q = b.Cylinder();
      return std::abs(p.Radius() - q.Radius()) < kLength && same_axis(p.Axis(), q.Axis());
    }
    case GeomAbs_Cone: {
      const gp_Cone p = a.Cone(), q = b.Cone();
      return std::abs(p.SemiAngle() - q.SemiAngle()) < kAngle &&
             p.Apex().Distance(q.Apex()) < kLength && same_axis(p.Axis(), q.Axis());
    }
    case GeomAbs_Sphere: {
      const gp_Sphere p = a.Sphere(), q = b.Sphere();
      return std::abs(p.Radius() - q.Radius()) < kLength &&
             p.Location().Distance(q.Location()) < kLength;
    }
    case GeomAbs_Torus: {
      const gp_Torus p = a.Torus(), q = b.Torus();
      return std::abs(p.MajorRadius() - q.MajorRadius()) < kLength &&
             std::abs(p.MinorRadius() - q.MinorRadius()) < kLength &&
             p.Location().Distance(q.Location()) < kLength && same_axis(p.Axis(), q.Axis());
    }
    default:
      return false;
  }
}

// The picked faces, each joined by the faces next to it on the same curved surface.
std::vector<TopoDS_Face> whole_surfaces(const TopoDS_Shape& body, const std::vector<TopoDS_Face>& picked) {
  TopTools_IndexedDataMapOfShapeListOfShape faces_of_edge;
  TopExp::MapShapesAndAncestors(body, TopAbs_EDGE, TopAbs_FACE, faces_of_edge);
  TopTools_IndexedMapOfShape taken;
  std::vector<TopoDS_Face> out;
  std::vector<TopoDS_Face> queue;
  for (const TopoDS_Face& f : picked) {
    if (taken.Contains(f)) continue;
    taken.Add(f);
    out.push_back(f);
    queue.push_back(f);
  }
  while (!queue.empty()) {
    const TopoDS_Face f = queue.back();
    queue.pop_back();
    for (TopExp_Explorer e(f, TopAbs_EDGE); e.More(); e.Next()) {
      const int i = faces_of_edge.FindIndex(e.Current());
      if (i == 0) continue;
      for (TopTools_ListIteratorOfListOfShape it(faces_of_edge.FindFromIndex(i)); it.More(); it.Next()) {
        const TopoDS_Face g = TopoDS::Face(it.Value());
        if (taken.Contains(g) || !same_curved_surface(f, g)) continue;
        taken.Add(g);
        out.push_back(g);
        queue.push_back(g);
      }
    }
  }
  return out;
}

}  // namespace

Tracked delete_faces_healed(const Tracked& src, const std::vector<TopoDS_Face>& picked) {
  BRepAlgoAPI_Defeaturing df;
  df.SetShape(src.shape);
  for (const TopoDS_Face& f : whole_surfaces(src.shape, picked)) df.AddFaceToRemove(f);
  df.SetRunParallel(Standard_False);
  df.SetToFillHistory(Standard_True);
  df.Build();
  if (!df.IsDone() || df.HasErrors()) {
    std::ostringstream m;
    df.DumpErrors(m);
    fail("the faces could not be removed and healed: " + m.str());
  }
  if (df.HasWarning(STANDARD_TYPE(BOPAlgo_AlertUnableToRemoveTheFeature)))
    fail("the faces don't bound a feature the neighbouring faces can close over; pick every "
         "face of the hole, boss or pocket");
  Tracked out;
  out.shape = df.Shape();
  propagate_history(src, df.History(), out);
  return out;
}

}  // namespace occt_kernel
}  // namespace runmat_geometry_io
