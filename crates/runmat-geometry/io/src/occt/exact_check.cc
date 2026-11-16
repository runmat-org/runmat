// Check geometry (kernel_analyze kind 7) and the area properties of planar faces (kind 8), on
// exact OCCT geometry.
//
// Kind 7, check geometry: the problems in one body's
// topology and geometry that make later features, exports or meshes fail.
//
// Reals in: short-edge length (m), gap limit (m), strict (0 or 1).
// Reals out: four per finding, (problem, entity, index, value); value is NaN when the finding
// has none. Entities: 0 body, 1 face, 2 edge, 3 vertex; indices are the body summary's.
// Problems:
//   0 invalid body (shell or solid structure)     1 invalid face
//   2 invalid edge                                 3 invalid vertex
//   4 self-intersecting face (strict only)        5 open edge (one face on a solid)
//   6 short edge (value: length)                   7 edge gap (value: edge tolerance)
//   8 vertex gap (value: vertex tolerance)
//   9 smallest radius of curvature (value: radius; one row, the face it is on)
//  10 largest edge gap (value)                    11 largest vertex gap (value)

#include "runmat-geometry-io/src/occt/kernel_ffi.rs.h"
#include "exact_kernel.hxx"

#include <BOPAlgo_ArgumentAnalyzer.hxx>
#include <BOPAlgo_CheckResult.hxx>
#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepGProp.hxx>
#include <GProp_GProps.hxx>
#include <gp_Pln.hxx>
#include <BRepLProp_SLProps.hxx>
#include <BRepTools.hxx>
#include <BRep_Tool.hxx>
#include <GCPnts_AbscissaPoint.hxx>
#include <NCollection_List.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopoDS.hxx>

#include <algorithm>
#include <cmath>
#include <limits>

namespace runmat_geometry_io {
namespace occt_kernel {
namespace {

void push(rust::Vec<double>& out, int problem, int entity, std::size_t index, double value) {
  out.push_back(problem);
  out.push_back(entity);
  out.push_back(static_cast<double>(index));
  out.push_back(value);
}

template <typename T>
long index_of(const std::vector<T>& list, const TopoDS_Shape& s) {
  for (std::size_t i = 0; i < list.size(); ++i)
    if (list[i].IsSame(s)) return static_cast<long>(i);
  return -1;
}

// The largest |curvature| sampled over a face (0 for a plane).
double face_max_curvature(const TopoDS_Face& f) {
  BRepAdaptor_Surface s(f, Standard_True);
  if (s.GetType() == GeomAbs_Plane) return 0;
  double u0, u1, v0, v1;
  BRepTools::UVBounds(f, u0, u1, v0, v1);
  double hi = 0;
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 7; ++j) {
      BRepLProp_SLProps pr(s, u0 + (u1 - u0) * (i + 0.5) / 7, v0 + (v1 - v0) * (j + 0.5) / 7, 2, 1e-9);
      if (!pr.IsCurvatureDefined()) continue;
      hi = std::max(hi, std::max(std::abs(pr.MinCurvature()), std::abs(pr.MaxCurvature())));
    }
  return hi;
}
}  // namespace

rust::Vec<double> check_body(const Body& b, double short_edge, double gap, bool strict) {
  rust::Vec<double> out;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  BRepCheck_Analyzer analyzer(b.shape, Standard_True);
  if (!analyzer.IsValid()) {
    bool located = false;
    for (std::size_t i = 0; i < b.faces.size(); ++i)
      if (!analyzer.IsValid(b.faces[i])) { push(out, 1, 1, i, nan); located = true; }
    for (std::size_t i = 0; i < b.edges.size(); ++i)
      if (!analyzer.IsValid(b.edges[i])) { push(out, 2, 2, i, nan); located = true; }
    for (std::size_t i = 0; i < b.vertices.size(); ++i)
      if (!analyzer.IsValid(b.vertices[i])) { push(out, 3, 3, i, nan); located = true; }
    // The shell or solid itself (orientation, closure) when no face, edge or vertex is at fault.
    if (!located) push(out, 0, 0, 0, nan);
  }
  if (strict) {
    BOPAlgo_ArgumentAnalyzer self;
    self.SetShape1(b.shape);
    self.ArgumentTypeMode() = Standard_False;
    self.SelfInterMode() = Standard_True;
    self.SmallEdgeMode() = Standard_False;
    self.RebuildFaceMode() = Standard_False;
    self.TangentMode() = Standard_False;
    self.MergeVertexMode() = Standard_False;
    self.MergeEdgeMode() = Standard_False;
    self.ContinuityMode() = Standard_False;
    self.CurveOnSurfaceMode() = Standard_False;
    self.Perform();
    if (self.HasFaulty()) {
      std::vector<bool> seen(b.faces.size(), false);
      for (NCollection_List<BOPAlgo_CheckResult>::Iterator it(self.GetCheckResult()); it.More(); it.Next()) {
        if (it.Value().GetCheckStatus() != BOPAlgo_SelfIntersect) continue;
        for (NCollection_List<TopoDS_Shape>::Iterator s(it.Value().GetFaultyShapes1()); s.More(); s.Next()) {
          for (TopExp_Explorer ex(s.Value(), TopAbs_FACE); ex.More(); ex.Next()) {
            const long k = index_of(b.faces, ex.Current());
            if (k >= 0 && !seen[k]) { seen[k] = true; push(out, 4, 1, k, nan); }
          }
        }
      }
    }
  }
  TopTools_IndexedDataMapOfShapeListOfShape ef;
  TopExp::MapShapesAndUniqueAncestors(b.shape, TopAbs_EDGE, TopAbs_FACE, ef);
  double worst_edge = 0;
  long worst_edge_at = -1;
  for (std::size_t i = 0; i < b.edges.size(); ++i) {
    const TopoDS_Edge& e = b.edges[i];
    if (!b.sheet && ef.Contains(e) && ef.FindFromKey(e).Extent() < 2) push(out, 5, 2, i, nan);
    BRepAdaptor_Curve c(e);
    const double length = GCPnts_AbscissaPoint::Length(c);
    if (length < short_edge) push(out, 6, 2, i, length);
    const double t = BRep_Tool::Tolerance(e);
    if (t > gap) push(out, 7, 2, i, t);
    if (t > worst_edge) { worst_edge = t; worst_edge_at = static_cast<long>(i); }
  }
  double worst_vertex = 0;
  long worst_vertex_at = -1;
  for (std::size_t i = 0; i < b.vertices.size(); ++i) {
    const double t = BRep_Tool::Tolerance(b.vertices[i]);
    if (t > gap) push(out, 8, 3, i, t);
    if (t > worst_vertex) { worst_vertex = t; worst_vertex_at = static_cast<long>(i); }
  }
  double k = 0;
  long k_at = -1;
  for (std::size_t i = 0; i < b.faces.size(); ++i) {
    const double ki = face_max_curvature(b.faces[i]);
    if (ki > k) { k = ki; k_at = static_cast<long>(i); }
  }
  if (k_at >= 0) push(out, 9, 1, k_at, 1.0 / k);
  if (worst_edge_at >= 0) push(out, 10, 2, worst_edge_at, worst_edge);
  if (worst_vertex_at >= 0) push(out, 11, 3, worst_vertex_at, worst_vertex);
  return out;
}

// Kind 8: faces `faces` of `b`, which must be planar and in one plane, as one region:
// area, centroid (3), the inertia matrix about the centroid for unit density (9, rows), and the
// plane's normal (3, the first face's outward normal).
rust::Vec<double> face_area_properties(const Body& b, rust::Slice<const double> faces) {
  if (faces.empty()) fail("pick a planar face");
  GProp_GProps all;
  gp_Pln first;
  bool have = false;
  for (double x : faces) {
    const long i = static_cast<long>(x);
    if (i < 0 || static_cast<std::size_t>(i) >= b.faces.size()) fail("the face isn't on the body");
    const TopoDS_Face& f = b.faces[i];
    BRepAdaptor_Surface s(f, Standard_True);
    if (s.GetType() != GeomAbs_Plane) fail("section properties need a planar face");
    gp_Pln pl = s.Plane();
    if (f.Orientation() == TopAbs_REVERSED) pl.SetAxis(pl.Axis().Reversed());
    if (!have) {
      first = pl;
      have = true;
    } else if (!first.Axis().Direction().IsParallel(pl.Axis().Direction(), 1e-9) ||
               first.Distance(pl.Location()) > 1e-9) {
      fail("the faces aren't in one plane");
    }
    GProp_GProps g;
    BRepGProp::SurfaceProperties(f, g);
    all.Add(g);
  }
  rust::Vec<double> out;
  const gp_Pnt c = all.CentreOfMass();
  const gp_Mat m = all.MatrixOfInertia();
  out.push_back(all.Mass());
  out.push_back(c.X()); out.push_back(c.Y()); out.push_back(c.Z());
  for (int i = 1; i <= 3; ++i)
    for (int j = 1; j <= 3; ++j) out.push_back(m(i, j));
  const gp_Dir n = first.Axis().Direction();
  out.push_back(n.X()); out.push_back(n.Y()); out.push_back(n.Z());
  return out;
}

}  // namespace occt_kernel
}  // namespace runmat_geometry_io
