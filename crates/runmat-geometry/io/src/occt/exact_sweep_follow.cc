// Sweeps whose profile follows the path: a pipe along one spine wire, the profile keeping its
// angle to the path through arcs and mitred at the corners between straight segments.
//
// The spine has one edge per path curve: two points are a line, three or more on one circle
// are an arc, any other samples a smooth curve through them. A planar path sweeps with its
// plane's normal as a fixed binormal, so the profile turns only in that plane; a non-planar
// path uses the corrected Frenet frame. Each side face takes the labels of the profile edge
// that made it, from the profile copy of the path curve the face runs along (the copies carry
// the path curve as their ordinal); the caps take the start and end labels.

#include "runmat-geometry-io/src/occt/kernel_ffi.rs.h"
#include "exact_kernel.hxx"

#include <BRepAdaptor_Surface.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepOffsetAPI_MakePipeShell.hxx>
#include <BRepTools.hxx>
#include <GC_MakeArcOfCircle.hxx>
#include <GeomAPI_Interpolate.hxx>
#include <Geom_Circle.hxx>
#include <Geom_TrimmedCurve.hxx>
#include <TColgp_HArray1OfPnt.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Wire.hxx>

#include <cmath>
#include <limits>

namespace runmat_geometry_io {
namespace occt_kernel {
namespace {

// The path's extent, for relative tolerances.
double extent_of(const std::vector<gp_Pnt>& pts) {
  double d = 0.0;
  for (const gp_Pnt& p : pts) d = std::max(d, p.Distance(pts.front()));
  return std::max(d, 1e-9);
}

// One spine edge through path[i0..=i1].
TopoDS_Edge spine_edge(const std::vector<gp_Pnt>& path, std::size_t i0, std::size_t i1, double tol) {
  if (i1 <= i0) fail("a sweep path curve has no length");
  if (i1 == i0 + 1) {
    if (path[i0].Distance(path[i1]) < tol) fail("a sweep path segment has no length");
    return BRepBuilderAPI_MakeEdge(path[i0], path[i1]).Edge();
  }
  const std::size_t mid = (i0 + i1) / 2;
  GC_MakeArcOfCircle arc(path[i0], path[mid], path[i1]);
  if (arc.IsDone()) {
    const Handle(Geom_TrimmedCurve)& c = arc.Value();
    Handle(Geom_Circle) circle = Handle(Geom_Circle)::DownCast(c->BasisCurve());
    bool on = !circle.IsNull();
    for (std::size_t k = i0; on && k <= i1; ++k)
      on = std::abs(path[k].Distance(circle->Location()) - circle->Radius()) < tol;
    if (on) return BRepBuilderAPI_MakeEdge(c).Edge();
  }
  Handle(TColgp_HArray1OfPnt) pts = new TColgp_HArray1OfPnt(1, static_cast<int>(i1 - i0 + 1));
  for (std::size_t k = i0; k <= i1; ++k) pts->SetValue(static_cast<int>(k - i0 + 1), path[k]);
  GeomAPI_Interpolate interp(pts, Standard_False, 1e-9);
  interp.Perform();
  if (!interp.IsDone()) fail("the sweep path cannot be fitted");
  return BRepBuilderAPI_MakeEdge(interp.Curve()).Edge();
}

// The normal of the plane holding every path point; false when the points are collinear or
// leave every plane.
bool path_plane(const std::vector<gp_Pnt>& path, double tol, gp_Dir& normal) {
  gp_Vec best;
  for (std::size_t i = 1; i < path.size(); ++i) {
    const gp_Vec a(path.front(), path[i]);
    for (std::size_t j = i + 1; j < path.size(); ++j) {
      const gp_Vec n = a.Crossed(gp_Vec(path.front(), path[j]));
      if (n.Magnitude() > best.Magnitude()) best = n;
    }
  }
  const double ext = extent_of(path);
  if (best.Magnitude() < 1e-10 * ext * ext) return false;
  normal = gp_Dir(best);
  for (const gp_Pnt& p : path)
    if (std::abs(gp_Vec(path.front(), p).Dot(gp_Vec(normal))) > tol) return false;
  return true;
}

// Each profile copy's edge origins, loop by loop in wire order.
std::vector<std::vector<Origins>> loop_origins(const Profile& p) {
  std::vector<std::vector<Origins>> out;
  for (TopExp_Explorer w(p.tracked.shape, TopAbs_WIRE); w.More(); w.Next()) {
    std::vector<Origins> edges;
    for (TopExp_Explorer e(w.Current(), TopAbs_EDGE); e.More(); e.Next())
      edges.push_back(origins_of(p.tracked.edges, e.Current()));
    out.push_back(std::move(edges));
  }
  return out;
}

// The spine edge a swept face runs along: the one nearest the middle of its surface.
std::size_t nearest_spine_edge(const TopoDS_Face& f, const std::vector<TopoDS_Edge>& spine) {
  if (spine.size() == 1) return 0;
  double u0, u1, v0, v1;
  BRepTools::UVBounds(f, u0, u1, v0, v1);
  const gp_Pnt mid = BRepAdaptor_Surface(f).Value(0.5 * (u0 + u1), 0.5 * (v0 + v1));
  const TopoDS_Vertex at = BRepBuilderAPI_MakeVertex(mid).Vertex();
  std::size_t best = 0;
  double best_d = std::numeric_limits<double>::infinity();
  for (std::size_t k = 0; k < spine.size(); ++k) {
    BRepExtrema_DistShapeShape d(at, spine[k]);
    if (d.IsDone() && d.Value() < best_d) {
      best_d = d.Value();
      best = k;
    }
  }
  return best;
}
}  // namespace

Tracked build_sweep_follow(const std::vector<const Profile*>& copies, const std::vector<gp_Pnt>& path,
                           const std::vector<std::size_t>& curve_ends, std::uint32_t start_label,
                           std::uint32_t end_label) {
  if (path.size() < 2) fail("a sweep path needs two points");
  if (copies.empty() || copies.size() != curve_ends.size()) fail("a sweep needs one profile per path curve");
  const Profile& p = *copies.front();
  if (!p.closed) fail("a sweep that follows its path needs a closed profile");
  const double tol = 1e-9 * extent_of(path);
  std::vector<TopoDS_Edge> spine;
  BRepBuilderAPI_MakeWire wire;
  std::size_t i0 = 0;
  for (std::size_t end : curve_ends) {
    if (end >= path.size() || end <= i0) fail("a sweep path curve is out of range");
    spine.push_back(spine_edge(path, i0, end, tol));
    wire.Add(spine.back());
    if (!wire.IsDone()) fail("the sweep path is not one connected chain");
    i0 = end;
  }
  const TopoDS_Wire spine_wire = wire.Wire();
  gp_Dir binormal;
  const bool planar = path_plane(path, tol, binormal);
  std::vector<std::vector<std::vector<Origins>>> labels;
  for (const Profile* c : copies) labels.push_back(loop_origins(*c));

  std::vector<TopoDS_Wire> loops;
  for (TopExp_Explorer ex(p.tracked.shape, TopAbs_WIRE); ex.More(); ex.Next())
    loops.push_back(TopoDS::Wire(ex.Current()));
  std::vector<Tracked> pieces;
  for (std::size_t li = 0; li < loops.size(); ++li) {
    BRepOffsetAPI_MakePipeShell mk(spine_wire);
    if (planar)
      mk.SetMode(binormal);
    else
      mk.SetMode(Standard_False);  // corrected Frenet
    mk.SetTransitionMode(BRepBuilderAPI_RightCorner);
    mk.Add(loops[li], Standard_False, Standard_False);
    mk.Build();
    if (!mk.IsDone()) fail("the sweep could not be built along its path");
    if (!mk.MakeSolid()) fail("the swept profile does not close a solid");
    Tracked t;
    t.shape = mk.Shape();
    TopTools_IndexedMapOfShape present;
    TopExp::MapShapes(t.shape, TopAbs_FACE, present);
    std::size_t ei = 0;
    for (TopExp_Explorer ex(loops[li], TopAbs_EDGE); ex.More(); ex.Next(), ++ei) {
      for (const TopoDS_Shape& g : mk.Generated(ex.Current()))
        for (TopExp_Explorer fx(g, TopAbs_FACE); fx.More(); fx.Next()) {
          if (!present.Contains(fx.Current())) continue;
          const std::size_t k = nearest_spine_edge(TopoDS::Face(fx.Current()), spine);
          const auto& per_loop = labels[k];
          if (li < per_loop.size() && ei < per_loop[li].size())
            add_origins(t.faces, fx.Current(), per_loop[li][ei]);
        }
    }
    for (TopExp_Explorer ex(mk.FirstShape(), TopAbs_FACE); ex.More(); ex.Next())
      if (start_label != kNoLabel) add_origins(t.faces, ex.Current(), label(start_label));
    for (TopExp_Explorer ex(mk.LastShape(), TopAbs_FACE); ex.More(); ex.Next())
      if (end_label != kNoLabel) add_origins(t.faces, ex.Current(), label(end_label));
    pieces.push_back(std::move(t));
  }
  if (pieces.size() == 1) return pieces.front();
  std::vector<const Tracked*> holes;
  for (std::size_t i = 1; i < pieces.size(); ++i) holes.push_back(&pieces[i]);
  return boolean_op(1, {&pieces[0]}, holes);
}

}  // namespace occt_kernel
}  // namespace runmat_geometry_io
