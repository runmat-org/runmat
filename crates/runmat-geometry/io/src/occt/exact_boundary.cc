// Boundary surfaces for K1 (op 44): a smooth sheet through edges in one direction, or a patch
// bounded by the outer edges of two directions that passes through the inner ones.
//
// The patch face takes the feature's face label. A sheet edge lying on an input edge takes that
// edge's boundary label; the two side edges of a one-direction surface take the side labels.

#include "runmat-geometry-io/src/occt/kernel_ffi.rs.h"
#include "exact_kernel.hxx"

#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepTools.hxx>
#include <GeomLib_IsPlanarSurface.hxx>
#include <Geom_Surface.hxx>
#include <BRepOffsetAPI_MakeFilling.hxx>
#include <BRepOffsetAPI_ThruSections.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <GeomAbs_Shape.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Shell.hxx>
#include <TopoDS_Vertex.hxx>

#include <utility>

namespace runmat_geometry_io {
namespace occt_kernel {

namespace {

// Two edges with the same end points: an input edge as the builder rebuilt it.
bool same_ends(const TopoDS_Edge& e, const TopoDS_Edge& f) {
  TopoDS_Vertex e0, e1, f0, f1;
  TopExp::Vertices(e, e0, e1);
  TopExp::Vertices(f, f0, f1);
  if (e0.IsNull() || e1.IsNull() || f0.IsNull() || f1.IsNull()) return false;
  const gp_Pnt a0 = BRep_Tool::Pnt(e0), a1 = BRep_Tool::Pnt(e1);
  const gp_Pnt b0 = BRep_Tool::Pnt(f0), b1 = BRep_Tool::Pnt(f1);
  const double tol = 1e-7;
  return (a0.Distance(b0) < tol && a1.Distance(b1) < tol) ||
         (a0.Distance(b1) < tol && a1.Distance(b0) < tol);
}

GeomAbs_Shape order(int continuity) {
  return continuity == 2 ? GeomAbs_G2 : continuity == 1 ? GeomAbs_G1 : GeomAbs_C0;
}

void add_curve(BRepOffsetAPI_MakeFilling& fill, const BoundaryCurve& c, bool bounding) {
  if (c.continuity == 0 || c.face.IsNull()) {
    fill.Add(c.edge, GeomAbs_C0, bounding ? Standard_True : Standard_False);
  } else {
    fill.Add(c.edge, c.face, order(c.continuity), bounding ? Standard_True : Standard_False);
  }
}

// One direction: a smooth surface through the edges in order.
TopoDS_Shape through(const std::vector<BoundaryCurve>& first) {
  if (first.size() < 2) fail("a boundary surface in one direction needs two or more edges");
  BRepOffsetAPI_ThruSections loft(Standard_False, Standard_False, 1e-6);
  loft.CheckCompatibility(Standard_False);
  // Each section runs the way the one before it does, so the surface doesn't twist: an edge
  // whose ends sit nearer the previous section's opposite ends is taken reversed.
  gp_Pnt prev_start, prev_end;
  bool have_prev = false;
  for (const BoundaryCurve& c : first) {
    TopoDS_Edge e = c.edge;
    TopoDS_Vertex v0, v1;
    TopExp::Vertices(e, v0, v1, Standard_True);
    if (v0.IsNull() || v1.IsNull()) fail("a boundary edge has no end points");
    gp_Pnt s = BRep_Tool::Pnt(v0), t = BRep_Tool::Pnt(v1);
    if (have_prev && s.Distance(prev_start) + t.Distance(prev_end) > s.Distance(prev_end) + t.Distance(prev_start)) {
      e = TopoDS::Edge(e.Reversed());
      std::swap(s, t);
    }
    prev_start = s;
    prev_end = t;
    have_prev = true;
    BRepBuilderAPI_MakeWire w(e);
    if (!w.IsDone()) fail("a boundary edge could not be made a section");
    loft.AddWire(w.Wire());
  }
  loft.Build();
  if (!loft.IsDone()) fail("no surface passes through the edges in order");
  return loft.Shape();
}

// Two directions: the first and last edge of each bound the patch; the rest guide it.
TopoDS_Shape bounded(const std::vector<BoundaryCurve>& first,
                     const std::vector<BoundaryCurve>& second) {
  std::vector<const BoundaryCurve*> outer, inner;
  auto split = [&](const std::vector<BoundaryCurve>& curves) {
    for (std::size_t k = 0; k < curves.size(); ++k) {
      const bool bounding = k == 0 || k + 1 == curves.size();
      (bounding ? outer : inner).push_back(&curves[k]);
    }
  };
  split(first);
  split(second);
  BRepOffsetAPI_MakeFilling fill;
  for (const BoundaryCurve* c : outer) add_curve(fill, *c, true);
  for (const BoundaryCurve* c : inner) add_curve(fill, *c, false);
  fill.Build();
  if (!fill.IsDone()) {
    fail("the outer edges of the two directions do not close one loop, or no patch fits them");
  }
  return fill.Shape();
}

// A face that came out flat (edges in one plane) as a plane bounded by the same wire, so later
// features (offset, thicken) see a plane, not a flat spline.
TopoDS_Face flattened(const TopoDS_Face& f) {
  Handle(Geom_Surface) surface = BRep_Tool::Surface(f);
  GeomLib_IsPlanarSurface planar(surface, 1e-7);
  if (!planar.IsPlanar()) return f;
  BRepBuilderAPI_MakeFace mf(planar.Plan(), BRepTools::OuterWire(f), Standard_True);
  return mf.IsDone() ? mf.Face() : f;
}

}  // namespace

Tracked build_boundary(const std::vector<BoundaryCurve>& first,
                       const std::vector<BoundaryCurve>& second, std::uint32_t face_label,
                       std::uint32_t side_label0, std::uint32_t side_label1) {
  const TopoDS_Shape made = second.empty() ? through(first) : bounded(first, second);
  Tracked out;
  BRep_Builder bb;
  TopoDS_Shell sh;
  bb.MakeShell(sh);
  int faces = 0;
  for (TopExp_Explorer ex(made, TopAbs_FACE); ex.More(); ex.Next()) {
    const TopoDS_Face face = flattened(TopoDS::Face(ex.Current()));
    bb.Add(sh, face);
    add_origins(out.faces, face, label(face_label));
    ++faces;
  }
  if (faces == 0) fail("the boundary surface made no face");
  out.shape = sh;
  out.sheet = true;
  int sides = 0;
  TopTools_IndexedMapOfShape edges;
  TopExp::MapShapes(sh, TopAbs_EDGE, edges);
  for (int i = 1; i <= edges.Extent(); ++i) {
    const TopoDS_Edge e = TopoDS::Edge(edges(i));
    if (BRep_Tool::Degenerated(e)) continue;
    bool matched = false;
    for (const auto* curves : {&first, &second})
      for (const BoundaryCurve& c : *curves)
        if (e.IsSame(c.edge) || same_ends(e, c.edge)) {
          add_origins(out.edges, e, label(c.label));
          matched = true;
        }
    // A one-direction surface's side edges, from the first section to the last.
    if (!matched && second.empty() && sides < 2) {
      add_origins(out.edges, e, label(sides == 0 ? side_label0 : side_label1));
      ++sides;
    }
  }
  return out;
}

}  // namespace occt_kernel
}  // namespace runmat_geometry_io
