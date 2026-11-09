// Tool shapes for K1: prisms (with draft and bounded ends), revolves, translational sweeps,
// lofts (ruled between two sections, smooth through more), offset profiles, planar sheets and STEP import with CAD-file names.
//
// Every face a builder makes takes the label of the profile edge (side) or cap that made it,
// read from OCCT's Generated/FirstShape/LastShape records; sheet boundary edges take the
// profile's cap or vertex labels the same way.

#include "runmat-geometry-io/src/occt/kernel_ffi.rs.h"
#include "exact_kernel.hxx"

#include <BRepAdaptor_Surface.hxx>
#include <BRepAlgoAPI_Common.hxx>
#include <BRepAlgoAPI_Splitter.hxx>
#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepClass3d_SolidClassifier.hxx>
#include <BRepGProp.hxx>
#include <BRepOffsetAPI_DraftAngle.hxx>
#include <BRepOffsetAPI_MakeOffset.hxx>
#include <BRepOffsetAPI_MakePipeShell.hxx>
#include <BRepOffsetAPI_ThruSections.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <BRepPrimAPI_MakeRevol.hxx>
#include <BRepTools.hxx>
#include <BRepTools_WireExplorer.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <GProp_GProps.hxx>
#include <GeomAPI_Interpolate.hxx>
#include <Geom_BSplineCurve.hxx>
#include <Geom_RectangularTrimmedSurface.hxx>
#include <Geom_Surface.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <Interface_Static.hxx>
#include <STEPCAFControl_Controller.hxx>
#include <STEPCAFControl_Reader.hxx>
#include <Standard_Failure.hxx>
#include <TCollection_AsciiString.hxx>
#include <TColgp_HArray1OfPnt.hxx>
#include <TDF_Label.hxx>
#include <TDF_LabelSequence.hxx>
#include <TDataStd_Name.hxx>
#include <TDocStd_Document.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Wire.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>
#include <XCAFApp_Application.hxx>
#include <gp_Pln.hxx>

#include <cmath>
#include <algorithm>
#include <map>

namespace runmat_geometry_io {
namespace occt_kernel {
namespace {

constexpr std::uint32_t kInternal = 0xfffffffeu;  // a cap a later fuse removes

std::uint32_t lab(std::int64_t v) { return v < 0 ? kNoLabel : static_cast<std::uint32_t>(v); }

void label_face(Tracked& t, const TopoDS_Shape& f, std::uint32_t l) {
  if (f.IsNull() || l == kNoLabel) return;
  for (TopExp_Explorer ex(f, TopAbs_FACE); ex.More(); ex.Next()) add_origins(t.faces, ex.Current(), label(l));
}

void label_edge(Tracked& t, const TopoDS_Shape& e, std::uint32_t l) {
  if (e.IsNull() || l == kNoLabel) return;
  for (TopExp_Explorer ex(e, TopAbs_EDGE); ex.More(); ex.Next()) add_origins(t.edges, ex.Current(), label(l));
}

// The profile moved along its own z, with its edges' labels and caps carried.
Profile moved(const Profile& p, const gp_Vec& v) {
  if (v.Magnitude() < 1e-15) return p;
  gp_Trsf t;
  t.SetTranslation(v);
  BRepBuilderAPI_Transform mk(p.tracked.shape, t, Standard_True);
  Profile q;
  q.closed = p.closed;
  q.frame = p.frame.Transformed(t);
  q.tracked.shape = mk.Shape();
  q.tracked.sheet = p.tracked.sheet;
  propagate(p.tracked, mk, q.tracked);
  for (const ProfileCap& c : p.caps) {
    const TopTools_ListOfShape& m = mk.Modified(c.edge);
    q.caps.push_back({m.IsEmpty() ? c.edge : TopoDS::Edge(m.First()), c.start_cap, c.end_cap});
  }
  return q;
}

// Faces generated from the profile's labelled edges, and caps.
void label_sweep(const Profile& p, BRepBuilderAPI_MakeShape& mk, Tracked& out) {
  TopTools_IndexedMapOfShape present;
  TopExp::MapShapes(out.shape, TopAbs_FACE, present);
  for (int i = 1; i <= p.tracked.edges.Extent(); ++i) {
    for (const TopoDS_Shape& g : mk.Generated(p.tracked.edges.FindKey(i))) {
      for (TopExp_Explorer ex(g, TopAbs_FACE); ex.More(); ex.Next())
        if (present.Contains(ex.Current())) add_origins(out.faces, ex.Current(), p.tracked.edges(i));
    }
  }
}

// Sheet boundaries: side edges generated from chain-end vertices, cap edges at both ends.
template <class Sweep>
void label_sheet_boundaries(const Profile& p, Sweep& mk, Tracked& out) {
  for (int i = 1; i <= p.tracked.vertices.Extent(); ++i) {
    for (const TopoDS_Shape& g : mk.Generated(p.tracked.vertices.FindKey(i)))
      for (TopExp_Explorer ex(g, TopAbs_EDGE); ex.More(); ex.Next())
        add_origins(out.edges, ex.Current(), p.tracked.vertices(i));
  }
  for (const ProfileCap& c : p.caps) {
    label_edge(out, mk.FirstShape(c.edge), c.start_cap);
    label_edge(out, mk.LastShape(c.edge), c.end_cap);
  }
}

TopoDS_Shape fuse_all(const std::vector<Tracked>& parts, Tracked& out) {
  std::vector<const Tracked*> rest;
  for (std::size_t i = 1; i < parts.size(); ++i) rest.push_back(&parts[i]);
  out = boolean_op(0, {&parts[0]}, rest);
  return out.shape;
}

void reject_internal(const Tracked& t) {
  TopTools_IndexedMapOfShape faces;
  TopExp::MapShapes(t.shape, TopAbs_FACE, faces);
  for (int i = 1; i <= faces.Extent(); ++i) {
    for (const Origin& o : origins_of(t.faces, faces(i)))
      if (o.kind == kLabel && o.a == kInternal) fail("the pieces of the tool do not join");
  }
}

void drop_internal(Tracked& t) {
  for (int i = 1; i <= t.faces.Extent(); ++i) {
    Origins& o = t.faces.ChangeFromIndex(i);
    o.erase(std::remove_if(o.begin(), o.end(), [](const Origin& x) {
              return x.kind == kLabel && x.a == kInternal;
            }),
            o.end());
  }
}

// Draft the side faces of a prism about its profile plane.
Tracked drafted(const Tracked& prism, const Profile& base, double angle, const gp_Dir& away) {
  if (std::abs(angle) < 1e-15) return prism;
  BRepOffsetAPI_DraftAngle mk(prism.shape);
  const gp_Pln neutral(base.frame.Location(), base.frame.Direction());
  for (int i = 1; i <= prism.faces.Extent(); ++i) {
    const TopoDS_Face& f = TopoDS::Face(prism.faces.FindKey(i));
    // Side faces only: caps are perpendicular to the direction.
    BRepAdaptor_Surface s(f);
    if (s.GetType() == GeomAbs_Plane &&
        std::abs(s.Plane().Axis().Direction().Dot(away)) > 1.0 - 1e-9)
      continue;
    // Positive narrows the profile moving away from the profile plane.
    mk.Add(f, away, angle, neutral);
    if (!mk.AddDone()) fail("a prism side cannot take the draft");
  }
  mk.Build();
  if (!mk.IsDone()) fail("the drafted prism could not be built");
  Tracked out;
  out.shape = mk.Shape();
  propagate_draft(prism, mk, out);
  return out;
}
}  // namespace

// ------------------------------------------------------------------------- prism

Tracked build_prism(const Profile& p0, double start, double end, std::uint32_t start_label,
                    std::uint32_t end_label, double draft_end, double draft_start) {
  if (std::abs(end - start) < 1e-12) fail("an extrude needs a nonzero depth");
  const gp_Dir z = p0.frame.Direction();
  if ((draft_end != 0.0 || draft_start != 0.0) && start < 0.0 && end > 0.0 && p0.closed) {
    // Both directions drafted: each half narrows away from the profile plane.
    Tracked a = build_prism(p0, 0.0, end, kInternal, end_label, draft_end, 0.0);
    Tracked b = build_prism(p0, 0.0, start, kInternal, start_label, draft_start, 0.0);
    Tracked out;
    fuse_all({a, b}, out);
    reject_internal(out);
    drop_internal(out);
    return out;
  }
  const Profile p = moved(p0, gp_Vec(z) * start);
  BRepPrimAPI_MakePrism mk(p.tracked.shape, gp_Vec(z) * (end - start), Standard_True);
  if (!mk.IsDone()) fail("the extrude could not be built");
  Tracked out;
  out.shape = mk.Shape();
  out.sheet = !p.closed;
  label_sweep(p, mk, out);
  if (p.closed) {
    label_face(out, mk.FirstShape(), start_label);
    label_face(out, mk.LastShape(), end_label);
  } else {
    label_sheet_boundaries(p, mk, out);
  }
  const double angle = std::abs(end) >= std::abs(start) ? draft_end : draft_start;
  if (angle != 0.0 && p.closed) {
    const gp_Dir away = (end - start) > 0 ? z : z.Reversed();
    // The neutral plane is the profile plane (distance 0), wherever the prism starts.
    Profile base = p0;
    out = drafted(out, base, angle, away);
  }
  return out;
}

// op 30: ints [slot, operand, face, keep label, cut label], reals [move (3)]. Split a tracked
// solid by an operand face's untrimmed surface moved by `move`, keep the piece holding a face
// labelled `keep`, and label the cut face. Bounded extrudes (up to a face, up to next, from a
// face) are a long prism trimmed this way: the cap takes the face's shape.
Tracked split_keep(const Tracked& src, const KInstr& in, const Body& operand) {
  const std::int64_t face = in.ints[2];
  const std::uint32_t keep = lab(in.ints[3]);
  const std::uint32_t cut_label = lab(in.ints[4]);
  if (face < 0 || static_cast<std::size_t>(face) >= operand.faces.size()) fail("end face out of range");
  const TopoDS_Face& f = operand.faces[static_cast<std::size_t>(face)];
  Handle(Geom_Surface) surf = BRep_Tool::Surface(f);
  if (surf->IsKind(STANDARD_TYPE(Geom_RectangularTrimmedSurface)))
    surf = Handle(Geom_RectangularTrimmedSurface)::DownCast(surf)->BasisSurface();
  Bnd_Box box;
  BRepBndLib::Add(src.shape, box);
  const double size = std::sqrt(box.SquareExtent()) * 4.0 + 1.0;
  double u0, u1, v0, v1;
  surf->Bounds(u0, u1, v0, v1);
  BRepAdaptor_Surface fa(f);
  auto clamp = [&](double lo, double hi, double flo, double fhi, bool periodic) {
    if (periodic) return std::make_pair(lo, hi);
    const double grow = std::max(size, (fhi - flo) * 4.0);
    return std::make_pair(std::max(lo, flo - grow), std::min(hi, fhi + grow));
  };
  auto ur = clamp(u0, u1, fa.FirstUParameter(), fa.LastUParameter(), surf->IsUPeriodic());
  auto vr = clamp(v0, v1, fa.FirstVParameter(), fa.LastVParameter(), surf->IsVPeriodic());
  BRepBuilderAPI_MakeFace mf(surf, ur.first, ur.second, vr.first, vr.second, 1e-7);
  if (!mf.IsDone()) fail("the end face's surface cannot bound the extrude");
  TopoDS_Shape cutter = mf.Face();
  const gp_Vec mv(in.reals[0], in.reals[1], in.reals[2]);
  if (mv.Magnitude() > 0) {
    gp_Trsf t;
    t.SetTranslation(mv);
    cutter = BRepBuilderAPI_Transform(cutter, t, Standard_True).Shape();
  }
  BRepAlgoAPI_Splitter sp;
  TopTools_ListOfShape args, tools;
  args.Append(src.shape);
  tools.Append(cutter);
  sp.SetArguments(args);
  sp.SetTools(tools);
  sp.SetRunParallel(Standard_False);
  sp.Build();
  if (!sp.IsDone() || sp.HasErrors()) fail("the extrude does not meet its end face");
  Tracked all;
  all.shape = sp.Shape();
  propagate(src, sp, all);
  TopoDS_Shape kept;
  int pieces = 0;
  for (TopExp_Explorer ex(all.shape, TopAbs_SOLID); ex.More(); ex.Next()) {
    ++pieces;
    if (!kept.IsNull()) continue;
    for (TopExp_Explorer fx(ex.Current(), TopAbs_FACE); fx.More(); fx.Next()) {
      bool is_keep = false;
      for (const Origin& x : origins_of(all.faces, fx.Current())) is_keep |= x.kind == kLabel && x.a == keep;
      if (is_keep) {
        kept = ex.Current();
        break;
      }
    }
  }
  if (pieces < 2 || kept.IsNull()) fail("the extrude does not reach its end face");
  Tracked out;
  out.shape = kept;
  for (TopExp_Explorer fx(kept, TopAbs_FACE); fx.More(); fx.Next()) {
    Origins o = origins_of(all.faces, fx.Current());
    if (o.empty()) o = label(cut_label);
    add_origins(out.faces, fx.Current(), o);
  }
  reject_internal(out);
  return out;
}

// A blind prism that stops where it first meets the operand body, plus an offset.
Tracked prism_to_body(const Profile& p, const KInstr& in, const Body& operand) {
  const std::uint32_t sl = lab(in.ints[1]);
  const std::uint32_t el = lab(in.ints[2]);
  const double start = in.reals[0], end_limit = in.reals[1], offset = in.reals[2];
  const double dir = end_limit >= start ? 1.0 : -1.0;
  Tracked probe = build_prism(p, start, end_limit, kNoLabel, kNoLabel);
  BRepAlgoAPI_Common common(probe.shape, operand.shape);
  if (!common.IsDone()) fail("the extrude cannot be intersected with its end body");
  gp_Trsf to_local;
  to_local.SetTransformation(p.frame);
  TopoDS_Shape local = BRepBuilderAPI_Transform(common.Shape(), to_local, Standard_True).Shape();
  Bnd_Box box;
  BRepBndLib::AddOptimal(local, box, Standard_False, Standard_False);
  if (box.IsVoid()) fail("the extrude does not meet its end body");
  double x0, y0, z0, x1, y1, z1;
  box.Get(x0, y0, z0, x1, y1, z1);
  const double hit = dir > 0 ? z0 : z1;
  return build_prism(p, start, hit + dir * offset, sl, el);
}

// ------------------------------------------------------------------------- revolve

Tracked build_revolve(const Profile& p, const gp_Ax1& axis, double angle,
                      std::uint32_t start_label, std::uint32_t end_label) {
  const bool full = std::abs(std::abs(angle) - 2 * M_PI) < 1e-12;
  if (std::abs(angle) < 1e-12) fail("a revolve needs a nonzero angle");
  BRepPrimAPI_MakeRevol mk(p.tracked.shape, axis, full ? 2 * M_PI : angle, Standard_True);
  if (!mk.IsDone()) fail("the revolve could not be built (the profile must not cross its axis)");
  Tracked out;
  out.shape = mk.Shape();
  out.sheet = !p.closed;
  label_sweep(p, mk, out);
  if (p.closed) {
    if (!full) {
      label_face(out, mk.FirstShape(), start_label);
      label_face(out, mk.LastShape(), end_label);
    }
  } else {
    label_sheet_boundaries(p, mk, out);
  }
  return out;
}

// ------------------------------------------------------------------------- sweep

// One path curve: the profile translated along the path (sections stay parallel). Two points
// make a prism; more make a smooth spine through them.
Tracked build_sweep(const Profile& p, const std::vector<gp_Pnt>& path, std::uint32_t start_label,
                    std::uint32_t end_label) {
  if (path.size() < 2) fail("a sweep path needs two points");
  const gp_Vec offset(path.front().XYZ() - path.front().XYZ());
  if (path.size() == 2) {
    const gp_Vec v(path[0], path[1]);
    if (v.Magnitude() < 1e-12) fail("a sweep path segment has no length");
    BRepPrimAPI_MakePrism mk(p.tracked.shape, v, Standard_True);
    if (!mk.IsDone()) fail("the sweep could not be built");
    Tracked out;
    out.shape = mk.Shape();
    out.sheet = !p.closed;
    label_sweep(p, mk, out);
    if (p.closed) {
      label_face(out, mk.FirstShape(), start_label);
      label_face(out, mk.LastShape(), end_label);
    } else {
      label_sheet_boundaries(p, mk, out);
    }
    return out;
  }
  Handle(TColgp_HArray1OfPnt) pts = new TColgp_HArray1OfPnt(1, static_cast<int>(path.size()));
  for (std::size_t i = 0; i < path.size(); ++i) pts->SetValue(static_cast<int>(i + 1), path[i]);
  GeomAPI_Interpolate interp(pts, Standard_False, 1e-9);
  interp.Perform();
  if (!interp.IsDone()) fail("the sweep path cannot be fitted");
  TopoDS_Edge spine_edge = BRepBuilderAPI_MakeEdge(interp.Curve()).Edge();
  TopoDS_Wire spine = BRepBuilderAPI_MakeWire(spine_edge).Wire();
  // Each loop of the profile sweeps on its own; holes are cut from the outer sweep.
  std::vector<TopoDS_Wire> loops;
  for (TopExp_Explorer ex(p.tracked.shape, TopAbs_WIRE); ex.More(); ex.Next())
    loops.push_back(TopoDS::Wire(ex.Current()));
  std::vector<Tracked> pieces;
  for (std::size_t li = 0; li < loops.size(); ++li) {
    BRepOffsetAPI_MakePipeShell mk(spine);
    mk.SetMode(gp_Ax2(path.front(), p.frame.Direction(), p.frame.XDirection()));
    mk.SetTransitionMode(BRepBuilderAPI_RightCorner);
    mk.Add(loops[li], Standard_False, Standard_False);
    mk.Build();
    if (!mk.IsDone()) fail("the sweep could not be built along its path");
    if (p.closed && !mk.MakeSolid()) fail("the swept profile does not close a solid");
    Tracked t;
    t.shape = mk.Shape();
    t.sheet = !p.closed;
    TopTools_IndexedMapOfShape present;
    TopExp::MapShapes(t.shape, TopAbs_FACE, present);
    for (TopExp_Explorer ex(loops[li], TopAbs_EDGE); ex.More(); ex.Next()) {
      const Origins& o = origins_of(p.tracked.edges, ex.Current());
      for (const TopoDS_Shape& g : mk.Generated(ex.Current()))
        for (TopExp_Explorer fx(g, TopAbs_FACE); fx.More(); fx.Next())
          if (present.Contains(fx.Current())) add_origins(t.faces, fx.Current(), o);
    }
    if (p.closed) {
      label_face(t, mk.FirstShape(), start_label);
      label_face(t, mk.LastShape(), end_label);
    } else {
      for (int i = 1; i <= p.tracked.vertices.Extent(); ++i)
        for (const TopoDS_Shape& g : mk.Generated(p.tracked.vertices.FindKey(i)))
          for (TopExp_Explorer ex(g, TopAbs_EDGE); ex.More(); ex.Next())
            add_origins(t.edges, ex.Current(), p.tracked.vertices(i));
      for (const ProfileCap& c : p.caps) {
        label_edge(t, c.edge, c.start_cap);
        TopTools_IndexedMapOfShape last_edges;
        TopExp::MapShapes(mk.LastShape(), TopAbs_EDGE, last_edges);
        // The end copy of a profile edge: the last section's edge generated from it.
        for (const TopoDS_Shape& g : mk.Generated(c.edge))
          for (TopExp_Explorer ex(g, TopAbs_EDGE); ex.More(); ex.Next())
            if (last_edges.Contains(ex.Current())) label_edge(t, ex.Current(), c.end_cap);
      }
    }
    pieces.push_back(std::move(t));
  }
  (void)offset;
  if (pieces.size() == 1) return pieces.front();
  std::vector<const Tracked*> holes;
  for (std::size_t i = 1; i < pieces.size(); ++i) holes.push_back(&pieces[i]);
  return boolean_op(1, {&pieces[0]}, holes);
}

// ------------------------------------------------------------------------- loft

// A solid loft between two sections is ruled; through more it is one smooth surface per loop
// (BRepOffsetAPI_ThruSections, not ruled), its sides named by the first section's edges. A
// sheet loft is ruled per span, section k's edge labels naming span k's sides (the lowering
// numbers them by span). Caps are the first and last sections.
namespace {
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

// A sheet loft (open chains): one ruled face per chain curve per span, tagged by the curve.
// Boundary edges: the first section's curves are start caps, the last section's end caps, and
// the edges from chain ends carry their vertices' labels.
Tracked build_sheet_loft(const std::vector<const Profile*>& sections) {
  std::vector<Tracked> spans;
  for (std::size_t k = 0; k + 1 < sections.size(); ++k) {
    const Profile& a = *sections[k];
    const Profile& b = *sections[k + 1];
    if (a.caps.size() != b.caps.size()) fail("loft sections have different numbers of curves");
    std::vector<TopoDS_Wire> wa, wb;
    for (TopExp_Explorer ex(a.tracked.shape, TopAbs_WIRE); ex.More(); ex.Next()) wa.push_back(TopoDS::Wire(ex.Current()));
    for (TopExp_Explorer ex(b.tracked.shape, TopAbs_WIRE); ex.More(); ex.Next()) wb.push_back(TopoDS::Wire(ex.Current()));
    if (wa.size() != wb.size()) fail("loft sections have different numbers of chains");
    for (std::size_t li = 0; li < wa.size(); ++li) {
      BRepOffsetAPI_ThruSections mk(Standard_False, Standard_True, 1e-6);
      mk.CheckCompatibility(Standard_False);
      mk.AddWire(wa[li]);
      mk.AddWire(wb[li]);
      mk.Build();
      if (!mk.IsDone()) fail("the surface loft could not be built through its sections");
      Tracked t;
      t.shape = mk.Shape();
      t.sheet = true;
      TopTools_IndexedMapOfShape chain;
      TopExp::MapShapes(wa[li], TopAbs_EDGE, chain);
      for (std::size_t ci = 0; ci < a.caps.size(); ++ci) {
        const ProfileCap& c = a.caps[ci];
        if (!chain.Contains(c.edge)) continue;
        const TopoDS_Shape f = mk.GeneratedFace(c.edge);
        if (f.IsNull()) continue;
        add_origins(t.faces, f, origins_of(a.tracked.edges, c.edge));
        for (TopExp_Explorer ex(f, TopAbs_EDGE); ex.More(); ex.Next()) {
          const TopoDS_Edge e = TopoDS::Edge(ex.Current());
          if (k == 0 && same_ends(e, c.edge)) label_edge(t, e, c.start_cap);
          if (k + 2 == sections.size() && same_ends(e, b.caps[ci].edge)) label_edge(t, e, c.end_cap);
        }
      }
      // Edges running from a chain-end vertex of this section to the next.
      for (TopExp_Explorer ex(t.shape, TopAbs_EDGE); ex.More(); ex.Next()) {
        TopoDS_Vertex v0, v1;
        TopExp::Vertices(TopoDS::Edge(ex.Current()), v0, v1);
        if (v0.IsNull() || v1.IsNull()) continue;
        for (int i = 1; i <= a.tracked.vertices.Extent(); ++i) {
          const gp_Pnt p = BRep_Tool::Pnt(TopoDS::Vertex(a.tracked.vertices.FindKey(i)));
          const gp_Pnt q0 = BRep_Tool::Pnt(v0), q1 = BRep_Tool::Pnt(v1);
          const bool at0 = p.Distance(q0) < 1e-7, at1 = p.Distance(q1) < 1e-7;
          if (at0 == at1) continue;  // not from this vertex, or degenerate
          const gp_Pnt other = at0 ? q1 : q0;
          auto on = [&](const TopoDS_Wire& w) {
            for (TopExp_Explorer vx(w, TopAbs_VERTEX); vx.More(); vx.Next())
              if (BRep_Tool::Pnt(TopoDS::Vertex(vx.Current())).Distance(other) < 1e-7) return true;
            return false;
          };
          if (on(wa[li]) || !on(wb[li])) continue;  // a section edge, not one along the loft
          add_origins(t.edges, ex.Current(), a.tracked.vertices(i));
        }
      }
      spans.push_back(std::move(t));
    }
  }
  if (spans.size() == 1) return spans.front();
  Tracked out;
  fuse_all(spans, out);
  out.sheet = true;
  return out;
}
}  // namespace

Tracked build_loft(const std::vector<const Profile*>& sections, std::uint32_t start_label,
                   std::uint32_t end_label) {
  if (sections.size() < 2) fail("a loft needs two sections");
  if (!sections.front()->closed) return build_sheet_loft(sections);
  // Three or more sections: one smooth surface through them all,
  // so the loft has no crease or edge at its inner sections.
  if (sections.size() > 2) {
    std::vector<std::vector<TopoDS_Wire>> wires(sections.size());
    for (std::size_t k = 0; k < sections.size(); ++k) {
      for (TopExp_Explorer ex(sections[k]->tracked.shape, TopAbs_WIRE); ex.More(); ex.Next()) wires[k].push_back(TopoDS::Wire(ex.Current()));
      if (wires[k].size() != wires[0].size()) fail("loft sections have different numbers of loops");
    }
    const Profile& a = *sections.front();
    std::vector<Tracked> loops;
    for (std::size_t li = 0; li < wires[0].size(); ++li) {
      BRepOffsetAPI_ThruSections mk(Standard_True, Standard_False, 1e-6);
      mk.CheckCompatibility(Standard_False);
      for (const auto& w : wires) mk.AddWire(w[li]);
      mk.Build();
      if (!mk.IsDone()) fail("the loft could not be built through its sections");
      Tracked t;
      t.shape = mk.Shape();
      for (TopExp_Explorer ex(wires[0][li], TopAbs_EDGE); ex.More(); ex.Next()) {
        const TopoDS_Shape f = mk.GeneratedFace(ex.Current());
        if (!f.IsNull()) add_origins(t.faces, f, origins_of(a.tracked.edges, ex.Current()));
      }
      label_face(t, mk.FirstShape(), start_label);
      label_face(t, mk.LastShape(), end_label);
      loops.push_back(std::move(t));
    }
    if (loops.size() == 1) return loops.front();
    std::vector<const Tracked*> holes;
    for (std::size_t i = 1; i < loops.size(); ++i) holes.push_back(&loops[i]);
    return boolean_op(1, {&loops[0]}, holes);
  }
  std::vector<Tracked> spans;
  for (std::size_t k = 0; k + 1 < sections.size(); ++k) {
    const Profile& a = *sections[k];
    const Profile& b = *sections[k + 1];
    std::vector<TopoDS_Wire> wa, wb;
    for (TopExp_Explorer ex(a.tracked.shape, TopAbs_WIRE); ex.More(); ex.Next()) wa.push_back(TopoDS::Wire(ex.Current()));
    for (TopExp_Explorer ex(b.tracked.shape, TopAbs_WIRE); ex.More(); ex.Next()) wb.push_back(TopoDS::Wire(ex.Current()));
    if (wa.size() != wb.size()) fail("loft sections have different numbers of loops");
    std::vector<Tracked> loops;
    for (std::size_t li = 0; li < wa.size(); ++li) {
      BRepOffsetAPI_ThruSections mk(Standard_True, Standard_True, 1e-6);
      mk.CheckCompatibility(Standard_False);
      mk.AddWire(wa[li]);
      mk.AddWire(wb[li]);
      mk.Build();
      if (!mk.IsDone()) fail("the loft could not be built through its sections");
      Tracked t;
      t.shape = mk.Shape();
      for (TopExp_Explorer ex(wa[li], TopAbs_EDGE); ex.More(); ex.Next()) {
        const TopoDS_Shape f = mk.GeneratedFace(ex.Current());
        if (!f.IsNull()) add_origins(t.faces, f, origins_of(a.tracked.edges, ex.Current()));
      }
      label_face(t, mk.FirstShape(), k == 0 ? start_label : kInternal);
      label_face(t, mk.LastShape(), k + 2 == sections.size() ? end_label : kInternal);
      loops.push_back(std::move(t));
    }
    if (loops.size() == 1) {
      spans.push_back(std::move(loops.front()));
    } else {
      std::vector<const Tracked*> holes;
      for (std::size_t i = 1; i < loops.size(); ++i) holes.push_back(&loops[i]);
      spans.push_back(boolean_op(1, {&loops[0]}, holes));
    }
  }
  if (spans.size() == 1) return spans.front();
  Tracked out;
  fuse_all(spans, out);
  reject_internal(out);
  return out;
}

// ------------------------------------------------------------------------- offset profiles

// op 28: ints [profile, segment count, new side label per segment], reals [distance].
// Positive grows the profile. Each offset edge takes the new label of the segment it offsets.
Profile offset_profile(const Profile& p, const KInstr& in) {
  if (!p.closed) unsupported("thin walls of open profiles");
  const std::int64_t n = in.ints[1];
  const double d = in.reals[0];
  if (p.tracked.shape.ShapeType() != TopAbs_FACE) unsupported("thin walls of a profile with separate regions");
  BRepOffsetAPI_MakeOffset mk(TopoDS::Face(p.tracked.shape), GeomAbs_Intersection, Standard_False);
  mk.Perform(d);
  if (!mk.IsDone()) fail("the profile cannot be offset by the wall thickness");
  Profile q;
  q.frame = p.frame;
  q.closed = true;
  std::vector<TopoDS_Wire> wires;
  for (TopExp_Explorer ex(mk.Shape(), TopAbs_WIRE); ex.More(); ex.Next()) wires.push_back(TopoDS::Wire(ex.Current()));
  if (wires.empty()) fail("the offset profile vanished");
  for (std::int64_t i = 0; i < n && static_cast<std::size_t>(i) < p.caps.size(); ++i) {
    const std::uint32_t l = lab(in.ints[2 + i]);
    for (const TopoDS_Shape& g : mk.Generated(p.caps[static_cast<std::size_t>(i)].edge))
      for (TopExp_Explorer ex(g, TopAbs_EDGE); ex.More(); ex.Next()) {
        add_origins(q.tracked.edges, ex.Current(), label(l));
        q.caps.push_back({TopoDS::Edge(ex.Current()), kNoLabel, kNoLabel});
      }
  }
  // Find the outer wire by area: the face of all loops.
  std::size_t outer = 0;
  double best = -1;
  for (std::size_t i = 0; i < wires.size(); ++i) {
    BRepBuilderAPI_MakeFace f(gp_Pln(p.frame), wires[i], Standard_True);
    if (!f.IsDone()) continue;
    GProp_GProps g;
    BRepGProp::SurfaceProperties(f.Face(), g);
    if (std::abs(g.Mass()) > best) {
      best = std::abs(g.Mass());
      outer = i;
    }
  }
  BRepBuilderAPI_MakeFace mf(gp_Pln(p.frame), wires[outer], Standard_True);
  for (std::size_t i = 0; i < wires.size(); ++i)
    if (i != outer) mf.Add(TopoDS::Wire(wires[i].Reversed()));
  if (!mf.IsDone()) fail("the offset profile does not make a face");
  q.tracked.shape = mf.Face();
  // Corner edges an offset may add carry no label: the thin wall fails rather than guess.
  for (TopExp_Explorer ex(q.tracked.shape, TopAbs_EDGE); ex.More(); ex.Next())
    if (origins_of(q.tracked.edges, ex.Current()).empty())
      incomplete("the offset profile has an edge no profile curve made");
  return q;
}

// op 26: a planar sheet from a closed profile.
Tracked build_planar_sheet(const Profile& p, std::uint32_t face_label) {
  if (!p.closed) fail("a planar patch needs closed loops");
  Tracked t = p.tracked;
  BRep_Builder b;
  TopoDS_Shell sh;
  b.MakeShell(sh);
  for (TopExp_Explorer ex(p.tracked.shape, TopAbs_FACE); ex.More(); ex.Next()) b.Add(sh, ex.Current());
  t.shape = sh;
  t.sheet = true;
  label_face(t, p.tracked.shape, face_label);
  return t;
}

// ------------------------------------------------------------------------- STEP import

namespace {
Handle(TDocStd_Document) read_step(const std::string& path) {
  Handle(XCAFApp_Application) app = XCAFApp_Application::GetApplication();
  Handle(TDocStd_Document) doc;
  app->NewDocument("MDTV-XCAF", doc);
  // The kernel works in metres. The reader converts the file's lengths into the document's
  // unit, which otherwise defaults to xstep.cascade.unit (millimetres).
  XCAFDoc_DocumentTool::SetLengthUnit(doc, 1.0);
  STEPCAFControl_Controller::Init();
  // Face names come back from named sub-shapes (off by default in OCCT).
  Interface_Static::SetIVal("read.stepcaf.subshapes.name", 1);
  STEPCAFControl_Reader reader;
  reader.SetNameMode(Standard_True);
  reader.SetColorMode(Standard_True);
  if (reader.ReadFile(path.c_str()) != IFSelect_RetDone) fail("the STEP file cannot be read: " + path);
  if (!reader.Transfer(doc)) fail("the STEP file's shapes cannot be transferred: " + path);
  return doc;
}
std::string label_name(const TDF_Label& l) {
  Handle(TDataStd_Name) n;
  if (l.FindAttribute(TDataStd_Name::GetID(), n)) {
    TCollection_AsciiString s(n->Get());
    return s.ToCString();
  }
  return std::string();
}
}  // namespace

rust::Vec<rust::String> kernel_step_products(rust::Str path) {
  const auto guard = kernel_lock();
  try {
    Handle(TDocStd_Document) doc = read_step(std::string(path));
    Handle(XCAFDoc_ShapeTool) st = XCAFDoc_DocumentTool::ShapeTool(doc->Main());
    TDF_LabelSequence free;
    st->GetFreeShapes(free);
    rust::Vec<rust::String> out;
    for (int i = 1; i <= free.Length(); ++i) {
      std::string n = label_name(free(i));
      if (n.empty()) n = "product" + std::to_string(i - 1);
      out.push_back(rust::String(n));
    }
    return out;
  } catch (const Standard_Failure& e) {
    fail(std::string("OCCT raised ") + e.GetMessageString());
  }
}

// op 18: strs [path], ints [product index]. Faces take the names the file gives them; a face
// with no name of its own is named by its product and its order in it.
Tracked import_step(const KInstr& in, KResult& result) {
  if (in.strs.empty() || in.ints.empty()) fail("an import needs a source path and product");
  const std::string path(in.strs[0]);
  const std::int64_t want = in.ints[0];
  Handle(TDocStd_Document) doc = read_step(path);
  Handle(XCAFDoc_ShapeTool) st = XCAFDoc_DocumentTool::ShapeTool(doc->Main());
  TDF_LabelSequence free;
  st->GetFreeShapes(free);
  BRep_Builder b;
  TopoDS_Compound c;
  b.MakeCompound(c);
  Tracked t;
  std::map<std::string, std::uint32_t> name_index;
  auto name_id = [&](const std::string& n) {
    auto it = name_index.find(n);
    if (it != name_index.end()) return it->second;
    const std::uint32_t id = static_cast<std::uint32_t>(result.names.size());
    result.names.push_back(rust::String(n));
    name_index[n] = id;
    return id;
  };
  int product = static_cast<int>(want);
  if (want < 0 || want >= free.Length()) fail("the STEP file has no such product");
  for (int i = static_cast<int>(want) + 1; i <= static_cast<int>(want) + 1; ++i) {
    TopoDS_Shape s = st->GetShape(free(i));
    const std::string pname = label_name(free(i));
    b.Add(c, s);
    int fi = 0;
    for (TopExp_Explorer ex(s, TopAbs_FACE); ex.More(); ex.Next(), ++fi) {
      TDF_Label fl;
      std::string n;
      if (st->Search(ex.Current(), fl, Standard_True, Standard_True, Standard_True)) n = label_name(fl);
      if (n.empty()) n = (pname.empty() ? "product" + std::to_string(product) : pname) + "/face" + std::to_string(fi);
      Origin o;
      o.kind = kName;
      o.a = name_id(n);
      add_origins(t.faces, ex.Current(), Origins{o});
    }
  }
  t.shape = c;
  return t;
}

}  // namespace occt_kernel
}  // namespace runmat_geometry_io
