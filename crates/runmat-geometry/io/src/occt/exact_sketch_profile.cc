// Sketch profiles to OCCT wires and faces (K1).
//
// Encoding of a PROFILE instruction (op 2):
//   reals: frame (origin, x, y, z: 12), then each segment's parameters in order;
//   ints:  closed (1: planar faces of closed loops, nested by containment: a loop inside an
//          even number of others bounds a face, and the loops directly inside it are its holes;
//          one face, or a compound of faces for separate regions; 0: open or closed chains as
//          wires), loop count, then per loop: segment count, start-vertex label, end-vertex
//          label (chain ends, for sheet boundaries; -1 none), then per segment: kind, side label,
//          start-cap boundary label, end-cap boundary label.
// Segment kinds and their reals (sketch coordinates in the frame):
//   0 line (sx, sy, ex, ey)
//   1 arc counter-clockwise (sx, sy, ex, ey, cx, cy)        2 arc clockwise (same)
//   3 circle (cx, cy, r)
//   4 elliptic arc (sx, sy, ex, ey, cx, cy, ax, ay, ratio, clockwise): (ax, ay) is the
//     semi-major axis vector, ratio = minor / major
//   5 ellipse (cx, cy, ax, ay, ratio)
//   6 B-spline (degree, pole count n, n x (x, y), n weights, knot count m, m knots,
//     m multiplicities, periodic)
// Every edge carries its side label; chain-end vertices carry their labels; the profile keeps
// each edge's cap labels for sheet boundaries.

#include "runmat-geometry-io/src/occt/kernel_ffi.rs.h"
#include "exact_kernel.hxx"

#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepLib.hxx>
#include <BRep_Builder.hxx>
#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepTools.hxx>
#include <BRepTools_WireExplorer.hxx>
#include <GeomConvert.hxx>
#include <BRepClass_FaceClassifier.hxx>
#include <BRepGProp.hxx>
#include <BRep_Tool.hxx>
#include <ElCLib.hxx>
#include <GProp_GProps.hxx>
#include <GC_MakeArcOfCircle.hxx>
#include <GC_MakeArcOfEllipse.hxx>
#include <Geom_BSplineCurve.hxx>
#include <Geom_Circle.hxx>
#include <Geom_Ellipse.hxx>
#include <Geom_Line.hxx>
#include <Geom_TrimmedCurve.hxx>
#include <ShapeFix_Face.hxx>
#include <Standard_Failure.hxx>
#include <TColStd_Array1OfInteger.hxx>
#include <TColStd_Array1OfReal.hxx>
#include <TColgp_Array1OfPnt.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Wire.hxx>
#include <gp_Ax2.hxx>
#include <gp_Circ.hxx>
#include <gp_Elips.hxx>
#include <gp_Pln.hxx>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace runmat_geometry_io {
namespace occt_kernel {
namespace {

struct Reader {
  const rust::Vec<double>& r;
  std::size_t at;
  double next() {
    if (at >= r.size()) fail("a profile segment is missing parameters");
    return r[at++];
  }
};

gp_Pnt world(const gp_Ax3& f, double x, double y) {
  return gp_Pnt(f.Location().XYZ() + f.XDirection().XYZ() * x + f.YDirection().XYZ() * y);
}

gp_Vec world_vec(const gp_Ax3& f, double x, double y) {
  return gp_Vec(f.XDirection().XYZ() * x + f.YDirection().XYZ() * y);
}

std::uint32_t lab(std::int64_t v) { return v < 0 ? kNoLabel : static_cast<std::uint32_t>(v); }

// The vertex at a sketch point, shared with the previous segment's end.
TopoDS_Vertex vertex_at(const gp_Pnt& p, TopoDS_Vertex& previous) {
  if (!previous.IsNull() && BRep_Tool::Pnt(previous).Distance(p) < 1e-9) return previous;
  return BRepBuilderAPI_MakeVertex(p).Vertex();
}

TopoDS_Edge make_segment(const gp_Ax3& f, std::int64_t kind, Reader& rd, TopoDS_Vertex& last,
                         const TopoDS_Vertex& first, bool closes) {
  const gp_Dir z = f.Direction();
  auto end_vertex = [&](const gp_Pnt& p) {
    if (closes && !first.IsNull() && BRep_Tool::Pnt(first).Distance(p) < 1e-9) return first;
    return BRepBuilderAPI_MakeVertex(p).Vertex();
  };
  switch (kind) {
    case 0: {
      const double sx = rd.next(), sy = rd.next(), ex = rd.next(), ey = rd.next();
      const gp_Pnt a = world(f, sx, sy), b = world(f, ex, ey);
      if (a.Distance(b) < 1e-12) fail("a profile line has no length");
      TopoDS_Vertex va = vertex_at(a, last), vb = end_vertex(b);
      Handle(Geom_Line) l = new Geom_Line(a, gp_Dir(gp_Vec(a, b)));
      BRepBuilderAPI_MakeEdge me(l, va, vb, 0.0, a.Distance(b));
      if (!me.IsDone()) fail("a profile line could not be built");
      last = vb;
      return me.Edge();
    }
    case 1:
    case 2: {
      const double sx = rd.next(), sy = rd.next(), ex = rd.next(), ey = rd.next();
      const double cx = rd.next(), cy = rd.next();
      const gp_Pnt a = world(f, sx, sy), b = world(f, ex, ey), c = world(f, cx, cy);
      const double r = a.Distance(c);
      if (r < 1e-12) fail("a profile arc has no radius");
      gp_Ax2 ax(c, z, gp_Dir(gp_Vec(c, a)));
      gp_Circ circ(ax, r);
      Handle(Geom_Circle) gc = new Geom_Circle(circ);
      double u0 = ElCLib::Parameter(circ, a);
      double u1 = ElCLib::Parameter(circ, b);
      TopoDS_Vertex va = vertex_at(a, last), vb = end_vertex(b);
      TopoDS_Edge e;
      if (kind == 1) {
        if (u1 <= u0 + 1e-12) u1 += 2 * M_PI;
        e = BRepBuilderAPI_MakeEdge(gc, va, vb, u0, u1).Edge();
      } else {
        // Clockwise from a to b is counter-clockwise from b to a; the edge is reversed so
        // the loop still runs a to b.
        double v0 = u1, v1 = u0;
        if (v1 <= v0 + 1e-12) v1 += 2 * M_PI;
        e = TopoDS::Edge(BRepBuilderAPI_MakeEdge(gc, vb, va, v0, v1).Edge().Reversed());
      }
      last = vb;
      return e;
    }
    case 3: {
      const double cx = rd.next(), cy = rd.next(), r = rd.next();
      if (r <= 0) fail("a profile circle has no radius");
      gp_Ax2 ax(world(f, cx, cy), z, f.XDirection());
      TopoDS_Edge e = BRepBuilderAPI_MakeEdge(gp_Circ(ax, r)).Edge();
      last.Nullify();
      return e;
    }
    case 4:
    case 5: {
      double sx = 0, sy = 0, ex = 0, ey = 0;
      if (kind == 4) {
        sx = rd.next(); sy = rd.next(); ex = rd.next(); ey = rd.next();
      }
      const double cx = rd.next(), cy = rd.next(), ax_ = rd.next(), ay = rd.next();
      const double ratio = rd.next();
      const bool cw = kind == 4 ? rd.next() != 0.0 : false;
      const gp_Pnt c = world(f, cx, cy);
      const gp_Vec major = world_vec(f, ax_, ay);
      const double ra = major.Magnitude();
      const double rb = ra * ratio;
      if (ra <= 0 || rb <= 0 || ratio > 1.0 + 1e-12) fail("a profile ellipse needs a major axis at least its minor axis");
      gp_Elips el(gp_Ax2(c, z, gp_Dir(major)), ra, rb);
      Handle(Geom_Ellipse) ge = new Geom_Ellipse(el);
      if (kind == 5) {
        last.Nullify();
        return BRepBuilderAPI_MakeEdge(ge).Edge();
      }
      const gp_Pnt a = world(f, sx, sy), b = world(f, ex, ey);
      double u0 = ElCLib::Parameter(el, a), u1 = ElCLib::Parameter(el, b);
      TopoDS_Vertex va = vertex_at(a, last), vb = end_vertex(b);
      TopoDS_Edge e;
      if (!cw) {
        if (u1 <= u0 + 1e-12) u1 += 2 * M_PI;
        e = BRepBuilderAPI_MakeEdge(ge, va, vb, u0, u1).Edge();
      } else {
        double v0 = u1, v1 = u0;
        if (v1 <= v0 + 1e-12) v1 += 2 * M_PI;
        e = TopoDS::Edge(BRepBuilderAPI_MakeEdge(ge, vb, va, v0, v1).Edge().Reversed());
      }
      last = vb;
      return e;
    }
    case 6: {
      const int degree = static_cast<int>(rd.next());
      const int n = static_cast<int>(rd.next());
      if (degree < 1 || n < degree + 1) fail("a profile spline needs more poles than its degree");
      TColgp_Array1OfPnt poles(1, n);
      for (int i = 1; i <= n; ++i) {
        const double x = rd.next(), y = rd.next();
        poles.SetValue(i, world(f, x, y));
      }
      TColStd_Array1OfReal weights(1, n);
      for (int i = 1; i <= n; ++i) weights.SetValue(i, rd.next());
      const int m = static_cast<int>(rd.next());
      if (m < 2) fail("a profile spline needs knots");
      TColStd_Array1OfReal knots(1, m);
      TColStd_Array1OfInteger mults(1, m);
      for (int i = 1; i <= m; ++i) knots.SetValue(i, rd.next());
      for (int i = 1; i <= m; ++i) mults.SetValue(i, static_cast<int>(rd.next()));
      const bool periodic = rd.next() != 0.0;
      Handle(Geom_BSplineCurve) bs;
      try {
        bs = new Geom_BSplineCurve(poles, weights, knots, mults, degree, periodic);
      } catch (const Standard_Failure&) {
        fail("a profile spline's knots, multiplicities and poles do not agree");
      }
      if (bs->IsClosed() && periodic) {
        last.Nullify();
        return BRepBuilderAPI_MakeEdge(bs).Edge();
      }
      const gp_Pnt a = bs->StartPoint(), b = bs->EndPoint();
      TopoDS_Vertex va = vertex_at(a, last);
      // A closed spline that is a loop on its own starts and ends on one vertex.
      const bool lone_loop = first.IsNull() && last.IsNull() && a.Distance(b) < 1e-9;
      TopoDS_Vertex vb = lone_loop ? va : end_vertex(b);
      // On one vertex the curve's parameters are given: projecting the vertex would find the
      // start parameter for both ends.
      TopoDS_Edge e = lone_loop ? BRepBuilderAPI_MakeEdge(bs, va, vb, bs->FirstParameter(), bs->LastParameter()).Edge()
                                : BRepBuilderAPI_MakeEdge(bs, va, vb).Edge();
      last = vb;
      return e;
    }
    default:
      unsupported("this profile segment kind");
  }
}
}  // namespace

Profile build_profile(const KInstr& in) {
  if (in.reals.size() < 12 || in.ints.size() < 2) fail("a profile needs a frame and loops");
  Profile p;
  p.frame = frame_from(in.reals.data());
  p.closed = in.ints[0] != 0;
  const std::int64_t nloops = in.ints[1];
  Reader rd{in.reals, 12};
  std::size_t ip = 2;
  std::vector<TopoDS_Wire> wires;
  BRep_Builder bb;
  for (std::int64_t l = 0; l < nloops; ++l) {
    const std::int64_t nseg = in.ints[ip++];
    const std::int64_t start_v = in.ints[ip++];
    const std::int64_t end_v = in.ints[ip++];
    BRepBuilderAPI_MakeWire mw;
    TopoDS_Vertex last, first;
    std::vector<TopoDS_Edge> edges;
    for (std::int64_t s = 0; s < nseg; ++s) {
      const std::int64_t kind = in.ints[ip++];
      const std::uint32_t side = lab(in.ints[ip++]);
      const std::uint32_t cap0 = lab(in.ints[ip++]);
      const std::uint32_t cap1 = lab(in.ints[ip++]);
      TopoDS_Edge e = make_segment(p.frame, kind, rd, last, first, s + 1 == nseg);
      if (s == 0) {
        TopoDS_Vertex a, b;
        TopExp::Vertices(e, a, b, Standard_True);
        first = a;
      }
      if (side != kNoLabel) add_origins(p.tracked.edges, e, label(side));
      p.caps.push_back({e, cap0, cap1});
      mw.Add(e);
      if (!mw.IsDone()) fail("profile curves do not join end to end");
      edges.push_back(e);
    }
    TopoDS_Wire w = mw.Wire();
    if (p.closed && !BRep_Tool::IsClosed(w)) fail("a profile loop is not closed");
    if (!edges.empty()) {
      TopoDS_Vertex a, b;
      TopExp::Vertices(w, a, b);
      if (start_v >= 0 && !a.IsNull()) add_origins(p.tracked.vertices, a, label(lab(start_v)));
      if (end_v >= 0 && !b.IsNull()) add_origins(p.tracked.vertices, b, label(lab(end_v)));
    }
    wires.push_back(w);
  }
  if (wires.empty()) fail("a profile has no loops");
  if (p.closed) {
    const gp_Pln plane(p.frame);
    // Orient loops: outer counter-clockwise about the frame's z, holes clockwise. Wire
    // orientation only; the edges (and their labels) are unchanged.
    auto make_face = [&](const TopoDS_Wire& outer, const std::vector<TopoDS_Wire>& holes) {
      BRepBuilderAPI_MakeFace mf(plane, outer, Standard_True);
      for (const TopoDS_Wire& h : holes) mf.Add(h);
      if (!mf.IsDone()) fail("the profile loops do not make a face");
      Handle(ShapeFix_Face) fix = new ShapeFix_Face(mf.Face());
      fix->FixOrientation();
      TopoDS_Face face = fix->Face();
      BRepLib::BuildCurves3d(face);
      return face;
    };
    // Each loop alone as a face, for its area and to test which loops lie inside it. Loops
    // don't cross (the sketch's regions are separate or nested), so one point of a loop tells.
    const std::size_t n = wires.size();
    std::vector<TopoDS_Face> alone;
    std::vector<double> area(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      alone.push_back(make_face(wires[i], {}));
      GProp_GProps g;
      BRepGProp::SurfaceProperties(alone[i], g);
      area[i] = std::abs(g.Mass());
    }
    auto inside = [&](std::size_t a, std::size_t b) {  // loop a inside loop b's face
      if (area[a] >= area[b]) return false;
      TopExp_Explorer ex(wires[a], TopAbs_EDGE);
      if (!ex.More()) return false;
      BRepAdaptor_Curve c(TopoDS::Edge(ex.Current()));
      const gp_Pnt q = c.Value(0.5 * (c.FirstParameter() + c.LastParameter()));
      BRepClass_FaceClassifier k(alone[b], q, 1e-7);
      return k.State() == TopAbs_IN;
    };
    // Each loop's depth (how many loops hold it) and its parent (the smallest that does).
    std::vector<int> depth(n, 0);
    std::vector<std::ptrdiff_t> parent(n, -1);
    for (std::size_t a = 0; a < n; ++a)
      for (std::size_t b = 0; b < n; ++b)
        if (a != b && inside(a, b)) {
          ++depth[a];
          if (parent[a] < 0 || area[b] < area[static_cast<std::size_t>(parent[a])]) parent[a] = static_cast<std::ptrdiff_t>(b);
        }
    std::vector<std::size_t> outers;
    for (std::size_t i = 0; i < n; ++i)
      if (depth[i] % 2 == 0) outers.push_back(i);
    std::stable_sort(outers.begin(), outers.end(), [&](std::size_t a, std::size_t b) { return area[a] > area[b]; });
    std::vector<TopoDS_Face> faces;
    for (std::size_t o : outers) {
      std::vector<TopoDS_Wire> holes;
      for (std::size_t i = 0; i < n; ++i)
        if (depth[i] % 2 == 1 && parent[i] == static_cast<std::ptrdiff_t>(o)) holes.push_back(wires[i]);
      faces.push_back(make_face(wires[o], holes));
    }
    if (faces.size() == 1) {
      p.tracked.shape = faces.front();
    } else {
      TopoDS_Compound c;
      bb.MakeCompound(c);
      for (const TopoDS_Face& f : faces) bb.Add(c, f);
      p.tracked.shape = c;
    }
  } else {
    TopoDS_Compound c;
    bb.MakeCompound(c);
    for (const TopoDS_Wire& w : wires) bb.Add(c, w);
    p.tracked.shape = c;
    p.tracked.sheet = true;
  }
  return p;
}

// ------------------------------------------------------------------ a face as a profile

// A planar face's boundary in a frame lying in its plane (frame: origin, x, y, z, 12 reals), for
// a prism, revolve or sweep of the face. Returns the loop count, then per loop (the outer wire
// first, then the inner ones) its edge count and each edge in wire order as the PROFILE
// segment encoding above: kind then reals. Lines, circular arcs and whole circles keep their
// kind; any other curve becomes its exact B-spline (kind 6, not periodic).
rust::Vec<double> kernel_face_loops(std::uint64_t body, std::uint32_t face,
                                    rust::Slice<const double> frame) {
  const auto guard = kernel_lock();
  try {
    auto b = registry_get(body);
    if (face >= b->faces.size()) fail("the profile face is not a face of its body");
    if (frame.size() < 12) fail("a face profile needs its frame");
    const TopoDS_Face& f = b->faces[face];
    const gp_XYZ o(frame[0], frame[1], frame[2]);
    const gp_XYZ x(frame[3], frame[4], frame[5]);
    const gp_XYZ y(frame[6], frame[7], frame[8]);
    const gp_XYZ z(frame[9], frame[10], frame[11]);
    BRepAdaptor_Surface surface(f, Standard_False);
    if (surface.GetType() != GeomAbs_Plane) fail("the profile face is not planar");
    const gp_Pln pln = surface.Plane();
    if (std::abs(pln.Axis().Direction().XYZ().Dot(z)) < 1.0 - 1e-9 ||
        std::abs((pln.Location().XYZ() - o).Dot(z)) > 1e-7)
      fail("the profile frame is not in the face's plane");
    auto uv = [&](const gp_Pnt& p, rust::Vec<double>& out) {
      const gp_XYZ d = p.XYZ() - o;
      out.push_back(d.Dot(x));
      out.push_back(d.Dot(y));
    };
    std::vector<TopoDS_Wire> wires;
    const TopoDS_Wire outer = BRepTools::OuterWire(f);
    if (outer.IsNull()) fail("the profile face has no outer boundary");
    wires.push_back(outer);
    for (TopExp_Explorer it(f, TopAbs_WIRE); it.More(); it.Next()) {
      const TopoDS_Wire w = TopoDS::Wire(it.Current());
      if (!w.IsSame(outer)) wires.push_back(w);
    }
    rust::Vec<double> out;
    out.push_back(static_cast<double>(wires.size()));
    for (const TopoDS_Wire& w : wires) {
      rust::Vec<double> loop;
      int count = 0;
      for (BRepTools_WireExplorer ex(w, f); ex.More(); ex.Next()) {
        const TopoDS_Edge& e = ex.Current();
        if (BRep_Tool::Degenerated(e)) continue;
        const bool reversed = e.Orientation() == TopAbs_REVERSED;
        const gp_Pnt ps = BRep_Tool::Pnt(TopExp::FirstVertex(e, Standard_True));
        const gp_Pnt pe = BRep_Tool::Pnt(TopExp::LastVertex(e, Standard_True));
        BRepAdaptor_Curve c(e);
        ++count;
        if (c.GetType() == GeomAbs_Line) {
          loop.push_back(0);
          uv(ps, loop);
          uv(pe, loop);
        } else if (c.GetType() == GeomAbs_Circle) {
          const gp_Circ circ = c.Circle();
          const double span = c.LastParameter() - c.FirstParameter();
          if (span > 2.0 * M_PI - 1e-9) {
            loop.push_back(3);
            uv(circ.Location(), loop);
            loop.push_back(circ.Radius());
          } else {
            const bool ccw = (circ.Axis().Direction().XYZ().Dot(z) > 0) != reversed;
            loop.push_back(ccw ? 1 : 2);
            uv(ps, loop);
            uv(pe, loop);
            uv(circ.Location(), loop);
          }
        } else {
          Standard_Real first = 0, last = 0;
          Handle(Geom_Curve) gc = BRep_Tool::Curve(e, first, last);
          if (gc.IsNull()) fail("a profile face edge has no curve");
          Handle(Geom_BSplineCurve) bs =
              GeomConvert::CurveToBSplineCurve(new Geom_TrimmedCurve(gc, first, last));
          if (bs->IsPeriodic()) bs->SetNotPeriodic();
          if (reversed) bs->Reverse();
          loop.push_back(6);
          loop.push_back(bs->Degree());
          loop.push_back(bs->NbPoles());
          for (int i = 1; i <= bs->NbPoles(); ++i) uv(bs->Pole(i), loop);
          for (int i = 1; i <= bs->NbPoles(); ++i) loop.push_back(bs->Weight(i));
          loop.push_back(bs->NbKnots());
          for (int i = 1; i <= bs->NbKnots(); ++i) loop.push_back(bs->Knot(i));
          for (int i = 1; i <= bs->NbKnots(); ++i) loop.push_back(bs->Multiplicity(i));
          loop.push_back(0);
        }
      }
      out.push_back(count);
      for (double v : loop) out.push_back(v);
    }
    return out;
  } catch (const Standard_Failure& e) {
    fail(std::string("OCCT raised ") + e.GetMessageString());
  }
}

}  // namespace occt_kernel
}  // namespace runmat_geometry_io
