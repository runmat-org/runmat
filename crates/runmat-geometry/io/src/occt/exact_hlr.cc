// Geometry analyses, hidden-line projection and proximity for K1, on exact OCCT geometry.
//
// kernel_analyze kinds (reals in, flat reals out):
//   0 mass properties (density) -> volume, mass, centroid(3), inertia about the centroid(9)
//   1 section (origin 3, normal 3, e1 3, e2 3) -> area, centroid(3), ixx, iyy, ixy in (e1, e2)
//   2 thickness () -> per face (face, min, max) along the inward normal to the next face
//   3 curvature () -> per face (face, min |k|, max |k|) over samples
//   4 zebra () -> per summary edge (edge, 0|1|2 for G0|G1|G2)
//   5 deviation () -> (max, rms) distance from samples of body 0 to body 1
//   6 undercut (pull 3) -> faces whose release path along the pull the body blocks

#include "runmat-geometry-io/src/occt/kernel_ffi.rs.h"
#include "exact_kernel.hxx"

#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepAlgoAPI_Common.hxx>
#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepClass_FaceClassifier.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepGProp.hxx>
#include <BRepLProp_SLProps.hxx>
#include <BRepLib.hxx>
#include <BRepTools.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <GCPnts_QuasiUniformDeflection.hxx>
#include <GProp_GProps.hxx>
#include <GeomAPI_ProjectPointOnSurf.hxx>
#include <HLRAlgo_Projector.hxx>
#include <HLRBRep_Algo.hxx>
#include <HLRBRep_HLRToShape.hxx>
#include <IntCurvesFace_ShapeIntersector.hxx>
#include <LocalAnalysis_SurfaceContinuity.hxx>
#include <Standard_Failure.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <TopLoc_Location.hxx>
#include <Geom2d_Curve.hxx>
#include <gp_Lin.hxx>
#include <gp_Pnt2d.hxx>
#include <gp_Pln.hxx>

#include <cmath>
#include <limits>

namespace runmat_geometry_io {
namespace occt_kernel {
namespace {

struct Sample {
  gp_Pnt p;
  gp_Dir n;  // outward
};

// Points inside a face on a parameter grid, with the outward normal there.
std::vector<Sample> face_samples(const TopoDS_Face& f, int grid = 6) {
  std::vector<Sample> out;
  BRepAdaptor_Surface s(f, Standard_True);
  double u0, u1, v0, v1;
  BRepTools::UVBounds(f, u0, u1, v0, v1);
  BRepClass_FaceClassifier cls;
  for (int i = 0; i < grid; ++i) {
    for (int j = 0; j < grid; ++j) {
      const double u = u0 + (u1 - u0) * (i + 0.5) / grid;
      const double v = v0 + (v1 - v0) * (j + 0.5) / grid;
      cls.Perform(f, gp_Pnt2d(u, v), 1e-9);
      if (cls.State() != TopAbs_IN) continue;
      BRepLProp_SLProps pr(s, u, v, 1, 1e-9);
      if (!pr.IsNormalDefined()) continue;
      gp_Dir n = pr.Normal();
      if (f.Orientation() == TopAbs_REVERSED) n.Reverse();
      out.push_back({pr.Value(), n});
    }
  }
  return out;
}

double first_hit(const TopoDS_Shape& shape, const gp_Pnt& o, const gp_Dir& d, double eps) {
  IntCurvesFace_ShapeIntersector it;
  it.Load(shape, 1e-9);
  it.Perform(gp_Lin(o, d), eps, 1e12);
  double best = std::numeric_limits<double>::infinity();
  if (!it.IsDone()) return best;
  for (int k = 1; k <= it.NbPnt(); ++k) best = std::min(best, it.WParameter(k));
  return best;
}

double scale_of(const TopoDS_Shape& s) {
  Bnd_Box b;
  BRepBndLib::Add(s, b);
  return b.IsVoid() ? 1.0 : std::sqrt(b.SquareExtent());
}

TopoDS_Shape placed(const Body& b, const double* t) {
  gp_Trsf tr = trsf_from(t);
  return BRepBuilderAPI_Transform(b.shape, tr, Standard_True).Shape();
}
}  // namespace

rust::Vec<double> kernel_analyze(std::uint32_t kind, rust::Slice<const double> r,
                                 rust::Slice<const std::uint64_t> ids) {
  const auto guard = kernel_lock();
  try {
    std::vector<std::shared_ptr<const Body>> bodies;
    for (auto id : ids) bodies.push_back(registry_get(id));
    if (bodies.empty()) fail("an analysis needs a body");
    rust::Vec<double> out;
    switch (kind) {
      case 0: {
        GProp_GProps all;
        for (const auto& b : bodies) {
          GProp_GProps g;
          if (b->sheet) fail("mass properties need a solid");
          BRepGProp::VolumeProperties(b->shape, g, Standard_True);
          all.Add(g);
        }
        const double v = all.Mass();
        if (v <= 0) fail("no volume");
        const double rho = r[0];
        const gp_Pnt c = all.CentreOfMass();
        const gp_Mat m = all.MatrixOfInertia();
        out.push_back(v);
        out.push_back(v * rho);
        out.push_back(c.X()); out.push_back(c.Y()); out.push_back(c.Z());
        for (int i = 1; i <= 3; ++i)
          for (int j = 1; j <= 3; ++j) out.push_back(rho * m(i, j));
        return out;
      }
      case 1: {
        const gp_Pnt o(r[0], r[1], r[2]);
        const gp_Dir n(r[3], r[4], r[5]);
        const gp_Vec e1(r[6], r[7], r[8]), e2(r[9], r[10], r[11]);
        GProp_GProps all;
        bool any = false;
        for (const auto& b : bodies) {
          const double size = scale_of(b->shape) * 4.0 + 1.0;
          TopoDS_Face plane = BRepBuilderAPI_MakeFace(gp_Pln(o, n), -size, size, -size, size).Face();
          BRepAlgoAPI_Common cut(b->shape, plane);
          if (!cut.IsDone()) continue;
          for (TopExp_Explorer ex(cut.Shape(), TopAbs_FACE); ex.More(); ex.Next()) {
            GProp_GProps g;
            BRepGProp::SurfaceProperties(ex.Current(), g);
            all.Add(g);
            any = true;
          }
        }
        if (!any || all.Mass() <= 0) fail("the plane does not cut the body");
        const double a = all.Mass();
        const gp_Pnt c = all.CentreOfMass();
        const gp_Mat m = all.MatrixOfInertia();
        // For a plane region about its centroid: ∫ r rᵀ = (tr M / 2) I − M.
        const double half_tr = 0.5 * (m(1, 1) + m(2, 2) + m(3, 3));
        auto rr = [&](const gp_Vec& x, const gp_Vec& y) {
          double s = half_tr * x.Dot(y);
          for (int i = 1; i <= 3; ++i)
            for (int j = 1; j <= 3; ++j) s -= x.Coord(i) * m(i, j) * y.Coord(j);
          return s;
        };
        out.push_back(a);
        out.push_back(c.X()); out.push_back(c.Y()); out.push_back(c.Z());
        out.push_back(rr(e2, e2));
        out.push_back(rr(e1, e1));
        out.push_back(rr(e1, e2));
        return out;
      }
      case 2: {
        const Body& b = *bodies[0];
        const double eps = scale_of(b.shape) * 1e-7;
        for (std::size_t f = 0; f < b.faces.size(); ++f) {
          double lo = std::numeric_limits<double>::infinity(), hi = 0;
          for (const Sample& s : face_samples(b.faces[f])) {
            const double t = first_hit(b.shape, s.p, s.n.Reversed(), eps);
            if (std::isfinite(t)) {
              lo = std::min(lo, t);
              hi = std::max(hi, t);
            }
          }
          if (std::isfinite(lo)) {
            out.push_back(static_cast<double>(f));
            out.push_back(lo);
            out.push_back(hi);
          }
        }
        return out;
      }
      case 3: {
        const Body& b = *bodies[0];
        for (std::size_t f = 0; f < b.faces.size(); ++f) {
          BRepAdaptor_Surface s(b.faces[f], Standard_True);
          double lo = std::numeric_limits<double>::infinity(), hi = 0;
          if (s.GetType() == GeomAbs_Plane) {
            lo = hi = 0;
          } else {
            double u0, u1, v0, v1;
            BRepTools::UVBounds(b.faces[f], u0, u1, v0, v1);
            for (int i = 0; i < 5; ++i)
              for (int j = 0; j < 5; ++j) {
                BRepLProp_SLProps pr(s, u0 + (u1 - u0) * (i + 0.5) / 5, v0 + (v1 - v0) * (j + 0.5) / 5, 2, 1e-9);
                if (!pr.IsCurvatureDefined()) continue;
                const double k1 = std::abs(pr.MinCurvature()), k2 = std::abs(pr.MaxCurvature());
                lo = std::min(lo, std::min(k1, k2));
                hi = std::max(hi, std::max(k1, k2));
              }
            if (!std::isfinite(lo)) lo = 0;
          }
          out.push_back(static_cast<double>(f));
          out.push_back(lo);
          out.push_back(hi);
        }
        return out;
      }
      case 4: {
        const Body& b = *bodies[0];
        TopTools_IndexedDataMapOfShapeListOfShape ef;
        TopExp::MapShapesAndUniqueAncestors(b.shape, TopAbs_EDGE, TopAbs_FACE, ef);
        for (std::size_t e = 0; e < b.edges.size(); ++e) {
          const TopTools_ListOfShape& fs = ef.FindFromKey(b.edges[e]);
          double c = 0;
          if (fs.Extent() == 2) {
            const TopoDS_Face f1 = TopoDS::Face(fs.First()), f2 = TopoDS::Face(fs.Last());
            BRepAdaptor_Curve ec(b.edges[e]);
            const double t = 0.5 * (ec.FirstParameter() + ec.LastParameter());
            const gp_Pnt p = ec.Value(t);
            Handle(Geom_Surface) s1 = BRep_Tool::Surface(f1), s2 = BRep_Tool::Surface(f2);
            GeomAPI_ProjectPointOnSurf p1(p, s1), p2(p, s2);
            if (p1.NbPoints() > 0 && p2.NbPoints() > 0) {
              double u1, v1, u2, v2;
              p1.LowerDistanceParameters(u1, v1);
              p2.LowerDistanceParameters(u2, v2);
              LocalAnalysis_SurfaceContinuity sc(s1, u1, v1, s2, u2, v2, GeomAbs_G2, 1e-6, 1e-3, 1e-3, 1e-3, 0.01);
              if (sc.IsDone()) {
                if (sc.IsG2()) c = 2;
                else if (sc.IsG1()) c = 1;
              } else {
                LocalAnalysis_SurfaceContinuity s1c(s1, u1, v1, s2, u2, v2, GeomAbs_G1, 1e-6, 1e-3);
                if (s1c.IsDone() && s1c.IsG1()) c = 1;
              }
            }
          }
          out.push_back(static_cast<double>(e));
          out.push_back(c);
        }
        return out;
      }
      case 5: {
        if (bodies.size() < 2) fail("a deviation needs two bodies");
        const Body& a = *bodies[0];
        const Body& b = *bodies[1];
        double mx = 0, sum = 0;
        std::size_t n = 0;
        for (const TopoDS_Face& f : a.faces) {
          for (const Sample& s : face_samples(f, 4)) {
            BRepExtrema_DistShapeShape d(BRepBuilderAPI_MakeVertex(s.p).Vertex(), b.shape);
            if (!d.IsDone()) continue;
            mx = std::max(mx, d.Value());
            sum += d.Value() * d.Value();
            ++n;
          }
        }
        if (n == 0) fail("nothing to compare");
        out.push_back(mx);
        out.push_back(std::sqrt(sum / static_cast<double>(n)));
        return out;
      }
      case 6: {
        const Body& b = *bodies[0];
        const gp_Dir pull(r[0], r[1], r[2]);
        const double eps = scale_of(b.shape) * 1e-7;
        for (std::size_t f = 0; f < b.faces.size(); ++f) {
          // The body without this face blocks the release when a ray along (or against) the
          // pull, from where the face faces that way, meets it.
          TopoDS_Compound others;
          BRep_Builder bb;
          bb.MakeCompound(others);
          for (std::size_t g = 0; g < b.faces.size(); ++g)
            if (g != f) bb.Add(others, b.faces[g]);
          bool blocked = false;
          for (const Sample& s : face_samples(b.faces[f], 4)) {
            const double along = s.n.Dot(pull);
            if (std::abs(along) < 1e-9) continue;
            const gp_Dir d = along > 0 ? pull : pull.Reversed();
            if (std::isfinite(first_hit(others, s.p, d, eps))) {
              blocked = true;
              break;
            }
          }
          if (blocked) out.push_back(static_cast<double>(f));
        }
        return out;
      }
      default:
        unsupported("this analysis");
    }
  } catch (const Standard_Failure& e) {
    fail(std::string("OCCT raised ") + e.GetMessageString());
  }
}

// ------------------------------------------------------------------------- hidden lines

// True when face `f` is edge-on to the viewer all along edge `e` (its normal, placed by `tr`,
// is perpendicular to `to_viewer` at a quarter, the middle and three quarters of the edge): a
// tangent edge there is the view's outline, as where a lug's rounded end runs into its flat
// sides, seen along those sides. HLR's outline pass gives no line where the silhouette falls
// on an edge, so the edge itself draws it.
static bool on_silhouette(const TopoDS_Edge& e, const TopoDS_Face& f, const gp_Trsf& tr, const gp_Dir& to_viewer) {
  double first = 0, last = 0;
  Handle(Geom2d_Curve) pc = BRep_Tool::CurveOnSurface(e, f, first, last);
  if (pc.IsNull() || !std::isfinite(first) || !std::isfinite(last)) return false;
  BRepAdaptor_Surface surface(f);
  for (const double t : {0.25, 0.5, 0.75}) {
    const gp_Pnt2d uv = pc->Value(first + (last - first) * t);
    BRepLProp_SLProps props(surface, uv.X(), uv.Y(), 1, 1e-9);
    if (!props.IsNormalDefined()) return false;
    if (std::abs(props.Normal().Transformed(tr).Dot(to_viewer)) > 1e-6) return false;
  }
  return true;
}

rust::Vec<KProjected> kernel_project(rust::Slice<const std::uint64_t> ids,
                                     rust::Slice<const double> transforms,
                                     rust::Slice<const double> frame_r, std::uint32_t flags,
                                     double tolerance) {
  const auto guard = kernel_lock();
  try {
    const bool hidden = flags & 1;
    const bool smooth = flags & 2;
    const gp_Ax3 frame = frame_from(frame_r.data());
    const gp_Dir to_viewer = frame.Direction();
    const gp_Pnt o = frame.Location();
    const gp_Dir X = frame.XDirection(), Y = frame.YDirection();
    auto to2d = [&](const gp_Pnt& p) {
      const gp_Vec v(o, p);
      return std::make_pair(v.Dot(gp_Vec(X)), v.Dot(gp_Vec(Y)));
    };
    std::vector<std::shared_ptr<const Body>> bodies;
    std::vector<TopoDS_Shape> world;
    TopoDS_Compound scene;
    BRep_Builder bb;
    bb.MakeCompound(scene);
    for (std::size_t i = 0; i < ids.size(); ++i) {
      bodies.push_back(registry_get(ids[i]));
      world.push_back(placed(*bodies.back(), transforms.data() + 12 * i));
      bb.Add(scene, world.back());
    }
    const double eps = scale_of(scene) * 1e-7;
    IntCurvesFace_ShapeIntersector occluder;
    occluder.Load(scene, 1e-9);
    auto visible_at = [&](const gp_Pnt& p) {
      occluder.Perform(gp_Lin(gp_Pnt(p.XYZ() + to_viewer.XYZ() * eps), to_viewer), 0.0, 1e12);
      if (!occluder.IsDone()) return true;
      for (int k = 1; k <= occluder.NbPnt(); ++k)
        if (occluder.WParameter(k) > eps) return false;
      return true;
    };
    rust::Vec<KProjected> out;
    for (std::size_t bi = 0; bi < bodies.size(); ++bi) {
      TopoDS_Shape s = world[bi];
      BRepLib::EncodeRegularity(s, 1e-6);
      // Summary edges of the placed copy, in the body's summary order.
      const Body& body = *bodies[bi];
      gp_Trsf tr = trsf_from(transforms.data() + 12 * bi);
      TopTools_IndexedDataMapOfShapeListOfShape ef;
      TopExp::MapShapesAndUniqueAncestors(body.shape, TopAbs_EDGE, TopAbs_FACE, ef);
      BRepLib::EncodeRegularity(body.shape, 1e-6);
      for (std::size_t ei = 0; ei < body.edges.size(); ++ei) {
        TopoDS_Edge e = TopoDS::Edge(body.edges[ei].Moved(TopLoc_Location(tr)));
        if (!smooth) {
          // Smooth edges between tangent faces are left out unless asked for.
          bool is_smooth = false;
          const TopTools_ListOfShape& fs = ef.FindFromKey(body.edges[ei]);
          if (fs.Extent() == 2)
            is_smooth = BRep_Tool::Continuity(body.edges[ei], TopoDS::Face(fs.First()), TopoDS::Face(fs.Last())) >= GeomAbs_G1 &&
                        !on_silhouette(TopoDS::Edge(body.edges[ei]), TopoDS::Face(fs.First()), tr, to_viewer);
          if (is_smooth) continue;
        }
        BRepAdaptor_Curve c(e);
        GCPnts_QuasiUniformDeflection pts(c, std::max(tolerance, 1e-9));
        if (!pts.IsDone() || pts.NbPoints() < 2) continue;
        KProjected run;
        bool have = false;
        bool vis = true;
        auto flush = [&]() {
          if (have && run.points.size() >= 4) {
            const double dx = run.points[run.points.size() - 2] - run.points[0];
            const double dy = run.points[run.points.size() - 1] - run.points[1];
            double extent = std::abs(dx) + std::abs(dy);
            for (std::size_t k = 2; k + 1 < run.points.size(); k += 2)
              extent = std::max(extent, std::abs(run.points[k] - run.points[0]) + std::abs(run.points[k + 1] - run.points[1]));
            if (extent > eps && (vis || hidden)) out.push_back(run);
          }
          run = KProjected();
          have = false;
        };
        auto add = [&](const gp_Pnt& p) {
          const auto q = to2d(p);
          run.points.push_back(q.first);
          run.points.push_back(q.second);
        };
        auto start = [&](const gp_Pnt& p, bool v) {
          run.body = static_cast<std::uint32_t>(bi);
          run.kind = 0;
          run.entity = static_cast<std::uint32_t>(ei);
          run.visible = v;
          vis = v;
          have = true;
          add(p);
        };
        // Visibility is tested along the curve at most `step` apart (the chord points of a
        // straight edge are only its ends); where it changes, the change is found by bisection
        // to the tolerance, and the run ends and the next starts there.
        const double step = std::max(scale_of(scene) / 256.0, std::max(tolerance, 1e-9));
        double prev_u = pts.Parameter(1);
        gp_Pnt prev = c.Value(prev_u);
        start(prev, visible_at(prev));
        for (int k = 2; k <= pts.NbPoints(); ++k) {
          const double next_u = pts.Parameter(k);
          const gp_Pnt next = c.Value(next_u);
          const int pieces = std::max(1, std::min(512, static_cast<int>(std::ceil(prev.Distance(next) / step))));
          for (int j = 1; j <= pieces; ++j) {
            const double ua = prev_u + (next_u - prev_u) * (static_cast<double>(j - 1) / pieces);
            const double ub = prev_u + (next_u - prev_u) * (static_cast<double>(j) / pieces);
            const bool v = visible_at(c.Value(ub));
            if (v != vis) {
              double lo = ua, hi = ub;
              for (int it = 0; it < 40 && c.Value(lo).Distance(c.Value(hi)) > std::max(tolerance * 0.1, eps); ++it) {
                const double mid = 0.5 * (lo + hi);
                if (visible_at(c.Value(mid)) == vis) lo = mid; else hi = mid;
              }
              const gp_Pnt at = c.Value(0.5 * (lo + hi));
              add(at);
              flush();
              start(at, v);
            }
          }
          add(next);
          prev = next;
          prev_u = next_u;
        }
        flush();
      }
      // Silhouettes of curved faces from OCCT's exact hidden-line algorithm, attributed to
      // the face the outline lies on.
      Handle(HLRBRep_Algo) algo = new HLRBRep_Algo();
      algo->Add(s);
      algo->Projector(HLRAlgo_Projector(gp_Ax2(o, to_viewer, X)));
      algo->Update();
      algo->Hide();
      HLRBRep_HLRToShape h(algo);
      for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1 && !hidden) break;
        TopoDS_Shape outlines = pass == 0 ? h.OutLineVCompound() : h.OutLineHCompound();
        if (outlines.IsNull()) continue;
        for (TopExp_Explorer ex(outlines, TopAbs_EDGE); ex.More(); ex.Next()) {
          BRepAdaptor_Curve c(TopoDS::Edge(ex.Current()));
          GCPnts_QuasiUniformDeflection pts(c, std::max(tolerance, 1e-9));
          if (!pts.IsDone() || pts.NbPoints() < 2) continue;
          // The outline curve's points are in the projector's plane coordinates.
          const gp_Pnt mid = c.Value(0.5 * (c.FirstParameter() + c.LastParameter()));
          const gp_Pnt lifted(o.XYZ() + X.XYZ() * mid.X() + Y.XYZ() * mid.Y());
          TopoDS_Edge line = BRepBuilderAPI_MakeEdge(gp_Lin(lifted, to_viewer), -1e6, 1e6).Edge();
          double best = std::numeric_limits<double>::infinity();
          std::size_t face = 0;
          for (std::size_t fi = 0; fi < body.faces.size(); ++fi) {
            BRepAdaptor_Surface fs(body.faces[fi]);
            if (fs.GetType() == GeomAbs_Plane) continue;
            BRepExtrema_DistShapeShape d(line, body.faces[fi].Moved(TopLoc_Location(tr)));
            if (d.IsDone() && d.Value() < best) {
              best = d.Value();
              face = fi;
            }
          }
          if (!std::isfinite(best) || best > std::max(tolerance, eps) * 10) continue;
          KProjected k;
          k.body = static_cast<std::uint32_t>(bi);
          k.kind = 1;
          k.entity = static_cast<std::uint32_t>(face);
          k.visible = pass == 0;
          for (int i = 1; i <= pts.NbPoints(); ++i) {
            k.points.push_back(pts.Value(i).X());
            k.points.push_back(pts.Value(i).Y());
          }
          out.push_back(std::move(k));
        }
      }
    }
    return out;
  } catch (const Standard_Failure& e) {
    fail(std::string("OCCT raised ") + e.GetMessageString());
  }
}

// ------------------------------------------------------------------------- proximity

rust::Vec<double> kernel_proximity(std::uint64_t a, rust::Slice<const double> ta, std::uint64_t b,
                                   rust::Slice<const double> tb) {
  const auto guard = kernel_lock();
  try {
    auto ba = registry_get(a);
    auto bb = registry_get(b);
    TopoDS_Shape sa = placed(*ba, ta.data());
    TopoDS_Shape sb = placed(*bb, tb.data());
    BRepExtrema_DistShapeShape d(sa, sb);
    if (!d.IsDone()) fail("the distance between the bodies could not be computed");
    double volume = 0;
    const double eps = std::max(scale_of(sa), scale_of(sb)) * 1e-9;
    double dist = d.Value();
    if (dist <= eps) {
      dist = 0;
      if (!ba->sheet && !bb->sheet) {
        BRepAlgoAPI_Common c(sa, sb);
        if (c.IsDone()) {
          GProp_GProps g;
          BRepGProp::VolumeProperties(c.Shape(), g);
          volume = std::max(0.0, g.Mass());
        }
      }
    }
    rust::Vec<double> out;
    out.push_back(dist);
    out.push_back(volume);
    return out;
  } catch (const Standard_Failure& e) {
    fail(std::string("OCCT raised ") + e.GetMessageString());
  }
}

}  // namespace occt_kernel
}  // namespace runmat_geometry_io
