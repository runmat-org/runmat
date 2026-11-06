#include "runmat-geometry-io/src/occt/kernel_ffi.rs.h"
#include "exact_kernel.hxx"

#include <BOPAlgo_Options.hxx>
#include <BRepAlgo_Image.hxx>
#include <IntCurveSurface_TransitionOnCurve.hxx>
#include <TColgp_Array1OfPnt2d.hxx>
#include <TopTools_MapOfShape.hxx>
#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepAlgoAPI_Common.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Defeaturing.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepAlgoAPI_Splitter.hxx>
#include <BRepClass_FaceClassifier.hxx>
#include <GeomAPI_ProjectPointOnSurf.hxx>
#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_Copy.hxx>
#include <BRepBuilderAPI_GTransform.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepOffsetAPI_MakeFilling.hxx>
#include <BRepBuilderAPI_MakeSolid.hxx>
#include <BRepBuilderAPI_Sewing.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepCheck_ListOfStatus.hxx>
#include <BRepCheck_Result.hxx>
#include <TopAbs.hxx>
#include <BRepClass3d_SolidClassifier.hxx>
#include <BRepFilletAPI_MakeChamfer.hxx>
#include <BRepFilletAPI_MakeFillet.hxx>
#include <BRepGProp.hxx>
#include <BRepGProp_Face.hxx>
#include <BRepLib.hxx>
#include <BRepOffsetAPI_DraftAngle.hxx>
#include <BRepOffset_MakeOffset.hxx>
#include <BRepOffset_MakeSimpleOffset.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakeHalfSpace.hxx>
#include <BRepTools.hxx>
#include <BRepTools_ReShape.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <GCPnts_AbscissaPoint.hxx>
#include <GProp_GProps.hxx>
#include <GeomAbs_SurfaceType.hxx>
#include <GeomLib_IsPlanarSurface.hxx>
#include <Geom2d_Curve.hxx>
#include <Geom_Plane.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <TColStd_Array1OfReal.hxx>
#include <Geom_BezierCurve.hxx>
#include <IntCurvesFace_ShapeIntersector.hxx>
#include <ShapeUpgrade_UnifySameDomain.hxx>
#include <Standard_Failure.hxx>
#include <Standard_Version.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Iterator.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Shell.hxx>
#include <TopoDS_Solid.hxx>
#include <gp.hxx>
#include <gp_GTrsf.hxx>
#include <gp_Lin.hxx>
#include <gp_Pln.hxx>

#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <cmath>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <sstream>

namespace runmat_geometry_io {
namespace occt_kernel {

// A label a later step removes (the Rust side's INTERNAL).
constexpr std::uint32_t kInternalLabel = 0xfffffffeu;

// ------------------------------------------------------------------------- errors

void fail(const std::string& message) { throw std::runtime_error("failed: " + message); }
void unsupported(const std::string& message) {
  throw std::runtime_error("unsupported: " + message);
}
void incomplete(const std::string& message) { throw std::runtime_error("history: " + message); }

// ------------------------------------------------------------------------- origins

void merge_into(Origins& into, const Origins& from) {
  if (from.empty()) return;
  Origins out;
  out.reserve(into.size() + from.size());
  std::set_union(into.begin(), into.end(), from.begin(), from.end(), std::back_inserter(out));
  into.swap(out);
}

Origins label(std::uint32_t id) {
  Origin o;
  o.kind = kLabel;
  o.a = id;
  return Origins{o};
}

static const Origins kEmpty;

const Origins& origins_of(const ShapeOrigins& map, const TopoDS_Shape& s) {
  const Origins* o = map.Seek(s);
  return o ? *o : kEmpty;
}

void add_origins(ShapeOrigins& map, const TopoDS_Shape& s, const Origins& o) {
  if (Origins* existing = map.ChangeSeek(s)) {
    merge_into(*existing, o);
  } else {
    map.Add(s, o);
  }
}

// ------------------------------------------------------------------------- registry

namespace {
std::mutex& registry_mutex() {
  static std::mutex m;
  return m;
}
std::map<std::uint64_t, std::shared_ptr<const Body>>& registry() {
  static std::map<std::uint64_t, std::shared_ptr<const Body>> r;
  return r;
}
std::atomic<std::uint64_t> next_id{1};
}  // namespace

std::shared_ptr<const Body> registry_get(std::uint64_t id) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto it = registry().find(id);
  if (it == registry().end()) fail("operand is not a live OCCT kernel body");
  return it->second;
}

std::uint64_t registry_put(std::shared_ptr<Body> body) {
  const std::uint64_t id = next_id.fetch_add(1);
  std::lock_guard<std::mutex> lock(registry_mutex());
  registry()[id] = std::move(body);
  return id;
}

void kernel_release(std::uint64_t id) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  registry().erase(id);
}

std::unique_lock<std::recursive_mutex> kernel_lock() {
  static std::recursive_mutex m;
  return std::unique_lock<std::recursive_mutex>(m);
}

std::uint64_t kernel_live_bodies() {
  std::lock_guard<std::mutex> lock(registry_mutex());
  return registry().size();
}

rust::String kernel_occt_version() { return rust::String(OCC_VERSION_COMPLETE); }

// ------------------------------------------------------------------------- summaries

static bool summary_edge(const TopoDS_Edge& e, const TopTools_ListOfShape& faces) {
  if (BRep_Tool::Degenerated(e)) return false;
  for (const TopoDS_Shape& f : faces) {
    if (BRep_Tool::IsClosed(e, TopoDS::Face(f))) return false;  // a seam
  }
  return true;
}

std::shared_ptr<Body> summarize(const TopoDS_Shape& shape, bool sheet) {
  auto body = std::make_shared<Body>();
  body->shape = shape;
  body->sheet = sheet;
  TopTools_IndexedMapOfShape faces;
  TopExp::MapShapes(shape, TopAbs_FACE, faces);
  for (int i = 1; i <= faces.Extent(); ++i) body->faces.push_back(TopoDS::Face(faces(i)));
  TopTools_IndexedDataMapOfShapeListOfShape edge_faces;
  TopExp::MapShapesAndUniqueAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_faces);
  TopTools_IndexedMapOfShape kept_edges;
  for (int i = 1; i <= edge_faces.Extent(); ++i) {
    const TopoDS_Edge& e = TopoDS::Edge(edge_faces.FindKey(i));
    if (summary_edge(e, edge_faces(i))) {
      body->edges.push_back(e);
      kept_edges.Add(e);
    }
  }
  TopTools_IndexedDataMapOfShapeListOfShape vertex_edges;
  TopExp::MapShapesAndUniqueAncestors(shape, TopAbs_VERTEX, TopAbs_EDGE, vertex_edges);
  for (int i = 1; i <= vertex_edges.Extent(); ++i) {
    bool any = false;
    for (const TopoDS_Shape& e : vertex_edges(i)) any |= kept_edges.Contains(e);
    if (any) body->vertices.push_back(TopoDS::Vertex(vertex_edges.FindKey(i)));
  }
  return body;
}

Tracked track_operand(const Body& body, std::uint32_t operand) {
  Tracked t;
  t.shape = body.shape;
  t.sheet = body.sheet;
  for (std::size_t i = 0; i < body.faces.size(); ++i) {
    Origin o;
    o.kind = kInput;
    o.a = operand;
    o.b = static_cast<std::uint32_t>(i);
    add_origins(t.faces, body.faces[i], Origins{o});
    t.kept.Add(body.faces[i], 1);
  }
  for (std::size_t i = 0; i < body.edges.size(); ++i) {
    Origin o;
    o.kind = kInput;
    o.a = operand;
    o.b = static_cast<std::uint32_t>(i);
    add_origins(t.edges, body.edges[i], Origins{o});
  }
  for (std::size_t i = 0; i < body.vertices.size(); ++i) {
    Origin o;
    o.kind = kInput;
    o.a = operand;
    o.b = static_cast<std::uint32_t>(i);
    add_origins(t.vertices, body.vertices[i], Origins{o});
  }
  return t;
}

// ------------------------------------------------------------------------- propagation

namespace {
template <class History>
void carry(const ShapeOrigins& from, TopAbs_ShapeEnum kind, History& h,
           const TopTools_IndexedMapOfShape& present, ShapeOrigins& to) {
  for (int i = 1; i <= from.Extent(); ++i) {
    const TopoDS_Shape& s = from.FindKey(i);
    const Origins& o = from(i);
    const TopTools_ListOfShape& mods = h.Modified(s);
    bool any = false;
    for (const TopoDS_Shape& m : mods) {
      if (m.ShapeType() != kind) {
        // A face modified into several pieces arrives as faces; anything else (a face
        // modified into a shell) is unpacked.
        TopExp_Explorer ex(m, kind);
        for (; ex.More(); ex.Next()) {
          if (present.Contains(ex.Current())) {
            add_origins(to, ex.Current(), o);
            any = true;
          }
        }
        continue;
      }
      if (present.Contains(m)) {
        add_origins(to, m, o);
        any = true;
      }
    }
    if (!any && !h.IsDeleted(s) && present.Contains(s)) add_origins(to, s, o);
  }
}

struct HistoryAdapter {
  Handle(BRepTools_History) h;
  const TopTools_ListOfShape& Modified(const TopoDS_Shape& s) { return h->Modified(s); }
  const TopTools_ListOfShape& Generated(const TopoDS_Shape& s) { return h->Generated(s); }
  bool IsDeleted(const TopoDS_Shape& s) { return h->IsRemoved(s); }
};
}  // namespace

void propagate(const Tracked& in, BRepBuilderAPI_MakeShape& h, Tracked& out) {
  TopTools_IndexedMapOfShape faces, edges, vertices;
  TopExp::MapShapes(out.shape, TopAbs_FACE, faces);
  TopExp::MapShapes(out.shape, TopAbs_EDGE, edges);
  TopExp::MapShapes(out.shape, TopAbs_VERTEX, vertices);
  carry(in.faces, TopAbs_FACE, h, faces, out.faces);
  carry(in.edges, TopAbs_EDGE, h, edges, out.edges);
  carry(in.vertices, TopAbs_VERTEX, h, vertices, out.vertices);
  for (int i = 1; i <= in.kept.Extent(); ++i) {
    const TopoDS_Shape& f = in.kept.FindKey(i);
    if (faces.Contains(f) && h.Modified(f).IsEmpty() && !h.IsDeleted(f)) out.kept.Add(f, 1);
  }
}

void propagate_history(const Tracked& in, const Handle(BRepTools_History) & h, Tracked& out) {
  if (h.IsNull()) incomplete("an OCCT algorithm returned no history");
  HistoryAdapter a{h};
  TopTools_IndexedMapOfShape faces, edges, vertices;
  TopExp::MapShapes(out.shape, TopAbs_FACE, faces);
  TopExp::MapShapes(out.shape, TopAbs_EDGE, edges);
  TopExp::MapShapes(out.shape, TopAbs_VERTEX, vertices);
  carry(in.faces, TopAbs_FACE, a, faces, out.faces);
  carry(in.edges, TopAbs_EDGE, a, edges, out.edges);
  carry(in.vertices, TopAbs_VERTEX, a, vertices, out.vertices);
  for (int i = 1; i <= in.kept.Extent(); ++i) {
    const TopoDS_Shape& f = in.kept.FindKey(i);
    if (faces.Contains(f) && a.Modified(f).IsEmpty() && !a.IsDeleted(f)) out.kept.Add(f, 1);
  }
}

void propagate_draft(const Tracked& src, BRepOffsetAPI_DraftAngle& mk, Tracked& out) {
  // DraftAngle reports a drafted face only through Generated and an edge or vertex only
  // through ModifiedShape, so carry every tracked entity through ModifiedShape (the image the
  // modifier built for it, the shape itself when unchanged).
  TopTools_IndexedMapOfShape present[3];
  const TopAbs_ShapeEnum kinds[3] = {TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX};
  const ShapeOrigins* from[3] = {&src.faces, &src.edges, &src.vertices};
  ShapeOrigins* to[3] = {&out.faces, &out.edges, &out.vertices};
  for (int k = 0; k < 3; ++k) {
    TopExp::MapShapes(out.shape, kinds[k], present[k]);
    for (int i = 1; i <= from[k]->Extent(); ++i) {
      const TopoDS_Shape& s0 = from[k]->FindKey(i);
      TopoDS_Shape m;
      try {
        m = mk.ModifiedShape(s0);
      } catch (const Standard_Failure&) {
        continue;
      }
      if (m.IsNull()) continue;
      for (TopExp_Explorer ex(m, kinds[k]); ex.More(); ex.Next())
        if (present[k].Contains(ex.Current())) add_origins(*to[k], ex.Current(), (*from[k])(i));
    }
  }
  for (int i = 1; i <= src.kept.Extent(); ++i)
    if (present[0].Contains(src.kept.FindKey(i))) out.kept.Add(src.kept.FindKey(i), 1);
}

// ------------------------------------------------------------------------- frames

gp_Ax3 frame_from(const double* r) {
  return gp_Ax3(gp_Pnt(r[0], r[1], r[2]), gp_Dir(r[9], r[10], r[11]), gp_Dir(r[3], r[4], r[5]));
}

gp_Trsf trsf_from(const double* r) {
  gp_Trsf t;
  // SetValues takes the 3x4 matrix row by row; reflections are accepted (det = -1).
  t.SetValues(r[0], r[1], r[2], r[9], r[3], r[4], r[5], r[10], r[6], r[7], r[8], r[11]);
  return t;
}

// ------------------------------------------------------------------------- booleans

Tracked boolean_op(int op, const std::vector<const Tracked*>& args,
                   const std::vector<const Tracked*>& tools) {
  if (args.empty()) fail("a boolean needs a target");
  if (tools.empty()) {
    Tracked t = *args.front();
    return t;
  }
  if (op != 0 && args.size() > 1) {
    // A cut or common applies to each target on its own: one general fuse over several
    // targets would also split the targets against each other where they overlap.
    BRep_Builder bb;
    TopoDS_Compound c;
    bb.MakeCompound(c);
    Tracked out;
    out.sheet = args.front()->sheet;
    for (const Tracked* t : args) {
      Tracked one = boolean_op(op, {t}, tools);
      bb.Add(c, one.shape);
      for (int i = 1; i <= one.faces.Extent(); ++i) add_origins(out.faces, one.faces.FindKey(i), one.faces(i));
      for (int i = 1; i <= one.edges.Extent(); ++i) add_origins(out.edges, one.edges.FindKey(i), one.edges(i));
      for (int i = 1; i <= one.vertices.Extent(); ++i)
        add_origins(out.vertices, one.vertices.FindKey(i), one.vertices(i));
      for (int i = 1; i <= one.kept.Extent(); ++i) out.kept.Add(one.kept.FindKey(i), 1);
    }
    out.shape = c;
    return out;
  }
  TopTools_ListOfShape a, b;
  for (const Tracked* t : args) a.Append(t->shape);
  for (const Tracked* t : tools) b.Append(t->shape);
  std::unique_ptr<BRepAlgoAPI_BooleanOperation> bop;
  switch (op) {
    case 0: bop = std::make_unique<BRepAlgoAPI_Fuse>(); break;
    case 1: bop = std::make_unique<BRepAlgoAPI_Cut>(); break;
    case 2: bop = std::make_unique<BRepAlgoAPI_Common>(); break;
    default: fail("unknown boolean");
  }
  bop->SetArguments(a);
  bop->SetTools(b);
  bop->SetRunParallel(Standard_False);
  bop->SetNonDestructive(Standard_True);
  bop->Build();
  if (!bop->IsDone() || bop->HasErrors()) {
    std::ostringstream m;
    bop->DumpErrors(m);
    fail("boolean failed: " + m.str());
  }
  Tracked out;
  out.shape = bop->Shape();
  out.sheet = args.front()->sheet;
  for (const Tracked* t : args) propagate(*t, *bop, out);
  for (const Tracked* t : tools) propagate(*t, *bop, out);
  return out;
}

// ------------------------------------------------------------------------- same-domain merge

namespace {
bool side_curve(const Origins& o, const MergePolicy& p) {
  if (o.empty()) return false;
  for (const Origin& x : o) {
    if (x.kind != kLabel) return false;
    if (!std::binary_search(p.side_labels.begin(), p.side_labels.end(), x.a)) return false;
  }
  return true;
}

// The contract's flush-face rule (the reference kernel's `coplanar_may_merge`): faces of
// different sources merge where they are flush; two faces already separate in one operand
// stay separate; side faces of distinct profile curves merge only when the definition asks.
bool may_merge(const Origins& a, const Origins& b, const MergePolicy& p) {
  for (const Origin& x : a) {
    if (std::binary_search(b.begin(), b.end(), x)) return true;
  }
  if (side_curve(a, p) && side_curve(b, p)) return p.merge_collinear;
  auto operand_of = [](const Origins& o, std::uint32_t& out) {
    bool have = false;
    for (const Origin& x : o) {
      if (x.kind != kInput) return false;
      if (have && x.a != out) return false;
      out = x.a;
      have = true;
    }
    return have;
  };
  std::uint32_t x = 0, y = 0;
  if (operand_of(a, x) && operand_of(b, y)) return x != y;
  return true;
}
}  // namespace

Tracked merge_same_domain(const Tracked& in, const MergePolicy& policy) {
  TopTools_IndexedDataMapOfShapeListOfShape edge_faces;
  TopExp::MapShapesAndUniqueAncestors(in.shape, TopAbs_EDGE, TopAbs_FACE, edge_faces);
  ShapeUpgrade_UnifySameDomain usd(in.shape, Standard_True, Standard_True, Standard_False);
  usd.AllowInternalEdges(Standard_False);
  bool any_mergeable = false;
  TopTools_MapOfShape keep;
  for (int i = 1; i <= edge_faces.Extent(); ++i) {
    const TopTools_ListOfShape& fs = edge_faces(i);
    if (fs.Extent() != 2) {
      keep.Add(edge_faces.FindKey(i));
      continue;
    }
    const Origins& a = origins_of(in.faces, fs.First());
    const Origins& b = origins_of(in.faces, fs.Last());
    if (may_merge(a, b, policy)) {
      any_mergeable = true;
    } else {
      keep.Add(edge_faces.FindKey(i));
    }
  }
  if (!any_mergeable) return in;
  usd.KeepShapes(keep);
  usd.Build();
  Tracked out;
  out.shape = usd.Shape();
  out.sheet = in.sheet;
  propagate_history(in, usd.History(), out);
  return out;
}

// ------------------------------------------------------------------------- geometry summaries

namespace {
void push3(rust::Vec<double>& v, const gp_XYZ& p) {
  v.push_back(p.X());
  v.push_back(p.Y());
  v.push_back(p.Z());
}

KFace face_payload(const TopoDS_Face& f) {
  KFace k;
  k.kept = false;
  BRepAdaptor_Surface s(f, Standard_True);
  const bool reversed = f.Orientation() == TopAbs_REVERSED;
  double g[17] = {0};
  k.surface = 2;
  if (s.GetType() == GeomAbs_Plane) {
    gp_Dir n = s.Plane().Axis().Direction();
    if (!s.Plane().Direct()) n.Reverse();
    if (reversed) n.Reverse();
    g[0] = n.X(); g[1] = n.Y(); g[2] = n.Z();
    k.surface = 0;
  } else if (s.GetType() == GeomAbs_Cylinder) {
    const gp_Cylinder c = s.Cylinder();
    g[0] = c.Location().X(); g[1] = c.Location().Y(); g[2] = c.Location().Z();
    g[3] = c.Axis().Direction().X(); g[4] = c.Axis().Direction().Y();
    g[5] = c.Axis().Direction().Z(); g[6] = c.Radius();
    k.surface = 1;
  } else if (s.GetType() == GeomAbs_BSplineSurface || s.GetType() == GeomAbs_BezierSurface ||
             s.GetType() == GeomAbs_SurfaceOfExtrusion || s.GetType() == GeomAbs_OffsetSurface) {
    // A free-form patch that is in fact flat (a scaled or transformed plane) reports a plane.
    Handle(Geom_Surface) gs = BRep_Tool::Surface(f);
    GeomLib_IsPlanarSurface planar(gs, 1e-9);
    if (planar.IsPlanar()) {
      gp_Dir n = planar.Plan().Axis().Direction();
      // Orient by the surface normal at the face's parametric middle.
      gp_Pnt p;
      gp_Vec du, dv;
      s.D1(0.5 * (s.FirstUParameter() + s.LastUParameter()),
           0.5 * (s.FirstVParameter() + s.LastVParameter()), p, du, dv);
      gp_Vec nn = du.Crossed(dv);
      if (nn.Magnitude() > 0 && nn.Dot(gp_Vec(n)) < 0) n.Reverse();
      if (reversed) n.Reverse();
      g[0] = n.X(); g[1] = n.Y(); g[2] = n.Z();
      k.surface = 0;
    }
  }
  GProp_GProps props;
  BRepGProp::SurfaceProperties(f, props);
  const gp_Pnt c = props.CentreOfMass();
  g[7] = c.X(); g[8] = c.Y(); g[9] = c.Z();
  g[10] = props.Mass();
  Bnd_Box box;
  BRepBndLib::AddOptimal(f, box, Standard_False, Standard_False);
  if (!box.IsVoid()) {
    double x0, y0, z0, x1, y1, z1;
    box.Get(x0, y0, z0, x1, y1, z1);
    g[11] = x0; g[12] = y0; g[13] = z0; g[14] = x1; g[15] = y1; g[16] = z1;
  }
  for (double x : g) k.geom.push_back(x);
  return k;
}

void push_origins(rust::Vec<KOrigin>& v, const Origins& o) {
  for (const Origin& x : o) {
    KOrigin k;
    k.kind = x.kind;
    k.a = x.a;
    k.b = x.b;
    k.c = x.c;
    v.push_back(k);
  }
}

KBody body_payload(const Tracked& t, const TopoDS_Shape& piece, bool sheet) {
  auto body = summarize(piece, sheet);
  KBody k;
  k.sheet = sheet;
  TopTools_IndexedMapOfShape face_index;
  for (const TopoDS_Face& f : body->faces) face_index.Add(f);
  for (const TopoDS_Face& f : body->faces) {
    KFace kf = face_payload(f);
    const Origins& o = origins_of(t.faces, f);
    if (o.empty()) incomplete("an output face has no recorded origin");
    push_origins(kf.origins, o);
    kf.kept = t.kept.Contains(f) && o.size() == 1;
    k.faces.push_back(std::move(kf));
  }
  TopTools_IndexedDataMapOfShapeListOfShape edge_faces;
  TopExp::MapShapesAndUniqueAncestors(piece, TopAbs_EDGE, TopAbs_FACE, edge_faces);
  TopTools_IndexedMapOfShape edge_index;
  for (const TopoDS_Edge& e : body->edges) {
    edge_index.Add(e);
    const TopTools_ListOfShape& fs = edge_faces.FindFromKey(e);
    KEdge ke;
    if (fs.Extent() == 2) {
      ke.f0 = static_cast<std::uint32_t>(face_index.FindIndex(fs.First()) - 1);
      ke.f1 = static_cast<std::uint32_t>(face_index.FindIndex(fs.Last()) - 1);
      ke.laminar = false;
    } else if (fs.Extent() == 1) {
      if (!sheet) fail("a solid body has a free edge (the result is not closed)");
      ke.f0 = ke.f1 = static_cast<std::uint32_t>(face_index.FindIndex(fs.First()) - 1);
      ke.laminar = true;
      const Origins& o = origins_of(t.edges, e);
      if (o.empty()) incomplete("a boundary edge of a sheet has no recorded origin");
      push_origins(ke.origins, o);
    } else {
      fail("the result is non-manifold: an edge has more than two faces");
    }
    TopoDS_Vertex v0, v1;
    TopExp::Vertices(e, v0, v1, Standard_True);
    gp_Pnt p0 = v0.IsNull() ? gp_Pnt() : BRep_Tool::Pnt(v0);
    gp_Pnt p1 = v1.IsNull() ? p0 : BRep_Tool::Pnt(v1);
    // The contract's canonical order: start precedes end lexicographically.
    {
      const double d[3] = {p1.X() - p0.X(), p1.Y() - p0.Y(), p1.Z() - p0.Z()};
      for (double c : d) {
        if (std::abs(c) <= 1e-9) continue;
        if (c < 0) std::swap(p0, p1);
        break;
      }
    }
    push3(ke.geom, p0.XYZ());
    push3(ke.geom, p1.XYZ());
    BRepAdaptor_Curve c(e);
    ke.geom.push_back(GCPnts_AbscissaPoint::Length(c));
    k.edges.push_back(std::move(ke));
  }
  TopTools_IndexedDataMapOfShapeListOfShape vertex_edges;
  TopExp::MapShapesAndUniqueAncestors(piece, TopAbs_VERTEX, TopAbs_EDGE, vertex_edges);
  for (const TopoDS_Vertex& v : body->vertices) {
    KVertex kv;
    std::vector<std::uint32_t> es;
    for (const TopoDS_Shape& e : vertex_edges.FindFromKey(v)) {
      const int i = edge_index.FindIndex(e);
      if (i > 0) es.push_back(static_cast<std::uint32_t>(i - 1));
    }
    std::sort(es.begin(), es.end());
    for (auto e : es) kv.edges.push_back(e);
    push3(kv.point, BRep_Tool::Pnt(v).XYZ());
    k.vertices.push_back(std::move(kv));
  }
  k.id = registry_put(body);
  return k;
}

// Output bodies of a shape, in a deterministic order: solids, then the faces outside solids
// as sheets, one per set of faces connected through shared edges (in first-face order). Faces
// are grouped as they are, never sewn: sewing would rebuild them and lose their origins.
std::vector<std::pair<TopoDS_Shape, bool>> pieces_of(const TopoDS_Shape& s) {
  std::vector<std::pair<TopoDS_Shape, bool>> out;
  TopTools_IndexedMapOfShape in_solid;
  for (TopExp_Explorer ex(s, TopAbs_SOLID); ex.More(); ex.Next()) {
    out.emplace_back(ex.Current(), false);
    TopExp::MapShapes(ex.Current(), TopAbs_FACE, in_solid);
  }
  std::vector<TopoDS_Shape> loose;
  TopTools_IndexedMapOfShape seen;
  for (TopExp_Explorer ex(s, TopAbs_FACE); ex.More(); ex.Next()) {
    if (in_solid.Contains(ex.Current()) || seen.Contains(ex.Current())) continue;
    seen.Add(ex.Current());
    loose.push_back(ex.Current());
  }
  if (loose.empty()) return out;
  // Union-find over faces sharing an edge.
  std::vector<int> parent(loose.size());
  for (std::size_t i = 0; i < loose.size(); ++i) parent[i] = static_cast<int>(i);
  std::function<int(int)> find = [&](int i) { return parent[i] == i ? i : parent[i] = find(parent[i]); };
  TopTools_IndexedDataMapOfShapeListOfShape edge_faces;
  TopTools_IndexedMapOfShape face_index;
  for (const TopoDS_Shape& f : loose) {
    face_index.Add(f);
    TopExp::MapShapesAndUniqueAncestors(f, TopAbs_EDGE, TopAbs_FACE, edge_faces);
  }
  for (int i = 1; i <= edge_faces.Extent(); ++i) {
    int first = -1;
    for (const TopoDS_Shape& f : edge_faces(i)) {
      const int k = face_index.FindIndex(f) - 1;
      if (k < 0) continue;
      if (first < 0) {
        first = k;
      } else {
        parent[find(k)] = find(first);
      }
    }
  }
  BRep_Builder b;
  std::map<int, std::size_t> sheet_of;
  std::vector<TopoDS_Shell> sheets;
  for (std::size_t i = 0; i < loose.size(); ++i) {
    const int root = find(static_cast<int>(i));
    auto it = sheet_of.find(root);
    if (it == sheet_of.end()) {
      it = sheet_of.emplace(root, sheets.size()).first;
      sheets.emplace_back();
      b.MakeShell(sheets.back());
    }
    b.Add(sheets[it->second], loose[i]);
  }
  for (const TopoDS_Shell& sh : sheets) out.emplace_back(sh, true);
  return out;
}
}  // namespace

// ------------------------------------------------------------------------- interpreter helpers

namespace {
struct Exec {
  rust::Slice<const KInstr> program;
  std::vector<std::shared_ptr<const Body>> operands;
  std::vector<Tracked> slots;
  std::vector<Profile> profiles;  // in step with slots: profile slots hold a profile
  std::vector<bool> is_profile;
  KResult result;

  const Tracked& slot(std::int64_t i) const {
    if (i < 0 || static_cast<std::size_t>(i) >= slots.size()) fail("bad program slot");
    return slots[static_cast<std::size_t>(i)];
  }
  const Profile& profile(std::int64_t i) const {
    if (i < 0 || static_cast<std::size_t>(i) >= slots.size() ||
        !is_profile[static_cast<std::size_t>(i)])
      fail("bad profile slot");
    return profiles[static_cast<std::size_t>(i)];
  }
  const Body& operand(std::int64_t i) const {
    if (i < 0 || static_cast<std::size_t>(i) >= operands.size()) fail("operand out of range");
    return *operands[static_cast<std::size_t>(i)];
  }
  TopoDS_Face face(std::int64_t op, std::int64_t f) const {
    const Body& b = operand(op);
    if (f < 0 || static_cast<std::size_t>(f) >= b.faces.size()) fail("face out of range");
    return b.faces[static_cast<std::size_t>(f)];
  }
  TopoDS_Edge edge(std::int64_t op, std::int64_t e) const {
    const Body& b = operand(op);
    if (e < 0 || static_cast<std::size_t>(e) >= b.edges.size()) fail("edge out of range");
    return b.edges[static_cast<std::size_t>(e)];
  }
  TopoDS_Vertex vertex(std::int64_t op, std::int64_t v) const {
    const Body& b = operand(op);
    if (v < 0 || static_cast<std::size_t>(v) >= b.vertices.size()) fail("vertex out of range");
    return b.vertices[static_cast<std::size_t>(v)];
  }
};

std::uint32_t lab(std::int64_t v) {
  return v < 0 ? kNoLabel : static_cast<std::uint32_t>(v);
}

void require_labels(const Tracked& t, const char* what) {
  TopTools_IndexedMapOfShape faces;
  TopExp::MapShapes(t.shape, TopAbs_FACE, faces);
  for (int i = 1; i <= faces.Extent(); ++i) {
    if (!origins_of(t.faces, faces(i)).empty()) continue;
    GProp_GProps props;
    BRepGProp::SurfaceProperties(faces(i), props);
    const gp_Pnt c = props.CentreOfMass();
    BRepAdaptor_Surface s(TopoDS::Face(faces(i)), Standard_False);
    std::ostringstream m;
    m << what << " made a face OCCT reports no history for (surface type " << int(s.GetType())
      << ", centroid " << c.X() << ", " << c.Y() << ", " << c.Z() << ")";
    incomplete(m.str());
  }
}

void check_valid(const TopoDS_Shape& s, const char* what) {
  if (s.IsNull()) fail(std::string(what) + " produced nothing");
  BRepCheck_Analyzer check(s, Standard_True);
  if (check.IsValid()) return;
  std::ostringstream m;
  m << what << " produced an invalid shape";
  // Name the first invalid entity and its BRepCheck status.
  bool named = false;
  for (TopAbs_ShapeEnum k : {TopAbs_VERTEX, TopAbs_EDGE, TopAbs_WIRE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID}) {
    for (TopExp_Explorer ex(s, k); ex.More() && !named; ex.Next()) {
      if (check.IsValid(ex.Current())) continue;
      const Handle(BRepCheck_Result)& r = check.Result(ex.Current());
      m << " (" << TopAbs::ShapeTypeToString(k);
      if (!r.IsNull()) {
        for (const BRepCheck_Status st : r->Status()) {
          if (st != BRepCheck_NoError) {
            m << ": BRepCheck status " << int(st);
            break;
          }
        }
      }
      m << ")";
      named = true;
    }
    if (named) break;
  }
  if (const char* dir = std::getenv("RUNMAT_K1_DUMP")) {
    static std::atomic<int> n{0};
    const std::string path = std::string(dir) + "/invalid-" + std::to_string(n++) + ".brep";
    BRepTools::Write(s, path.c_str());
    m << " [written to " << path << "]";
  }
  fail(m.str());
}

// ---- blends

// Whether the two faces of an edge meet tangentially along it (sampled at a few points): such an
// edge has no corner to round or bevel.
bool smooth_edge(const TopoDS_Edge& e, const TopoDS_Shape& shape) {
  TopTools_IndexedDataMapOfShapeListOfShape edge_faces;
  TopExp::MapShapesAndUniqueAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_faces);
  const TopTools_ListOfShape* fs = edge_faces.Seek(e);
  if (fs == nullptr || fs->Extent() != 2) return false;
  const TopoDS_Face f1 = TopoDS::Face(fs->First());
  const TopoDS_Face f2 = TopoDS::Face(fs->Last());
  double a = 0, b = 0;
  Handle(Geom2d_Curve) c1 = BRep_Tool::CurveOnSurface(e, f1, a, b);
  double a2 = 0, b2 = 0;
  Handle(Geom2d_Curve) c2 = BRep_Tool::CurveOnSurface(e, f2, a2, b2);
  if (c1.IsNull() || c2.IsNull()) return false;
  BRepGProp_Face g1(f1), g2(f2);
  for (double t : {0.25, 0.5, 0.75}) {
    const gp_Pnt2d p1 = c1->Value(a + t * (b - a));
    const gp_Pnt2d p2 = c2->Value(a2 + t * (b2 - a2));
    gp_Pnt q;
    gp_Vec n1, n2;
    g1.Normal(p1.X(), p1.Y(), q, n1);
    g2.Normal(p2.X(), p2.Y(), q, n2);
    if (n1.Magnitude() < 1e-12 || n2.Magnitude() < 1e-12) return false;
    if (n1.Normalized().Dot(n2.Normalized()) < 1.0 - 1e-9) return false;
  }
  return true;
}

void require_corner_edge(const TopoDS_Edge& e, const TopoDS_Shape& shape, const char* what) {
  if (smooth_edge(e, shape))
    unsupported(std::string(what) + " blends of an edge where its two faces meet tangentially");
}

// ---- fillet

// OCCT's blend approximations use absolute tolerances tuned for millimetre models (a variable
// radius fillet on a metre-scale box fails where the same box in millimetres blends): such an
// operation works on a copy scaled up by 1000 and scales the result back, carrying origins and
// kept faces through both transforms.
struct MmFrame {
  bool on = false;
  std::unique_ptr<BRepBuilderAPI_Transform> up;
  Tracked work;

  MmFrame(const Tracked& src, bool scale) : on(scale) {
    if (!on) {
      work = src;
      return;
    }
    gp_Trsf t;
    t.SetScale(gp::Origin(), 1000.0);
    up = std::make_unique<BRepBuilderAPI_Transform>(src.shape, t, Standard_True);
    work.shape = up->Shape();
    work.sheet = src.sheet;
    propagate(src, *up, work);
    for (int i = 1; i <= src.kept.Extent(); ++i) work.kept.Add(up->ModifiedShape(src.kept.FindKey(i)), 1);
  }
  TopoDS_Shape image(const TopoDS_Shape& s) const { return on ? up->ModifiedShape(s) : s; }
  double len(double v) const { return on ? v * 1000.0 : v; }
  Tracked back(const Tracked& res) const {
    if (!on) return res;
    gp_Trsf t;
    t.SetScale(gp::Origin(), 1e-3);
    BRepBuilderAPI_Transform down(res.shape, t, Standard_True);
    Tracked out;
    out.shape = down.Shape();
    out.sheet = res.sheet;
    propagate(res, down, out);
    for (int i = 1; i <= res.kept.Extent(); ++i) out.kept.Add(down.ModifiedShape(res.kept.FindKey(i)), 1);
    return out;
  }
};

Tracked do_fillet(const Exec& x, const KInstr& in) {
  const Tracked& src = x.slot(in.ints[0]);
  const std::int64_t operand = in.ints[1];
  const std::int64_t edge_base = in.ints[2];
  const std::int64_t vertex_base = in.ints[3];
  const std::int64_t n = in.ints[4];
  const double radius = in.reals[0];
  const std::size_t nvar = static_cast<std::size_t>(in.reals[1]);
  // Rational sections are exact circles; a radius law needs the polynomial approximation.
  const MmFrame mm(src, nvar > 0);
  BRepFilletAPI_MakeFillet mk(mm.work.shape, nvar > 0 ? ChFi3d_Polynomial : ChFi3d_Rational);
  for (std::int64_t i = 0; i < n; ++i) {
    require_corner_edge(x.edge(operand, in.ints[5 + i]), src.shape, "fillet");
    const TopoDS_Edge e = TopoDS::Edge(mm.image(x.edge(operand, in.ints[5 + i])));
    mk.Add(mm.len(radius), e);
    if (nvar > 0) {
      // Positions are fractions of the edge from its canonical start (the lexicographically
      // first end point, as in the summary); OCCT takes relative parameters on the spine, which
      // may run the other way.
      const int ic = mk.Contour(e);
      BRepAdaptor_Curve curve(e);
      gp_Pnt q0 = curve.Value(curve.FirstParameter()), q1 = curve.Value(curve.LastParameter());
      for (double c : {q1.X() - q0.X(), q1.Y() - q0.Y(), q1.Z() - q0.Z()}) {
        if (std::abs(c) <= 1e-9) continue;
        if (c < 0) std::swap(q0, q1);
        break;
      }
      const gp_Pnt first = BRep_Tool::Pnt(mk.FirstVertex(ic));
      const bool flip = first.Distance(q1) < first.Distance(q0);
      std::vector<gp_Pnt2d> pts;
      bool at0 = false, at1 = false;
      for (std::size_t k = 0; k < nvar; ++k) {
        const double t = in.reals[2 + 2 * k];
        at0 |= std::abs(t) < 1e-12;
        at1 |= std::abs(t - 1.0) < 1e-12;
        pts.emplace_back(flip ? 1.0 - t : t, mm.len(in.reals[3 + 2 * k]));
      }
      // The constant radius holds at an end the definition leaves open.
      if (!at0) pts.emplace_back(flip ? 1.0 : 0.0, mm.len(radius));
      if (!at1) pts.emplace_back(flip ? 0.0 : 1.0, mm.len(radius));
      std::sort(pts.begin(), pts.end(), [](const gp_Pnt2d& p, const gp_Pnt2d& q) { return p.X() < q.X(); });
      TColgp_Array1OfPnt2d ur(1, static_cast<int>(pts.size()));
      for (std::size_t k = 0; k < pts.size(); ++k) ur.SetValue(static_cast<int>(k + 1), pts[k]);
      int iinc = 1;
      for (int k = 1; k <= mk.NbEdges(ic); ++k)
        if (mk.Edge(ic, k).IsSame(e)) iinc = k;
      mk.SetRadius(ur, ic, iinc);
    }
  }
  mk.Build();
  if (!mk.IsDone()) {
    std::ostringstream m;
    m << "the fillet could not be built (radius too large, or the edges do not allow it; "
      << mk.NbFaultyContours() << " of " << mk.NbContours() << " contours failed)";
    fail(m.str());
  }
  Tracked res;
  res.shape = mk.Shape();
  propagate(mm.work, mk, res);
  const Body& b = x.operand(operand);
  // Per-edge label overrides (a full round tags both blends by the centre face).
  std::map<std::int64_t, std::uint32_t> overrides;
  const std::size_t ov = static_cast<std::size_t>(5 + n);
  if (in.ints.size() > ov) {
    const std::int64_t no = in.ints[ov];
    for (std::int64_t k = 0; k < no; ++k)
      overrides[in.ints[ov + 1 + 2 * k]] = lab(in.ints[ov + 2 + 2 * k]);
  }
  // Blend faces take the edge they replace; corner patches the vertex they round.
  TopTools_IndexedMapOfShape faces;
  TopExp::MapShapes(res.shape, TopAbs_FACE, faces);
  for (std::size_t i = 0; i < b.edges.size(); ++i) {
    auto it = overrides.find(static_cast<std::int64_t>(i));
    const std::uint32_t l = it != overrides.end()
                                ? it->second
                                : static_cast<std::uint32_t>(edge_base + static_cast<std::int64_t>(i));
    for (const TopoDS_Shape& g : mk.Generated(mm.image(b.edges[i]))) {
      if (g.ShapeType() == TopAbs_FACE && faces.Contains(g)) add_origins(res.faces, g, label(l));
    }
  }
  for (std::size_t i = 0; i < b.vertices.size(); ++i) {
    for (const TopoDS_Shape& g : mk.Generated(mm.image(b.vertices[i]))) {
      if (g.ShapeType() == TopAbs_FACE && faces.Contains(g))
        add_origins(res.faces, g, label(static_cast<std::uint32_t>(vertex_base + static_cast<std::int64_t>(i))));
    }
  }
  Tracked out = mm.back(res);
  require_labels(out, "the fillet");
  return out;
}

// ---- full round

// The half cylinder that replaces a rib's centre face: the slab above the axis between the
// side planes is cut away and the half cylinder fused in; its curved face is the round.
Tracked do_full_round(const Tracked& src, std::uint32_t round_label, const double* r) {
  const gp_Pnt a0(r[0], r[1], r[2]);
  const gp_Dir d(r[3], r[4], r[5]), ydir(r[6], r[7], r[8]), nc(r[9], r[10], r[11]);
  const double len = r[12], rad = r[13];
  if (len <= 0 || rad <= 0) fail("the full round has no extent");
  const gp_Pnt corner(a0.XYZ() - ydir.XYZ() * rad);
  BRepPrimAPI_MakeBox box(gp_Ax2(corner, nc, d), len, 2 * rad, 3 * rad);
  Tracked slab;
  slab.shape = box.Solid();
  Tracked cut = boolean_op(1, {&src}, {&slab});
  BRepPrimAPI_MakeCylinder cyl(gp_Ax2(a0, d, ydir), rad, len, M_PI);
  Tracked half;
  half.shape = cyl.Solid();
  for (TopExp_Explorer ex(half.shape, TopAbs_FACE); ex.More(); ex.Next()) {
    BRepAdaptor_Surface s(TopoDS::Face(ex.Current()));
    if (s.GetType() == GeomAbs_Cylinder) add_origins(half.faces, ex.Current(), label(round_label));
  }
  Tracked out = boolean_op(0, {&cut}, {&half});
  // Same-domain pieces of the rib's ends and the half cylinder's flat ends become one face
  // each when the bodies are output; nothing else of the tools may remain.
  return out;
}

// ---- chamfer

Tracked do_chamfer(const Exec& x, const KInstr& in) {
  const Tracked& src = x.slot(in.ints[0]);
  const std::int64_t operand = in.ints[1];
  const std::int64_t edge_base = in.ints[2];
  const std::int64_t n = in.ints[3];
  const Body& b = x.operand(operand);
  TopTools_IndexedDataMapOfShapeListOfShape edge_faces;
  TopExp::MapShapesAndUniqueAncestors(b.shape, TopAbs_EDGE, TopAbs_FACE, edge_faces);
  BRepFilletAPI_MakeChamfer mk(src.shape);
  for (std::int64_t i = 0; i < n; ++i) {
    const std::int64_t ei = in.ints[4 + 3 * i];
    const std::int64_t mode = in.ints[5 + 3 * i];
    const bool second = in.ints[6 + 3 * i] != 0;
    TopoDS_Edge e = x.edge(operand, ei);
    require_corner_edge(e, src.shape, "chamfer");
    const double a = in.reals[2 * i];
    const double c = in.reals[2 * i + 1];
    const TopTools_ListOfShape& fs = edge_faces.FindFromKey(e);
    if (fs.Extent() != 2) fail("a chamfered edge must have two faces");
    // The summary's first face of an edge is its first ancestor (same map order).
    TopoDS_Face first = TopoDS::Face(second ? fs.Last() : fs.First());
    switch (mode) {
      case 0: mk.Add(a, e); break;
      case 1: mk.Add(a, c, e, TopoDS::Face(fs.First())); break;
      case 2: mk.AddDA(a, c, e, first); break;
      default: fail("unknown chamfer form");
    }
  }
  mk.Build();
  if (!mk.IsDone()) fail("the chamfer could not be built");
  Tracked out;
  out.shape = mk.Shape();
  propagate(src, mk, out);
  TopTools_IndexedMapOfShape faces;
  TopExp::MapShapes(out.shape, TopAbs_FACE, faces);
  for (std::size_t i = 0; i < b.edges.size(); ++i) {
    for (const TopoDS_Shape& g : mk.Generated(b.edges[i])) {
      if (g.ShapeType() == TopAbs_FACE && faces.Contains(g))
        add_origins(out.faces, g, label(static_cast<std::uint32_t>(edge_base + static_cast<std::int64_t>(i))));
    }
  }
  require_labels(out, "the chamfer");
  return out;
}

// ---- cut by a half-space (vertex chamfer, clip planes)

Tracked cut_half_space(const Tracked& src, const gp_Pln& plane, bool keep_negative,
                       std::uint32_t face_label) {
  // The removed side as a large box-bounded half-space: material is removed where
  // (p - o).n > 0 when keep_negative.
  gp_Dir n = plane.Axis().Direction();
  if (!keep_negative) n.Reverse();
  const gp_Pnt o = plane.Location();
  BRepBuilderAPI_MakeFace mf(gp_Pln(o, n));
  const gp_Pnt ref(o.XYZ() + n.XYZ());
  TopoDS_Solid hs = BRepPrimAPI_MakeHalfSpace(mf.Face(), ref).Solid();
  Tracked tool;
  tool.shape = hs;
  TopExp_Explorer ex(hs, TopAbs_FACE);
  for (; ex.More(); ex.Next()) add_origins(tool.faces, ex.Current(), label(face_label));
  return boolean_op(1, {&src}, {&tool});
}

Tracked do_vertex_chamfer(const Exec& x, const KInstr& in) {
  Tracked cur = x.slot(in.ints[0]);
  const std::int64_t operand = in.ints[1];
  const std::int64_t vertex_base = in.ints[2];
  const std::int64_t n = in.ints[3];
  const Body& b = x.operand(operand);
  TopTools_IndexedDataMapOfShapeListOfShape vertex_edges;
  TopExp::MapShapesAndUniqueAncestors(b.shape, TopAbs_VERTEX, TopAbs_EDGE, vertex_edges);
  TopTools_IndexedMapOfShape summary_edges;
  for (const auto& e : b.edges) summary_edges.Add(e);
  for (std::int64_t i = 0; i < n; ++i) {
    const std::int64_t vi = in.ints[4 + i];
    TopoDS_Vertex v = x.vertex(operand, vi);
    std::vector<std::pair<int, TopoDS_Edge>> es;
    for (const TopoDS_Shape& e : vertex_edges.FindFromKey(v)) {
      const int k = summary_edges.FindIndex(e);
      if (k > 0) es.emplace_back(k, TopoDS::Edge(e));
    }
    std::sort(es.begin(), es.end(), [](const auto& l, const auto& r) { return l.first < r.first; });
    if (es.size() != 3) unsupported("a vertex chamfer needs a vertex with three edges");
    gp_Pnt pts[3];
    const gp_Pnt pv = BRep_Tool::Pnt(v);
    for (int k = 0; k < 3; ++k) {
      BRepAdaptor_Curve c(es[k].second);
      TopoDS_Vertex a, z;
      TopExp::Vertices(es[k].second, a, z);
      const bool from_start = a.IsSame(v);
      const double u0 = from_start ? c.FirstParameter() : c.LastParameter();
      const double d = in.reals[k] * (from_start ? 1.0 : -1.0);
      GCPnts_AbscissaPoint ap(c, d, u0);
      if (!ap.IsDone()) fail("a vertex chamfer distance is longer than its edge");
      pts[k] = c.Value(ap.Parameter());
    }
    gp_Vec n1(pts[0], pts[1]), n2(pts[0], pts[2]);
    gp_Vec nn = n1.Crossed(n2);
    if (nn.Magnitude() < 1e-15) fail("vertex chamfer points are collinear");
    gp_Dir nd(nn);
    // Remove the side holding the vertex.
    if (gp_Vec(pts[0], pv).Dot(gp_Vec(nd)) < 0) nd.Reverse();
    cur = cut_half_space(cur, gp_Pln(pts[0], nd), true,
                         static_cast<std::uint32_t>(vertex_base + vi));
  }
  return cur;
}

// ---- shell

// A closed shell: the operand's own faces stay as they are and an offset skin, reversed, bounds
// the sealed void (inward) or encloses the body (outward). Building the solid from the two
// shells keeps every original face, so the body continues its operand.
Tracked closed_shell(const Exec& x, const KInstr& in, const Tracked& src, const Body& b,
                     std::int64_t wall_base, bool outward, double t, std::int64_t nper) {
  BRepOffset_MakeOffset mo;
  mo.Initialize(src.shape, outward ? t : -t, 1e-6, BRepOffset_Skin, Standard_False,
                Standard_False, GeomAbs_Intersection, Standard_False, Standard_False);
  for (std::int64_t i = 0; i < nper; ++i) {
    TopoDS_Face f = x.face(in.ints[1], in.ints[7 + i]);
    const double ti = in.reals[1 + i];
    mo.SetOffsetOnFace(f, outward ? ti : -ti);
  }
  mo.MakeOffsetShape();
  if (!mo.IsDone() || mo.Shape().IsNull()) fail("the shell could not be built (a wall is thicker than the part allows)");
  auto one_shell = [](const TopoDS_Shape& s) {
    TopExp_Explorer ex(s, TopAbs_SHELL);
    if (!ex.More()) fail("the shell's offset skin is not closed");
    TopoDS_Shell sh = TopoDS::Shell(ex.Current());
    ex.Next();
    if (ex.More()) fail("the shell's offset skin splits into several pieces");
    return sh;
  };
  const TopoDS_Shell original = one_shell(src.shape);
  const TopoDS_Shell skin = one_shell(mo.Shape());
  // Orient each shell outward on its own, then reverse the inner one.
  auto oriented = [](const TopoDS_Shell& sh) {
    BRep_Builder bb;
    TopoDS_Solid so;
    bb.MakeSolid(so);
    bb.Add(so, sh);
    BRepLib::OrientClosedSolid(so);
    TopExp_Explorer ex(so, TopAbs_SHELL);
    return TopoDS::Shell(ex.Current());
  };
  TopoDS_Shell outer = oriented(outward ? skin : original);
  TopoDS_Shell inner = oriented(outward ? original : skin);
  inner.Reverse();
  BRep_Builder bb;
  TopoDS_Solid solid;
  bb.MakeSolid(solid);
  bb.Add(solid, outer);
  bb.Add(solid, inner);
  check_valid(solid, "the shell");
  Tracked out;
  out.shape = solid;
  for (int i = 1; i <= src.faces.Extent(); ++i) add_origins(out.faces, src.faces.FindKey(i), src.faces(i));
  for (int i = 1; i <= src.edges.Extent(); ++i) add_origins(out.edges, src.edges.FindKey(i), src.edges(i));
  for (int i = 1; i <= src.vertices.Extent(); ++i)
    add_origins(out.vertices, src.vertices.FindKey(i), src.vertices(i));
  for (int i = 1; i <= src.kept.Extent(); ++i) out.kept.Add(src.kept.FindKey(i), 1);
  TopTools_IndexedMapOfShape faces;
  TopExp::MapShapes(skin, TopAbs_FACE, faces);
  const BRepAlgo_Image& image = mo.OffsetFacesFromShapes();
  for (std::size_t i = 0; i < b.faces.size(); ++i) {
    const std::uint32_t wall = static_cast<std::uint32_t>(wall_base + static_cast<std::int64_t>(i));
    if (!image.HasImage(b.faces[i])) continue;
    TopTools_ListOfShape imgs;
    image.LastImage(b.faces[i], imgs);
    for (const TopoDS_Shape& g : imgs)
      for (TopExp_Explorer ex(g, TopAbs_FACE); ex.More(); ex.Next())
        if (faces.Contains(ex.Current())) add_origins(out.faces, ex.Current(), label(wall));
  }
  require_labels(out, "the shell");
  return out;
}

Tracked do_shell(const Exec& x, const KInstr& in) {
  const Tracked& src = x.slot(in.ints[0]);
  const std::int64_t operand = in.ints[1];
  const std::int64_t inner_base = in.ints[2];
  const std::int64_t rim_base = in.ints[3];
  const bool outward = in.ints[4] != 0;
  const std::int64_t nopen = in.ints[5];
  const std::int64_t nper = in.ints[6 + nopen];
  const double t = in.reals[0];
  const Body& b = x.operand(operand);
  if (nopen == 0) return closed_shell(x, in, src, b, inner_base, outward, t, nper);
  BRepOffset_MakeOffset mk;
  mk.Initialize(src.shape, outward ? t : -t, 1e-6, BRepOffset_Skin, Standard_True, Standard_False,
                GeomAbs_Intersection, Standard_False, Standard_False);
  TopTools_IndexedMapOfShape open;
  for (std::int64_t i = 0; i < nopen; ++i) {
    TopoDS_Face f = x.face(operand, in.ints[6 + i]);
    mk.AddFace(f);
    open.Add(f);
  }
  for (std::int64_t i = 0; i < nper; ++i) {
    TopoDS_Face f = x.face(operand, in.ints[7 + nopen + i]);
    const double ti = in.reals[1 + i];
    mk.SetOffsetOnFace(f, outward ? ti : -ti);
  }
  mk.MakeThickSolid();
  if (!mk.IsDone() || mk.Shape().IsNull()) fail("the shell could not be built (a wall is thicker than the part allows)");
  Tracked out;
  out.shape = mk.Shape();
  TopTools_IndexedMapOfShape faces;
  TopExp::MapShapes(out.shape, TopAbs_FACE, faces);
  // Original faces kept as walls continue their inputs; offset images are the new walls.
  const BRepAlgo_Image& image = mk.OffsetFacesFromShapes();
  for (std::size_t i = 0; i < b.faces.size(); ++i) {
    const TopoDS_Face& f = b.faces[i];
    const std::uint32_t wall = static_cast<std::uint32_t>(inner_base + static_cast<std::int64_t>(i));
    if (open.Contains(f)) {
      // An opened face is consumed: what OCCT keeps of it in its plane (its image, or the face
      // itself trimmed to the wall ring) is the rim.
      const std::uint32_t rim = static_cast<std::uint32_t>(rim_base + static_cast<std::int64_t>(i));
      if (faces.Contains(f)) add_origins(out.faces, f, label(rim));
      for (const TopoDS_Shape& m : mk.Modified(f))
        for (TopExp_Explorer ex(m, TopAbs_FACE); ex.More(); ex.Next())
          if (faces.Contains(ex.Current())) add_origins(out.faces, ex.Current(), label(rim));
      continue;
    }
    if (faces.Contains(f)) {
      add_origins(out.faces, f, origins_of(src.faces, f));
      out.kept.Add(f, 1);
    }
    for (const TopoDS_Shape& m : mk.Modified(f)) {
      if (faces.Contains(m)) add_origins(out.faces, m, origins_of(src.faces, f));
    }
    if (image.HasImage(f)) {
      TopTools_ListOfShape imgs;
      image.LastImage(f, imgs);
      for (const TopoDS_Shape& g : imgs) {
        for (TopExp_Explorer ex(g, TopAbs_FACE); ex.More(); ex.Next())
          if (faces.Contains(ex.Current()) && origins_of(out.faces, ex.Current()).empty())
            add_origins(out.faces, ex.Current(), label(wall));
      }
    }
  }
  // Rims: faces generated along the boundary of an opened face.
  for (int oi = 1; oi <= open.Extent(); ++oi) {
    const TopoDS_Face& f = TopoDS::Face(open(oi));
    std::uint32_t fi = 0;
    for (std::size_t i = 0; i < b.faces.size(); ++i)
      if (b.faces[i].IsSame(f)) fi = static_cast<std::uint32_t>(i);
    for (TopExp_Explorer ex(f, TopAbs_EDGE); ex.More(); ex.Next()) {
      for (const TopoDS_Shape& g : mk.Generated(ex.Current())) {
        for (TopExp_Explorer fx(g, TopAbs_FACE); fx.More(); fx.Next())
          if (faces.Contains(fx.Current()) && origins_of(out.faces, fx.Current()).empty())
            add_origins(out.faces, fx.Current(), label(static_cast<std::uint32_t>(rim_base + fi)));
      }
    }
  }
  // A rim not reported as generated still holds the opened face's boundary edge.
  for (int i = 1; i <= faces.Extent(); ++i) {
    if (!origins_of(out.faces, faces(i)).empty()) continue;
    for (int oi = 1; oi <= open.Extent(); ++oi) {
      TopTools_IndexedMapOfShape open_edges;
      TopExp::MapShapes(open(oi), TopAbs_EDGE, open_edges);
      bool touches = false;
      for (TopExp_Explorer ex(faces(i), TopAbs_EDGE); ex.More() && !touches; ex.Next())
        touches = open_edges.Contains(ex.Current());
      if (touches) {
        std::uint32_t fi = 0;
        for (std::size_t k = 0; k < b.faces.size(); ++k)
          if (b.faces[k].IsSame(open(oi))) fi = static_cast<std::uint32_t>(k);
        add_origins(out.faces, faces(i), label(static_cast<std::uint32_t>(rim_base + fi)));
        break;
      }
    }
  }
  require_labels(out, "the shell");
  return out;
}

// ---- draft

Tracked do_draft(const Exec& x, const KInstr& in) {
  const Tracked& src = x.slot(in.ints[0]);
  const std::int64_t operand = in.ints[1];
  const std::int64_t nfaces = in.ints[2];
  const std::int64_t nparting = in.ints[3 + nfaces];
  const gp_Ax3 neutral = frame_from(in.reals.data());
  const double angle = in.reals[12];
  const gp_Dir pull = neutral.Direction();
  BRepOffsetAPI_DraftAngle mk(src.shape);
  std::vector<TopoDS_Edge> parting;
  for (std::int64_t i = 0; i < nparting; ++i) parting.push_back(x.edge(operand, in.ints[4 + nfaces + i]));
  for (std::int64_t i = 0; i < nfaces; ++i) {
    TopoDS_Face f = x.face(operand, in.ints[3 + i]);
    gp_Pln plane(neutral);
    if (!parting.empty()) {
      // Hinge about the parting edge this face holds: the neutral plane through that edge,
      // perpendicular to the pull.
      bool found = false;
      for (const TopoDS_Edge& e : parting) {
        for (TopExp_Explorer ex(f, TopAbs_EDGE); ex.More(); ex.Next()) {
          if (ex.Current().IsSame(e)) {
            BRepAdaptor_Curve c(e);
            plane = gp_Pln(c.Value(0.5 * (c.FirstParameter() + c.LastParameter())), pull);
            found = true;
            break;
          }
        }
        if (found) break;
      }
      if (!found) fail("a drafted face touches none of the parting edges");
    }
    mk.Add(f, pull, angle, plane);
    if (!mk.AddDone()) fail("a face cannot be drafted about the neutral plane");
  }
  mk.Build();
  if (!mk.IsDone()) fail("the draft could not be built");
  Tracked out;
  out.shape = mk.Shape();
  propagate_draft(src, mk, out);
  require_labels(out, "the draft");
  return out;
}

// ---- transforms

Tracked do_transform(const Tracked& src, const gp_Trsf& t, std::int64_t instance) {
  BRepBuilderAPI_Transform mk(src.shape, t, Standard_True);
  if (!mk.IsDone()) fail("the transform failed");
  Tracked out;
  out.shape = mk.Shape();
  out.sheet = src.sheet;
  if (instance < 0) {
    propagate(src, mk, out);
    return out;
  }
  // A copy: each face is an image of its source face.
  auto copy = [&](const ShapeOrigins& from, ShapeOrigins& to) {
    for (int i = 1; i <= from.Extent(); ++i) {
      Origins o;
      for (Origin s : from(i)) {
        if (s.kind == kInput) {
          s.kind = kCopy;
          s.c = static_cast<std::uint32_t>(instance);
        }
        o.push_back(s);
      }
      std::sort(o.begin(), o.end());
      for (const TopoDS_Shape& m : mk.Modified(from.FindKey(i))) add_origins(to, m, o);
    }
  };
  copy(src.faces, out.faces);
  copy(src.edges, out.edges);
  copy(src.vertices, out.vertices);
  return out;
}

Tracked do_scale(const Tracked& src, const double* r) {
  const gp_Pnt c(r[0], r[1], r[2]);
  if (std::abs(r[3] - r[4]) < 1e-15 && std::abs(r[3] - r[5]) < 1e-15) {
    gp_Trsf t;
    t.SetScale(c, r[3]);
    return do_transform(src, t, -1);
  }
  gp_GTrsf g;
  g.SetValue(1, 1, r[3]);
  g.SetValue(2, 2, r[4]);
  g.SetValue(3, 3, r[5]);
  g.SetValue(1, 4, c.X() * (1 - r[3]));
  g.SetValue(2, 4, c.Y() * (1 - r[4]));
  g.SetValue(3, 4, c.Z() * (1 - r[5]));
  BRepBuilderAPI_GTransform mk(src.shape, g, Standard_True);
  if (!mk.IsDone()) fail("the scale failed");
  Tracked out;
  out.shape = mk.Shape();
  out.sheet = src.sheet;
  propagate(src, mk, out);
  return out;
}

// ---- split by a plane

namespace {
// Each solid of `in` as an independent copy, so a face two solids share (the splitter's cut
// face, bounding both sides in opposite orientations) becomes one face per solid. Origins and
// kept marks follow the copy's history; shapes outside solids are carried as they are.
Tracked separate_solids(const Tracked& in) {
  BRep_Builder b;
  TopoDS_Compound c;
  b.MakeCompound(c);
  Tracked out;
  out.sheet = in.sheet;
  auto carry_copy = [](const ShapeOrigins& from, ShapeOrigins& to, BRepBuilderAPI_Copy& cp,
                       const TopoDS_Shape& solid, TopAbs_ShapeEnum kind) {
    TopTools_IndexedMapOfShape sub;
    TopExp::MapShapes(solid, kind, sub);
    for (int i = 1; i <= sub.Extent(); ++i) {
      const Origins& o = origins_of(from, sub(i));
      if (o.empty()) continue;
      for (const TopoDS_Shape& m : cp.Modified(sub(i))) add_origins(to, m, o);
    }
  };
  TopTools_IndexedMapOfShape in_solid;
  for (TopExp_Explorer ex(in.shape, TopAbs_SOLID); ex.More(); ex.Next()) {
    const TopoDS_Shape& solid = ex.Current();
    TopExp::MapShapes(solid, TopAbs_FACE, in_solid);
    BRepBuilderAPI_Copy cp(solid, Standard_False, Standard_False);
    b.Add(c, cp.Shape());
    carry_copy(in.faces, out.faces, cp, solid, TopAbs_FACE);
    carry_copy(in.edges, out.edges, cp, solid, TopAbs_EDGE);
    carry_copy(in.vertices, out.vertices, cp, solid, TopAbs_VERTEX);
    TopTools_IndexedMapOfShape faces;
    TopExp::MapShapes(solid, TopAbs_FACE, faces);
    for (int i = 1; i <= faces.Extent(); ++i)
      if (in.kept.Contains(faces(i)))
        for (const TopoDS_Shape& m : cp.Modified(faces(i))) out.kept.Add(m, 1);
  }
  for (TopExp_Explorer ex(in.shape, TopAbs_FACE); ex.More(); ex.Next()) {
    if (in_solid.Contains(ex.Current())) continue;
    b.Add(c, ex.Current());
    add_origins(out.faces, ex.Current(), origins_of(in.faces, ex.Current()));
    if (in.kept.Contains(ex.Current())) out.kept.Add(ex.Current(), 1);
    TopTools_IndexedMapOfShape sub;
    TopExp::MapShapes(ex.Current(), TopAbs_EDGE, sub);
    for (int i = 1; i <= sub.Extent(); ++i) add_origins(out.edges, sub(i), origins_of(in.edges, sub(i)));
    sub.Clear();
    TopExp::MapShapes(ex.Current(), TopAbs_VERTEX, sub);
    for (int i = 1; i <= sub.Extent(); ++i) add_origins(out.vertices, sub(i), origins_of(in.vertices, sub(i)));
  }
  out.shape = c;
  return out;
}
}  // namespace

Tracked do_split(const Tracked& src, const gp_Ax3& frame, std::uint32_t pos_label,
                 std::uint32_t neg_label) {
  Bnd_Box box;
  BRepBndLib::Add(src.shape, box);
  const double size = box.IsVoid() ? 1.0 : std::sqrt(box.SquareExtent()) * 2.0 + 1.0;
  gp_Pln plane(frame);
  BRepBuilderAPI_MakeFace mf(plane, -size, size, -size, size);
  BRepAlgoAPI_Splitter sp;
  TopTools_ListOfShape args, tools;
  args.Append(src.shape);
  tools.Append(mf.Face());
  sp.SetArguments(args);
  sp.SetTools(tools);
  sp.SetRunParallel(Standard_False);
  sp.Build();
  if (!sp.IsDone() || sp.HasErrors()) fail("the split failed");
  Tracked split;
  split.shape = sp.Shape();
  split.sheet = src.sheet;
  propagate(src, sp, split);
  // The cut face is shared by the two sides; give each side its own copy before naming it.
  Tracked out = separate_solids(split);
  // Split faces (images of the cutting plane, so without an origin): tagged by the side of the
  // body they bound, which is opposite their outward normal.
  for (TopExp_Explorer sx(out.shape, TopAbs_SOLID); sx.More(); sx.Next()) {
    for (TopExp_Explorer ex(sx.Current(), TopAbs_FACE); ex.More(); ex.Next()) {
      if (!origins_of(out.faces, ex.Current()).empty()) continue;
      BRepAdaptor_Surface s(TopoDS::Face(ex.Current()));
      if (s.GetType() != GeomAbs_Plane) continue;
      gp_Dir n = s.Plane().Axis().Direction();
      if (!s.Plane().Direct()) n.Reverse();
      if (ex.Current().Orientation() == TopAbs_REVERSED) n.Reverse();
      const bool solid_positive = n.Dot(frame.Direction()) < 0;
      add_origins(out.faces, ex.Current(), label(solid_positive ? pos_label : neg_label));
    }
  }
  require_labels(out, "the split");
  return out;
}

// ---- cut a solid with a sheet

namespace {
// A point inside a face (a UV grid sample the face classifier puts IN) and the face's outward
// normal there, as the face is oriented in its shape.
bool face_sample(const TopoDS_Face& f, gp_Pnt& p, gp_Vec& n) {
  double u0, u1, v0, v1;
  BRepTools::UVBounds(f, u0, u1, v0, v1);
  BRepGProp_Face g(f);
  const double tol = BRep_Tool::Tolerance(f);
  for (int i = 1; i < 8; ++i) {
    for (int j = 1; j < 8; ++j) {
      const gp_Pnt2d uv(u0 + (u1 - u0) * ((i * 37) % 8) / 8.0, v0 + (v1 - v0) * ((j * 53) % 8) / 8.0);
      BRepClass_FaceClassifier c(f, uv, tol);
      if (c.State() != TopAbs_IN) continue;
      g.Normal(uv.X(), uv.Y(), p, n);
      if (n.Magnitude() < 1e-12) continue;
      n.Normalize();
      return true;
    }
  }
  return false;
}

// The sheet's normal at a point on it (as its faces are oriented in the sheet).
bool sheet_normal(const TopoDS_Shape& sheet, const gp_Pnt& p, gp_Vec& n) {
  for (TopExp_Explorer ex(sheet, TopAbs_FACE); ex.More(); ex.Next()) {
    const TopoDS_Face f = TopoDS::Face(ex.Current());
    GeomAPI_ProjectPointOnSurf proj(p, BRep_Tool::Surface(f));
    if (proj.NbPoints() == 0 || proj.LowerDistance() > 1e-6) continue;
    double u, v;
    proj.LowerDistanceParameters(u, v);
    BRepClass_FaceClassifier c(f, gp_Pnt2d(u, v), BRep_Tool::Tolerance(f));
    if (c.State() == TopAbs_OUT) continue;
    gp_Pnt q;
    BRepGProp_Face(f).Normal(u, v, q, n);
    if (n.Magnitude() < 1e-12) continue;
    n.Normalize();
    return true;
  }
  return false;
}
}  // namespace

// Split the solid by the sheet and keep the pieces on the `keep_positive` side of the sheet's
// normal; the faces the sheet cut take `cut_label`.
Tracked do_cut_with_sheet(const Tracked& solid, const Tracked& sheet, bool keep_positive,
                          std::uint32_t cut_label) {
  BRepAlgoAPI_Splitter sp;
  TopTools_ListOfShape args, tools;
  args.Append(solid.shape);
  tools.Append(sheet.shape);
  sp.SetArguments(args);
  sp.SetTools(tools);
  sp.SetRunParallel(Standard_False);
  sp.SetNonDestructive(Standard_True);
  sp.Build();
  if (!sp.IsDone() || sp.HasErrors()) fail("the cut with the surface failed");
  Tracked split;
  split.shape = sp.Shape();
  propagate(solid, sp, split);
  Tracked pieces = separate_solids(split);
  BRep_Builder b;
  TopoDS_Compound kept;
  b.MakeCompound(kept);
  int npieces = 0, nkept = 0;
  std::vector<TopoDS_Face> cut_faces;
  for (TopExp_Explorer sx(pieces.shape, TopAbs_SOLID); sx.More(); sx.Next()) {
    ++npieces;
    int side = 0;  // +1 positive, -1 negative
    std::vector<TopoDS_Face> mine;
    for (TopExp_Explorer ex(sx.Current(), TopAbs_FACE); ex.More(); ex.Next()) {
      if (!origins_of(pieces.faces, ex.Current()).empty()) continue;
      const TopoDS_Face f = TopoDS::Face(ex.Current());
      mine.push_back(f);
      gp_Pnt p;
      gp_Vec out_n, sheet_n;
      if (side == 0 && face_sample(f, p, out_n) && sheet_normal(sheet.shape, p, sheet_n))
        side = out_n.Dot(sheet_n) < 0 ? +1 : -1;  // the piece lies against its cut face's normal
    }
    if (side == 0) fail("a piece of the cut does not meet the surface");
    if ((side > 0) != keep_positive) continue;
    ++nkept;
    b.Add(kept, sx.Current());
    cut_faces.insert(cut_faces.end(), mine.begin(), mine.end());
  }
  if (npieces < 2) fail("the surface does not cut through the body");
  if (nkept == 0) fail("nothing of the body lies on the kept side of the surface");
  Tracked out;
  out.shape = kept;
  TopTools_IndexedMapOfShape present[3];
  const TopAbs_ShapeEnum kinds[3] = {TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX};
  const ShapeOrigins* from[3] = {&pieces.faces, &pieces.edges, &pieces.vertices};
  ShapeOrigins* to[3] = {&out.faces, &out.edges, &out.vertices};
  for (int k = 0; k < 3; ++k) {
    TopExp::MapShapes(kept, kinds[k], present[k]);
    for (int i = 1; i <= from[k]->Extent(); ++i)
      if (present[k].Contains(from[k]->FindKey(i))) add_origins(*to[k], from[k]->FindKey(i), (*from[k])(i));
  }
  for (int i = 1; i <= pieces.kept.Extent(); ++i)
    if (present[0].Contains(pieces.kept.FindKey(i))) out.kept.Add(pieces.kept.FindKey(i), 1);
  for (const TopoDS_Face& f : cut_faces) add_origins(out.faces, f, label(cut_label));
  require_labels(out, "the cut with the surface");
  return out;
}

// ---- ruled strips along sheet edges

// op 34: ints [n, then per edge: operand, edge, face label, boundary labels 0..3], reals
// [distance, angle]. Each strip runs from its edge along the bounded face's normal turned
// `angle` outward across the edge. Boundary labels: 0 the edge itself, 1 the far edge, 2 and 3
// the edges from the edge's start and end vertices (as the face's loop runs).
Tracked do_ruled(const Exec& x, const KInstr& in) {
  const auto& I = in.ints;
  const std::int64_t n = I[0];
  const double dist = in.reals[0], ang = in.reals[1];
  BRepBuilderAPI_Sewing sew(1e-7);
  struct Strip {
    TopoDS_Face face;
    std::uint32_t face_label;
    std::array<std::pair<TopoDS_Shape, std::uint32_t>, 4> edges;
  };
  std::vector<Strip> strips;
  for (std::int64_t k = 0; k < n; ++k) {
    const std::int64_t* e7 = &I[1 + 7 * k];
    const Body& b = x.operand(e7[0]);
    const TopoDS_Edge e = x.edge(e7[0], e7[1]);
    TopTools_IndexedDataMapOfShapeListOfShape edge_faces;
    TopExp::MapShapesAndUniqueAncestors(b.shape, TopAbs_EDGE, TopAbs_FACE, edge_faces);
    const TopTools_ListOfShape* fs = edge_faces.Seek(e);
    if (fs == nullptr || fs->Extent() != 1) fail("a ruled surface runs along boundary edges of a sheet");
    const TopoDS_Face f = TopoDS::Face(fs->First());
    BRepAdaptor_Surface surf(f);
    if (surf.GetType() != GeomAbs_Plane) unsupported("ruled surfaces along edges of curved faces");
    gp_Dir nrm = surf.Plane().Axis().Direction();
    if (!surf.Plane().Direct()) nrm.Reverse();
    if (f.Orientation() == TopAbs_REVERSED) nrm.Reverse();
    TopoDS_Edge eo;
    for (TopExp_Explorer ex(f, TopAbs_EDGE); ex.More(); ex.Next())
      if (ex.Current().IsSame(e)) eo = TopoDS::Edge(ex.Current());
    TopoDS_Vertex va, vc;
    TopExp::Vertices(eo, va, vc, Standard_True);
    const gp_Pnt a = BRep_Tool::Pnt(va), c = BRep_Tool::Pnt(vc);
    gp_Vec w = gp_Vec(nrm) * std::cos(ang);
    if (std::abs(std::sin(ang)) > 1e-15) {
      if (BRepAdaptor_Curve(eo).GetType() != GeomAbs_Line)
        unsupported("ruled surfaces turned off the face normal along curved edges");
      const gp_Vec out = gp_Vec(a, c).Crossed(gp_Vec(nrm)).Normalized();
      w += out * std::sin(ang);
    }
    w *= dist;
    BRepPrimAPI_MakePrism mk(eo, w, Standard_True);
    if (!mk.IsDone()) fail("the ruled strip could not be built");
    Strip st;
    TopExp_Explorer fx(mk.Shape(), TopAbs_FACE);
    if (!fx.More()) fail("the ruled strip has no face");
    st.face = TopoDS::Face(fx.Current());
    st.face_label = lab(e7[2]);
    st.edges[0] = {mk.FirstShape(), lab(e7[3])};
    st.edges[1] = {mk.LastShape(), lab(e7[4])};
    st.edges[2] = {mk.Generated(va).IsEmpty() ? TopoDS_Shape() : mk.Generated(va).First(), lab(e7[5])};
    st.edges[3] = {mk.Generated(vc).IsEmpty() ? TopoDS_Shape() : mk.Generated(vc).First(), lab(e7[6])};
    sew.Add(st.face);
    strips.push_back(std::move(st));
  }
  sew.Perform();
  Tracked out;
  out.shape = sew.SewedShape();
  out.sheet = true;
  auto image = [&](const TopoDS_Shape& s) {
    if (s.IsNull()) return s;
    return sew.IsModifiedSubShape(s) ? sew.ModifiedSubShape(s) : s;
  };
  for (const Strip& st : strips) {
    add_origins(out.faces, image(st.face), label(st.face_label));
    for (const auto& [edge, l] : st.edges) {
      if (edge.IsNull()) continue;
      for (TopExp_Explorer ex(image(edge), TopAbs_EDGE); ex.More(); ex.Next())
        add_origins(out.edges, ex.Current(), label(l));
    }
  }
  require_labels(out, "the ruled surface");
  return out;
}

namespace {
// Two edges with the same end points (a fill's boundary edge as the patch rebuilt it).
bool same_edge_ends(const TopoDS_Edge& e, const TopoDS_Edge& f) {
  TopoDS_Vertex e0, e1, f0, f1;
  TopExp::Vertices(e, e0, e1);
  TopExp::Vertices(f, f0, f1);
  if (e0.IsNull() || e1.IsNull() || f0.IsNull() || f1.IsNull()) return false;
  const gp_Pnt a0 = BRep_Tool::Pnt(e0), a1 = BRep_Tool::Pnt(e1);
  const gp_Pnt b0 = BRep_Tool::Pnt(f0), b1 = BRep_Tool::Pnt(f1);
  return (a0.Distance(b0) < 1e-7 && a1.Distance(b1) < 1e-7) ||
         (a0.Distance(b1) < 1e-7 && a1.Distance(b0) < 1e-7);
}
}  // namespace

// ---- delete holes, fill loops

// op 35: ints [slot, operand, n, edges...]. Inner loops of a face whose edges are all selected
// are dropped; the face keeps its origin. A selected edge on a face's outer loop is a hole
// whose rim runs over several faces, which needs a patch.
Tracked do_delete_hole(const Exec& x, const KInstr& in) {
  const Tracked& src = x.slot(in.ints[0]);
  const std::int64_t operand = in.ints[1];
  const std::int64_t n = in.ints[2];
  TopTools_IndexedMapOfShape picked;
  for (std::int64_t i = 0; i < n; ++i) picked.Add(x.edge(operand, in.ints[3 + i]));
  Handle(BRepTools_ReShape) re = new BRepTools_ReShape();
  Tracked out = src;
  int closed = 0;
  for (TopExp_Explorer fx(src.shape, TopAbs_FACE); fx.More(); fx.Next()) {
    const TopoDS_Face f = TopoDS::Face(fx.Current());
    const TopoDS_Face ff = TopoDS::Face(f.Oriented(TopAbs_FORWARD));
    const TopoDS_Wire outer = BRepTools::OuterWire(ff);
    std::vector<TopoDS_Wire> keep;
    bool dropped = false;
    for (TopExp_Explorer wx(ff, TopAbs_WIRE); wx.More(); wx.Next()) {
      const TopoDS_Wire w = TopoDS::Wire(wx.Current());
      int total = 0, sel = 0;
      for (TopExp_Explorer ex(w, TopAbs_EDGE); ex.More(); ex.Next()) {
        ++total;
        sel += picked.Contains(ex.Current()) ? 1 : 0;
      }
      if (w.IsSame(outer)) {
        if (sel > 0) unsupported("closing a hole whose rim runs over several faces");
        continue;
      }
      if (sel == total && total > 0) {
        dropped = true;
        ++closed;
        continue;
      }
      if (sel > 0) fail("a hole's rim is only partly selected");
      keep.push_back(w);
    }
    if (!dropped) continue;
    BRepBuilderAPI_MakeFace mf(BRep_Tool::Surface(ff), outer, Standard_True);
    for (const TopoDS_Wire& w : keep) mf.Add(w);
    if (!mf.IsDone()) fail("the face could not be rebuilt without its hole");
    TopoDS_Face nf = mf.Face();
    nf.Orientation(f.Orientation());
    re->Replace(f, nf);
    add_origins(out.faces, nf, origins_of(src.faces, f));
  }
  if (closed == 0) fail("the selected edges close no hole of a sheet face");
  out.shape = re->Apply(src.shape);
  require_labels(out, "the hole deletion");
  return out;
}

// op 36: ints [face label, n, then per edge: operand, edge, continuity (0 G0, 1 G1, 2 G2),
// boundary label]. A planar loop is patched by its plane; any other loop, or one that must meet
// its faces tangentially or with curvature, by a filling surface.
Tracked do_fill(const Exec& x, const KInstr& in) {
  const auto& I = in.ints;
  const std::uint32_t face_label = lab(I[0]);
  const std::int64_t n = I[1];
  struct Bound {
    TopoDS_Edge edge;
    TopoDS_Face face;
    int continuity;
    std::uint32_t label;
  };
  std::vector<Bound> bounds;
  bool smooth = false;
  for (std::int64_t k = 0; k < n; ++k) {
    const std::int64_t* q = &I[2 + 4 * k];
    Bound b;
    b.edge = x.edge(q[0], q[1]);
    b.continuity = static_cast<int>(q[2]);
    b.label = lab(q[3]);
    smooth |= b.continuity > 0;
    if (b.continuity > 0) {
      TopTools_IndexedDataMapOfShapeListOfShape edge_faces;
      TopExp::MapShapesAndUniqueAncestors(x.operand(q[0]).shape, TopAbs_EDGE, TopAbs_FACE, edge_faces);
      const TopTools_ListOfShape* fs = edge_faces.Seek(b.edge);
      if (fs == nullptr || fs->IsEmpty()) fail("a smooth fill edge needs a face to meet");
      b.face = TopoDS::Face(fs->First());
    }
    bounds.push_back(b);
  }
  // Chain the edges into one loop.
  BRepBuilderAPI_MakeWire mw;
  std::vector<bool> used(bounds.size(), false);
  TopoDS_Vertex start, tip;
  for (std::size_t added = 0; added < bounds.size(); ++added) {
    bool found = false;
    for (std::size_t i = 0; i < bounds.size() && !found; ++i) {
      if (used[i]) continue;
      TopoDS_Vertex v0, v1;
      TopExp::Vertices(bounds[i].edge, v0, v1);
      if (added > 0) {
        const gp_Pnt t = BRep_Tool::Pnt(tip);
        if (BRep_Tool::Pnt(v0).Distance(t) < 1e-7) {
          tip = v1;
        } else if (BRep_Tool::Pnt(v1).Distance(t) < 1e-7) {
          tip = v0;
        } else {
          continue;
        }
      } else {
        start = v0;
        tip = v1;
      }
      mw.Add(bounds[i].edge);
      if (!mw.IsDone()) fail("the fill edges do not chain into a loop");
      used[i] = true;
      found = true;
    }
    if (!found) fail("the fill edges do not chain into one loop");
  }
  if (BRep_Tool::Pnt(start).Distance(BRep_Tool::Pnt(tip)) > 1e-7) fail("the fill edges do not close a loop");
  TopoDS_Face patch;
  if (!smooth) {
    BRepBuilderAPI_MakeFace mf(mw.Wire(), Standard_True);
    if (mf.IsDone()) patch = mf.Face();
  }
  if (patch.IsNull()) {
    BRepOffsetAPI_MakeFilling fill;
    for (const Bound& b : bounds) {
      if (b.continuity == 0) {
        fill.Add(b.edge, GeomAbs_C0);
      } else {
        fill.Add(b.edge, b.face, b.continuity == 1 ? GeomAbs_G1 : GeomAbs_G2);
      }
    }
    fill.Build();
    if (!fill.IsDone()) fail("the loop could not be filled");
    TopExp_Explorer ex(fill.Shape(), TopAbs_FACE);
    if (!ex.More()) fail("the fill made no face");
    patch = TopoDS::Face(ex.Current());
  }
  Tracked out;
  BRep_Builder bb;
  TopoDS_Shell sh;
  bb.MakeShell(sh);
  bb.Add(sh, patch);
  out.shape = sh;
  out.sheet = true;
  add_origins(out.faces, patch, label(face_label));
  for (TopExp_Explorer ex(patch, TopAbs_EDGE); ex.More(); ex.Next()) {
    const TopoDS_Edge e = TopoDS::Edge(ex.Current());
    for (const Bound& b : bounds)
      if (e.IsSame(b.edge) || same_edge_ends(e, b.edge)) add_origins(out.edges, e, label(b.label));
  }
  require_labels(out, "the fill");
  return out;
}

// op 44: ints [face label, side label 0, side label 1, n1, n2, then per edge (first's, then
// second's): operand, edge, continuity, boundary label]. Built in exact_boundary.cc.
Tracked do_boundary(const Exec& x, const KInstr& in) {
  const auto& I = in.ints;
  const std::int64_t n1 = I[3], n2 = I[4];
  std::vector<BoundaryCurve> first, second;
  for (std::int64_t k = 0; k < n1 + n2; ++k) {
    const std::int64_t* q = &I[5 + 4 * k];
    BoundaryCurve c;
    c.edge = x.edge(q[0], q[1]);
    c.continuity = static_cast<int>(q[2]);
    c.label = lab(q[3]);
    if (c.continuity > 0) {
      TopTools_IndexedDataMapOfShapeListOfShape edge_faces;
      TopExp::MapShapesAndUniqueAncestors(x.operand(q[0]).shape, TopAbs_EDGE, TopAbs_FACE, edge_faces);
      const TopTools_ListOfShape* fs = edge_faces.Seek(c.edge);
      if (fs == nullptr || fs->IsEmpty()) fail("a tangent or curvature boundary edge needs a face to meet");
      c.face = TopoDS::Face(fs->First());
    }
    (k < n1 ? first : second).push_back(c);
  }
  Tracked out = build_boundary(first, second, lab(I[0]), lab(I[1]), lab(I[2]));
  require_labels(out, "the boundary surface");
  return out;
}

// ---- extend sheet edges

// op 37: ints [slot, operand, n, then per edge: edge, extension label], reals [distance]. Each
// straight boundary edge of a planar face grows a strip of the face's plane outward across it;
// the strip shares the edge and holds no origin of its own, so the output's same-domain merge
// makes it part of the face. Its new boundary edges take the edge's extension label.
Tracked do_extend(const Exec& x, const KInstr& in) {
  const Tracked& src = x.slot(in.ints[0]);
  const std::int64_t operand = in.ints[1];
  const std::int64_t n = in.ints[2];
  const double dist = in.reals[0];
  const Body& body = x.operand(operand);
  TopTools_IndexedDataMapOfShapeListOfShape edge_faces;
  TopExp::MapShapesAndUniqueAncestors(body.shape, TopAbs_EDGE, TopAbs_FACE, edge_faces);
  BRep_Builder bb;
  TopoDS_Shell sh;
  bb.MakeShell(sh);
  for (TopExp_Explorer fx(src.shape, TopAbs_FACE); fx.More(); fx.Next()) bb.Add(sh, fx.Current());
  Tracked out = src;
  for (std::int64_t k = 0; k < n; ++k) {
    const TopoDS_Edge e = x.edge(operand, in.ints[3 + 2 * k]);
    const std::uint32_t l = lab(in.ints[4 + 2 * k]);
    const TopTools_ListOfShape* fs = edge_faces.Seek(e);
    if (fs == nullptr || fs->Extent() != 1) fail("extend needs boundary edges of a sheet");
    const TopoDS_Face f = TopoDS::Face(fs->First());
    BRepAdaptor_Surface surf(f);
    if (surf.GetType() != GeomAbs_Plane) unsupported("extending edges of curved faces");
    TopoDS_Edge eo;
    for (TopExp_Explorer ex(f, TopAbs_EDGE); ex.More(); ex.Next())
      if (ex.Current().IsSame(e)) eo = TopoDS::Edge(ex.Current());
    if (BRepAdaptor_Curve(eo).GetType() != GeomAbs_Line) unsupported("extending curved edges");
    gp_Dir nrm = surf.Plane().Axis().Direction();
    if (!surf.Plane().Direct()) nrm.Reverse();
    if (f.Orientation() == TopAbs_REVERSED) nrm.Reverse();
    TopoDS_Vertex va, vc;
    TopExp::Vertices(eo, va, vc, Standard_True);
    const gp_Vec along(BRep_Tool::Pnt(va), BRep_Tool::Pnt(vc));
    const gp_Vec w = along.Crossed(gp_Vec(nrm)).Normalized() * dist;
    BRepPrimAPI_MakePrism mk(eo, w, Standard_False);
    if (!mk.IsDone()) fail("the extension strip could not be built");
    TopExp_Explorer sx(mk.Shape(), TopAbs_FACE);
    if (!sx.More()) fail("the extension strip has no face");
    TopoDS_Face strip = TopoDS::Face(sx.Current());
    // Orient the strip with the face it extends.
    BRepAdaptor_Surface ss(strip);
    gp_Dir sn = ss.Plane().Axis().Direction();
    if (!ss.Plane().Direct()) sn.Reverse();
    if (strip.Orientation() == TopAbs_REVERSED) sn.Reverse();
    if (sn.Dot(nrm) < 0) strip.Reverse();
    bb.Add(sh, strip);
    for (const TopoDS_Shape& g : {mk.LastShape(), mk.Generated(va).IsEmpty() ? TopoDS_Shape() : mk.Generated(va).First(),
                                  mk.Generated(vc).IsEmpty() ? TopoDS_Shape() : mk.Generated(vc).First()})
      if (!g.IsNull())
        for (TopExp_Explorer ex(g, TopAbs_EDGE); ex.More(); ex.Next()) add_origins(out.edges, ex.Current(), label(l));
  }
  out.shape = sh;
  out.sheet = true;
  return out;
}

// ---- named boxes (imports of box fixtures)

Tracked do_box(const KInstr& in) {
  const double* r = in.reals.data();
  BRepPrimAPI_MakeBox mk(gp_Pnt(r[0], r[1], r[2]), gp_Pnt(r[3], r[4], r[5]));
  Tracked t;
  t.shape = mk.Solid();
  // Fixture order -x, +x, -y, +y, -z, +z (MakeBox: Back is XMin, Front XMax, Left YMin,
  // Right YMax, Bottom ZMin, Top ZMax).
  const TopoDS_Face fs[6] = {mk.BackFace(), mk.FrontFace(), mk.LeftFace(),
                             mk.RightFace(), mk.BottomFace(), mk.TopFace()};
  for (int i = 0; i < 6; ++i) add_origins(t.faces, fs[i], label(lab(in.ints[i])));
  return t;
}

// ---- delete faces

Tracked do_defeature(const Exec& x, const KInstr& in) {
  const Tracked& src = x.slot(in.ints[0]);
  const std::int64_t operand = in.ints[1];
  const std::int64_t n = in.ints[2];
  BRepAlgoAPI_Defeaturing df;
  df.SetShape(src.shape);
  for (std::int64_t i = 0; i < n; ++i) df.AddFaceToRemove(x.face(operand, in.ints[3 + i]));
  df.SetRunParallel(Standard_False);
  df.SetToFillHistory(Standard_True);
  df.Build();
  if (!df.IsDone() || df.HasErrors()) {
    std::ostringstream m;
    df.DumpErrors(m);
    fail("the faces could not be removed and healed: " + m.str());
  }
  Tracked out;
  out.shape = df.Shape();
  propagate_history(src, df.History(), out);
  require_labels(out, "the face deletion");
  return out;
}

Tracked do_remove_faces(const Exec& x, const KInstr& in) {
  const Tracked& src = x.slot(in.ints[0]);
  const std::int64_t operand = in.ints[1];
  const std::int64_t n = in.ints[2];
  Handle(BRepTools_ReShape) rs = new BRepTools_ReShape();
  for (std::int64_t i = 0; i < n; ++i) rs->Remove(x.face(operand, in.ints[3 + i]));
  TopoDS_Shape s = rs->Apply(src.shape);
  // A solid with faces removed is an open shell.
  TopoDS_Shape shell;
  for (TopExp_Explorer ex(s, TopAbs_SHELL); ex.More(); ex.Next()) {
    shell = ex.Current();
    break;
  }
  if (shell.IsNull()) fail("nothing remains after deleting the faces");
  Tracked out;
  out.shape = shell;
  out.sheet = true;
  propagate_history(src, rs->History(), out);
  return out;
}

// ---- offset faces (push-pull)

Tracked do_offset_faces(const Exec& x, const KInstr& in) {
  const Tracked& src = x.slot(in.ints[0]);
  const std::int64_t operand = in.ints[1];
  const std::int64_t n = in.ints[2];
  // One distance for every face, or one per face; optional per-face labels follow the faces
  // in the ints (-1 keeps the face's origin).
  auto dist = [&](std::int64_t i) {
    return in.reals.size() == static_cast<std::size_t>(n) ? in.reals[static_cast<std::size_t>(i)]
                                                           : in.reals[0];
  };
  const bool labelled = in.ints.size() == static_cast<std::size_t>(3 + 2 * n);
  BRepOffset_MakeOffset mk;
  mk.Initialize(src.shape, 0.0, 1e-6, BRepOffset_Skin, Standard_True, Standard_False,
                GeomAbs_Intersection, Standard_False, Standard_False);
  for (std::int64_t i = 0; i < n; ++i)
    mk.SetOffsetOnFace(x.face(operand, in.ints[3 + i]), dist(i));
  mk.MakeOffsetShape();
  if (!mk.IsDone() || mk.Shape().IsNull()) fail("the faces could not be offset");
  Tracked out;
  out.shape = mk.Shape();
  TopTools_IndexedMapOfShape faces;
  TopExp::MapShapes(out.shape, TopAbs_FACE, faces);
  const Body& b = x.operand(operand);
  const BRepAlgo_Image& image = mk.OffsetFacesFromShapes();
  for (std::size_t i = 0; i < b.faces.size(); ++i) {
    Origins o = origins_of(src.faces, b.faces[i]);
    if (labelled) {
      for (std::int64_t k = 0; k < n; ++k)
        if (in.ints[3 + k] == static_cast<std::int64_t>(i) && in.ints[3 + n + k] >= 0)
          o = label(lab(in.ints[3 + n + k]));
    }
    if (faces.Contains(b.faces[i])) add_origins(out.faces, b.faces[i], o);
    if (image.HasImage(b.faces[i])) {
      TopTools_ListOfShape imgs;
      image.LastImage(b.faces[i], imgs);
      for (const TopoDS_Shape& g : imgs)
        for (TopExp_Explorer ex(g, TopAbs_FACE); ex.More(); ex.Next())
          if (faces.Contains(ex.Current())) add_origins(out.faces, ex.Current(), o);
    }
  }
  require_labels(out, "the face offset");
  return out;
}

// ---- thicken a sheet

Tracked do_thicken(const Exec& x, const KInstr& in) {
  const Tracked& src = x.slot(in.ints[0]);
  const std::int64_t operand = in.ints[1];
  const std::int64_t far_base = in.ints[2];
  const std::int64_t rim_base = in.ints[3];
  const double d = in.reals[0];
  const bool both = in.reals[1] != 0.0;
  Tracked base = src;
  if (both) {
    // Centre the solid on the sheet: thicken an offset copy of it by the full distance.
    BRepOffset_MakeOffset half;
    half.Initialize(src.shape, -d / 2.0, 1e-6, BRepOffset_Skin, Standard_True, Standard_False,
                    GeomAbs_Intersection, Standard_False, Standard_False);
    half.MakeOffsetShape();
    if (!half.IsDone()) fail("the sheet could not be offset to centre the thicken");
    Tracked moved;
    moved.shape = half.Shape();
    moved.sheet = true;
    const BRepAlgo_Image& image = half.OffsetFacesFromShapes();
    for (int i = 1; i <= src.faces.Extent(); ++i) {
      if (!image.HasImage(src.faces.FindKey(i))) continue;
      TopTools_ListOfShape imgs;
      image.LastImage(src.faces.FindKey(i), imgs);
      for (const TopoDS_Shape& g : imgs)
        for (TopExp_Explorer ex(g, TopAbs_FACE); ex.More(); ex.Next())
          add_origins(moved.faces, ex.Current(), src.faces(i));
    }
    for (int i = 1; i <= src.edges.Extent(); ++i)
      for (const TopoDS_Shape& g : half.Generated(src.edges.FindKey(i)))
        for (TopExp_Explorer ex(g, TopAbs_EDGE); ex.More(); ex.Next())
          add_origins(moved.edges, ex.Current(), src.edges(i));
    base = moved;
  }
  BRepOffset_MakeOffset mk;
  mk.Initialize(base.shape, d, 1e-6, BRepOffset_Skin, Standard_True, Standard_False,
                GeomAbs_Intersection, Standard_True, Standard_False);
  mk.MakeOffsetShape();
  if (!mk.IsDone() || mk.Shape().IsNull()) fail("the sheet could not be thickened");
  Tracked out;
  out.shape = mk.Shape();
  TopTools_IndexedMapOfShape faces;
  TopExp::MapShapes(out.shape, TopAbs_FACE, faces);
  const BRepAlgo_Image& image = mk.OffsetFacesFromShapes();
  const Body& b = x.operand(operand);
  for (int i = 1; i <= base.faces.Extent(); ++i) {
    const TopoDS_Shape& f = base.faces.FindKey(i);
    if (faces.Contains(f)) add_origins(out.faces, f, base.faces(i));
    // The far face is tagged by the sheet face it offsets.
    std::uint32_t fi = 0;
    for (const Origin& o : base.faces(i))
      if (o.kind == kInput) fi = o.b;
    if (image.HasImage(f)) {
      TopTools_ListOfShape imgs;
      image.LastImage(f, imgs);
      for (const TopoDS_Shape& g : imgs)
        for (TopExp_Explorer ex(g, TopAbs_FACE); ex.More(); ex.Next())
          if (faces.Contains(ex.Current()) && !ex.Current().IsSame(f))
            add_origins(out.faces, ex.Current(), label(static_cast<std::uint32_t>(far_base + fi)));
    }
  }
  (void)b;
  for (int i = 1; i <= base.edges.Extent(); ++i) {
    std::uint32_t ei = kNoLabel;
    for (const Origin& o : base.edges(i))
      if (o.kind == kInput) ei = o.b;
    if (ei == kNoLabel) continue;
    for (const TopoDS_Shape& g : mk.Generated(base.edges.FindKey(i)))
      for (TopExp_Explorer ex(g, TopAbs_FACE); ex.More(); ex.Next())
        if (faces.Contains(ex.Current()) && origins_of(out.faces, ex.Current()).empty())
          add_origins(out.faces, ex.Current(), label(static_cast<std::uint32_t>(rim_base + ei)));
  }
  // A rim OCCT does not report as generated still holds the sheet's boundary edge it stands on.
  for (int i = 1; i <= faces.Extent(); ++i) {
    if (!origins_of(out.faces, faces(i)).empty()) continue;
    for (TopExp_Explorer ex(faces(i), TopAbs_EDGE); ex.More(); ex.Next()) {
      std::uint32_t ei = kNoLabel;
      for (const Origin& o : origins_of(base.edges, ex.Current()))
        if (o.kind == kInput) ei = o.b;
      if (ei == kNoLabel) {
        for (int k = 1; k <= base.edges.Extent() && ei == kNoLabel; ++k)
          for (const TopoDS_Shape& m : mk.Modified(base.edges.FindKey(k)))
            if (m.IsSame(ex.Current()))
              for (const Origin& o : base.edges(k))
                if (o.kind == kInput) ei = o.b;
      }
      if (ei == kNoLabel) continue;
      add_origins(out.faces, faces(i), label(static_cast<std::uint32_t>(rim_base + ei)));
      break;
    }
  }
  require_labels(out, "the thicken");
  return out;
}

// ---- sew sheets

Tracked do_sew(const Exec& x, const KInstr& in) {
  const bool solid = in.ints[0] != 0;
  const std::int64_t n = in.ints[1];
  BRepBuilderAPI_Sewing sew(1e-6);
  for (std::int64_t i = 0; i < n; ++i) sew.Add(x.slot(in.ints[2 + i]).shape);
  sew.Perform();
  Tracked out;
  out.shape = sew.SewedShape();
  out.sheet = true;
  for (std::int64_t i = 0; i < n; ++i) {
    const Tracked& t = x.slot(in.ints[2 + i]);
    TopTools_IndexedMapOfShape faces, edges;
    TopExp::MapShapes(out.shape, TopAbs_FACE, faces);
    TopExp::MapShapes(out.shape, TopAbs_EDGE, edges);
    for (int k = 1; k <= t.faces.Extent(); ++k) {
      const TopoDS_Shape& f = t.faces.FindKey(k);
      // A face or edge of an added sheet is a sub-shape: sewing records its image there.
      TopoDS_Shape m = sew.IsModifiedSubShape(f) ? sew.ModifiedSubShape(f) : f;
      for (TopExp_Explorer ex(m, TopAbs_FACE); ex.More(); ex.Next())
        if (faces.Contains(ex.Current())) add_origins(out.faces, ex.Current(), t.faces(k));
    }
    for (int k = 1; k <= t.edges.Extent(); ++k) {
      const TopoDS_Shape& e = t.edges.FindKey(k);
      TopoDS_Shape m = sew.IsModifiedSubShape(e) ? sew.ModifiedSubShape(e) : e;
      for (TopExp_Explorer ex(m, TopAbs_EDGE); ex.More(); ex.Next())
        if (edges.Contains(ex.Current())) add_origins(out.edges, ex.Current(), t.edges(k));
    }
  }
  if (solid) {
    TopoDS_Shell shell;
    int shells = 0;
    for (TopExp_Explorer ex(out.shape, TopAbs_SHELL); ex.More(); ex.Next()) {
      shell = TopoDS::Shell(ex.Current());
      ++shells;
    }
    if (shells != 1 || !BRep_Tool::IsClosed(shell)) fail("the sheets do not close a watertight volume");
    BRepBuilderAPI_MakeSolid ms(shell);
    if (!ms.IsDone()) fail("the knitted shell does not make a solid");
    TopoDS_Solid s = ms.Solid();
    BRepLib::OrientClosedSolid(s);
    Tracked solid_out;
    solid_out.shape = s;
    // The solid shares the shell's faces (orientation may flip; maps ignore it).
    solid_out.faces = out.faces;
    solid_out.edges = out.edges;
    return solid_out;
  }
  return out;
}

// ---- direct edits: faces hinged about their own lines (rotate, replace by a tilted plane)

// op 38: ints [slot, operand, n, faces (n), labels (n; -1 keeps the face's origin)]; reals per
// face: hinge point (3), pull (3), angle. Each planar face turns about the line where it meets
// the plane through the hinge point normal to the pull; its neighbours extend or trim to follow.
Tracked do_hinge_faces(const Exec& x, const KInstr& in) {
  const Tracked& src = x.slot(in.ints[0]);
  const std::int64_t operand = in.ints[1];
  const std::int64_t n = in.ints[2];
  BRepOffsetAPI_DraftAngle mk(src.shape);
  std::vector<TopoDS_Face> faces;
  for (std::int64_t i = 0; i < n; ++i) {
    const TopoDS_Face f = x.face(operand, in.ints[3 + i]);
    const double* r = in.reals.data() + 7 * i;
    const gp_Dir pull(r[3], r[4], r[5]);
    mk.Add(f, pull, r[6], gp_Pln(gp_Pnt(r[0], r[1], r[2]), pull));
    if (!mk.AddDone()) fail("a face cannot turn about its hinge line");
    faces.push_back(f);
  }
  mk.Build();
  if (!mk.IsDone()) fail("the turned faces could not be rebuilt with their neighbours");
  Tracked out;
  out.shape = mk.Shape();
  propagate_draft(src, mk, out);
  for (std::int64_t i = 0; i < n; ++i) {
    const std::int64_t l = in.ints[3 + n + i];
    if (l < 0) continue;
    const TopoDS_Shape m = mk.ModifiedShape(faces[static_cast<std::size_t>(i)]);
    for (TopExp_Explorer ex(m, TopAbs_FACE); ex.More(); ex.Next())
      if (Origins* o = out.faces.ChangeSeek(ex.Current())) *o = label(lab(l));
  }
  require_labels(out, "the face rotation");
  return out;
}

// ---- offset surfaces

namespace {
// The faces of `operand` adjacent to each of its edges.
TopTools_IndexedDataMapOfShapeListOfShape edge_faces_of(const Body& b) {
  TopTools_IndexedDataMapOfShapeListOfShape m;
  TopExp::MapShapesAndUniqueAncestors(b.shape, TopAbs_EDGE, TopAbs_FACE, m);
  return m;
}

int face_index_of(const Body& b, const TopoDS_Shape& f) {
  for (std::size_t i = 0; i < b.faces.size(); ++i)
    if (b.faces[i].IsSame(f)) return static_cast<int>(i);
  return -1;
}

// The boundary label of an edge of a face set taken out of `b`: the input edge itself where it
// was already a boundary of `b`, else the face of `b` across it outside the set.
Origins boundary_origins(const Tracked& src, const Body& b,
                         const TopTools_IndexedDataMapOfShapeListOfShape& ef,
                         const TopTools_IndexedMapOfShape& chosen, const TopoDS_Shape& e,
                         std::int64_t neighbour_base, std::uint32_t none_label) {
  const TopTools_ListOfShape* fs = ef.Seek(e);
  if (fs == nullptr || fs->Extent() < 2) {
    const Origins& o = origins_of(src.edges, e);
    if (!o.empty()) return o;
    return label(none_label);
  }
  for (const TopoDS_Shape& f : *fs) {
    if (chosen.Contains(f)) continue;
    const int k = face_index_of(b, f);
    if (k >= 0 && neighbour_base >= 0) return label(lab(neighbour_base + k));
  }
  return label(none_label);
}
}  // namespace

// op 39: ints [slot, operand, offset label base, neighbour label base, none label, n, faces
// (n; none: every face)]; reals [distance]. A new sheet: the chosen faces of a sheet or solid
// offset along their normals, topology kept face for face. Faces take the offset label of the
// face they came from; boundary edges the input boundary edge, or the face across them.
Tracked do_offset_sheet(const Exec& x, const KInstr& in) {
  const Tracked& src = x.slot(in.ints[0]);
  const std::int64_t operand = in.ints[1];
  const std::int64_t offset_base = in.ints[2];
  const std::int64_t neighbour_base = in.ints[3];
  const std::uint32_t none_label = lab(in.ints[4]);
  const std::int64_t n = in.ints[5];
  const Body& b = x.operand(operand);
  TopTools_IndexedMapOfShape chosen;
  if (n == 0) {
    for (const TopoDS_Face& f : b.faces) chosen.Add(f);
  } else {
    for (std::int64_t i = 0; i < n; ++i) chosen.Add(x.face(operand, in.ints[6 + i]));
  }
  // The chosen faces as one shell, oriented as in the body.
  TopoDS_Shell shell;
  BRep_Builder bb;
  bb.MakeShell(shell);
  for (TopExp_Explorer ex(b.shape, TopAbs_FACE); ex.More(); ex.Next())
    if (chosen.Contains(ex.Current())) bb.Add(shell, ex.Current());
  BRepOffset_MakeSimpleOffset mk(shell, in.reals[0]);
  mk.SetBuildSolidFlag(Standard_False);
  mk.Perform();
  if (!mk.IsDone() || mk.GetResultShape().IsNull()) fail("the faces could not be offset");
  Tracked out;
  out.shape = mk.GetResultShape();
  out.sheet = true;
  TopTools_IndexedMapOfShape present_faces, present_edges;
  TopExp::MapShapes(out.shape, TopAbs_FACE, present_faces);
  TopExp::MapShapes(out.shape, TopAbs_EDGE, present_edges);
  for (int i = 1; i <= chosen.Extent(); ++i) {
    const int k = face_index_of(b, chosen(i));
    if (k < 0) fail("an offset face is not a face of its body");
    const TopoDS_Shape g = mk.Generated(chosen(i));
    if (g.IsNull()) incomplete("an offset face has no image");
    for (TopExp_Explorer ex(g, TopAbs_FACE); ex.More(); ex.Next())
      if (present_faces.Contains(ex.Current()))
        add_origins(out.faces, ex.Current(), label(lab(offset_base + k)));
  }
  const auto ef = edge_faces_of(b);
  TopTools_IndexedDataMapOfShapeListOfShape shell_ef;
  TopExp::MapShapesAndUniqueAncestors(shell, TopAbs_EDGE, TopAbs_FACE, shell_ef);
  for (int i = 1; i <= shell_ef.Extent(); ++i) {
    if (shell_ef(i).Extent() != 1) continue;
    const TopoDS_Shape& e = shell_ef.FindKey(i);
    const TopoDS_Shape g = mk.Generated(e);
    if (g.IsNull()) incomplete("an offset boundary edge has no image");
    const Origins o = boundary_origins(src, b, ef, chosen, e, neighbour_base, none_label);
    for (TopExp_Explorer ex(g, TopAbs_EDGE); ex.More(); ex.Next())
      if (present_edges.Contains(ex.Current())) add_origins(out.edges, ex.Current(), o);
  }
  require_labels(out, "the surface offset");
  return out;
}

// ---- up to next over several faces

namespace {
// The sign of a face's oriented normal along `dir` over its parameter range: +1, -1, 0 when the
// face runs along `dir`, 2 when it turns both ways.
int facing(const TopoDS_Face& f, const gp_Dir& dir) {
  BRepAdaptor_Surface s(f, Standard_True);
  const bool rev = f.Orientation() == TopAbs_REVERSED;
  int sign = 0;
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      const double u = s.FirstUParameter() + (s.LastUParameter() - s.FirstUParameter()) * (i + 0.5) / 5.0;
      const double v = s.FirstVParameter() + (s.LastVParameter() - s.FirstVParameter()) * (j + 0.5) / 5.0;
      gp_Pnt p;
      gp_Vec du, dv;
      s.D1(u, v, p, du, dv);
      gp_Vec n = du.Crossed(dv);
      if (n.Magnitude() < 1e-12) continue;
      if (rev) n.Reverse();
      const double d = n.Normalized().Dot(gp_Vec(dir));
      if (std::abs(d) < 1e-9) continue;
      const int sg = d > 0 ? 1 : -1;
      if (sign == 0) sign = sg;
      else if (sign != sg) return 2;
    }
  }
  return sign;
}
}  // namespace

// op 41: ints [slot, keep label, cut label, n, operands (n)], reals [direction (3), length].
// A long tool trimmed where each of its rays along `direction` first meets an operand face:
// the operand faces inside the tool, swept on along the direction, are cut away, and the piece
// holding the `keep` face stays. Every face the cut leaves takes the cut label.
Tracked trim_to_next(const Exec& x, const KInstr& in) {
  const auto& I = in.ints;
  const Tracked& src = x.slot(I[0]);
  const std::uint32_t keep = lab(I[1]);
  const std::uint32_t cut = lab(I[2]);
  const std::int64_t n = I[3];
  const gp_Dir dir(in.reals[0], in.reals[1], in.reals[2]);
  const double length = in.reals[3];
  std::vector<Tracked> ops;
  for (std::int64_t k = 0; k < n; ++k)
    ops.push_back(track_operand(x.operand(I[4 + k]), static_cast<std::uint32_t>(I[4 + k])));
  std::vector<const Tracked*> tools;
  for (const Tracked& t : ops) tools.push_back(&t);
  const Tracked inside = boolean_op(2, {&src}, tools);
  TopTools_ListOfShape prisms;
  for (TopExp_Explorer ex(inside.shape, TopAbs_FACE); ex.More(); ex.Next()) {
    const Origins& o = origins_of(inside.faces, ex.Current());
    if (std::none_of(o.begin(), o.end(), [](const Origin& g) { return g.kind == kInput; })) continue;
    const TopoDS_Face f = TopoDS::Face(ex.Current());
    const int s = facing(f, dir);
    if (s == 0) continue;
    if (s == 2) unsupported("up to next over a face that turns both ways along the extrude");
    BRepPrimAPI_MakePrism mk(f, gp_Vec(dir) * length, Standard_True);
    if (!mk.IsDone()) fail("up to next: a face in the way cannot be swept on");
    prisms.Append(mk.Shape());
  }
  if (prisms.IsEmpty()) fail("up to next: the profile does not meet a next face");
  Tracked shadow;
  if (prisms.Extent() == 1) {
    shadow.shape = prisms.First();
  } else {
    BRepAlgoAPI_Fuse fu;
    TopTools_ListOfShape a, b;
    a.Append(prisms.First());
    bool first = true;
    for (const TopoDS_Shape& p : prisms) {
      if (first) { first = false; continue; }
      b.Append(p);
    }
    fu.SetArguments(a);
    fu.SetTools(b);
    fu.SetRunParallel(Standard_False);
    fu.Build();
    if (!fu.IsDone() || fu.HasErrors()) fail("up to next: the faces in the way could not be joined");
    shadow.shape = fu.Shape();
  }
  for (TopExp_Explorer ex(shadow.shape, TopAbs_FACE); ex.More(); ex.Next())
    add_origins(shadow.faces, ex.Current(), label(cut));
  Tracked cutres = boolean_op(1, {&src}, {&shadow});
  // The piece holding the kept face.
  TopoDS_Compound c;
  BRep_Builder bb;
  bb.MakeCompound(c);
  int pieces = 0;
  for (TopExp_Explorer sx(cutres.shape, TopAbs_SOLID); sx.More(); sx.Next()) {
    bool has = false;
    for (TopExp_Explorer fx(sx.Current(), TopAbs_FACE); fx.More() && !has; fx.Next()) {
      const Origins& o = origins_of(cutres.faces, fx.Current());
      has = std::any_of(o.begin(), o.end(), [&](const Origin& g) { return g.kind == kLabel && g.a == keep; });
    }
    if (!has) continue;
    for (TopExp_Explorer fx(sx.Current(), TopAbs_FACE); fx.More(); fx.Next()) {
      const Origins& o = origins_of(cutres.faces, fx.Current());
      if (std::any_of(o.begin(), o.end(), [](const Origin& g) { return g.kind == kLabel && g.a == 0xfffffffeu; }))
        fail("up to next: the profile does not meet a next face");
    }
    bb.Add(c, sx.Current());
    ++pieces;
  }
  if (pieces == 0) fail("up to next: nothing of the extrude is left before the next face");
  Tracked out = cutres;
  out.shape = pieces == 1 ? TopoDS_Iterator(c).Value() : TopoDS_Shape(c);
  return out;
}

// ---- conic and curvature-continuous blends

// op 43: ints [slot, operand, edge label base, n, edges (n), profile (0 conic, 1 curvature)],
// reals [radius, rho]. Each straight edge between two planar faces is blended by a section
// swept along it: tangent to both faces at the setback a circular blend of `radius` has
// (radius / tan(α/2) from the edge, α the angle between the faces). The section is a conic of
// `rho` (a rational quadratic through the corner point) or a quartic whose ends have zero
// curvature (curvature-continuous with the planes). The region between the section and the
// edge is cut away (a convex edge) or filled (a concave edge). The edge's ends must be planar
// faces square to it.
Tracked do_section_blend(const Exec& x, const KInstr& in) {
  const auto& I = in.ints;
  const Tracked& src = x.slot(I[0]);
  const std::int64_t operand = I[1];
  const std::int64_t base = I[2];
  const std::int64_t n = I[3];
  const int kind = static_cast<int>(I[4 + n]);
  const double radius = in.reals[0];
  const double rho = in.reals[1];
  if (radius <= 0) fail("a blend needs a positive radius");
  if (kind == 0 && (rho <= 0.0 || rho >= 1.0)) fail("a conic blend's rho must lie between 0 and 1");
  const Body& b = x.operand(operand);
  TopTools_IndexedDataMapOfShapeListOfShape ef, vf;
  TopExp::MapShapesAndUniqueAncestors(b.shape, TopAbs_EDGE, TopAbs_FACE, ef);
  TopExp::MapShapesAndUniqueAncestors(b.shape, TopAbs_VERTEX, TopAbs_FACE, vf);
  std::vector<Tracked> cut_tools, fill_tools;
  for (std::int64_t i = 0; i < n; ++i) {
    const std::int64_t ei = I[4 + i];
    const TopoDS_Edge e = x.edge(operand, ei);
    BRepAdaptor_Curve c(e);
    if (c.GetType() != GeomAbs_Line) unsupported("conic and curvature-continuous blends of curved edges");
    const TopTools_ListOfShape& fs = ef.FindFromKey(e);
    if (fs.Extent() != 2) fail("a blended edge needs two faces");
    const TopoDS_Face f1 = TopoDS::Face(fs.First()), f2 = TopoDS::Face(fs.Last());
    gp_Pnt p0 = c.Value(c.FirstParameter()), p1 = c.Value(c.LastParameter());
    const gp_Vec along(p0, p1);
    const double len = along.Magnitude();
    if (len < 1e-12) fail("a blended edge has no length");
    const gp_Dir t(along);
    // Per face: its outward normal and the direction into it across the edge.
    auto side = [&](const TopoDS_Face& f, gp_Dir& normal, gp_Dir& into) {
      BRepAdaptor_Surface s(f, Standard_True);
      if (s.GetType() != GeomAbs_Plane) unsupported("conic and curvature-continuous blends next to curved faces");
      normal = s.Plane().Axis().Direction();
      if (!s.Plane().Direct()) normal.Reverse();
      if (f.Orientation() == TopAbs_REVERSED) normal.Reverse();
      into = gp_Dir(gp_Vec(t).Crossed(gp_Vec(normal)));
      const gp_Pnt m = c.Value(0.5 * (c.FirstParameter() + c.LastParameter()));
      const double probe = std::min(radius, len) * 1e-3;
      gp_Pnt q = m.Translated(gp_Vec(into) * probe);
      BRepClass_FaceClassifier fc(f, q, 1e-9);
      if (fc.State() != TopAbs_IN) into.Reverse();
    };
    gp_Dir n1, n2, d1, d2;
    side(f1, n1, d1);
    side(f2, n2, d2);
    const double cosa = gp_Vec(d1).Dot(gp_Vec(d2));
    const double alpha = std::acos(std::max(-1.0, std::min(1.0, cosa)));
    if (alpha < 1e-6 || alpha > M_PI - 1e-6) unsupported("blends of an edge where its two faces meet tangentially");
    const bool convex = gp_Vec(n1).Dot(gp_Vec(d2)) < 0;
    const double s = radius / std::tan(alpha / 2.0);
    // The edge's ends: planar faces square to it. The swept section's caps continue them.
    TopoDS_Shape ends[2];
    {
      const TopoDS_Vertex vs[2] = {TopExp::FirstVertex(e, Standard_True), TopExp::LastVertex(e, Standard_True)};
      for (int k = 0; k < 2; ++k) {
        for (const TopoDS_Shape& g : vf.FindFromKey(vs[k])) {
          if (g.IsSame(f1) || g.IsSame(f2)) continue;
          BRepAdaptor_Surface gs(TopoDS::Face(g), Standard_True);
          if (gs.GetType() != GeomAbs_Plane || !gs.Plane().Axis().Direction().IsParallel(t, 1e-9))
            unsupported("conic and curvature-continuous blends of edges whose ends are not square faces");
          if (ends[k].IsNull()) ends[k] = g;
        }
      }
      // The first vertex is where the section starts.
      if (BRep_Tool::Pnt(vs[0]).Distance(p0) > 1e-9) std::swap(ends[0], ends[1]);
    }
    // The section in the plane square to the edge at its start.
    const gp_Pnt V = p0;
    const gp_Pnt T1 = V.Translated(gp_Vec(d1) * s), T2 = V.Translated(gp_Vec(d2) * s);
    Handle(Geom_Curve) arc;
    if (kind == 0) {
      TColgp_Array1OfPnt poles(1, 3);
      poles(1) = T1; poles(2) = V; poles(3) = T2;
      TColStd_Array1OfReal w(1, 3);
      w(1) = 1.0; w(2) = rho / (1.0 - rho); w(3) = 1.0;
      arc = new Geom_BezierCurve(poles, w);
    } else {
      TColgp_Array1OfPnt poles(1, 5);
      poles(1) = T1;
      poles(2) = gp_Pnt(0.5 * (T1.XYZ() + V.XYZ()));
      poles(3) = V;
      poles(4) = gp_Pnt(0.5 * (T2.XYZ() + V.XYZ()));
      poles(5) = T2;
      arc = new Geom_BezierCurve(poles);
    }
    BRepBuilderAPI_MakeWire mw;
    const TopoDS_Edge ce = BRepBuilderAPI_MakeEdge(arc).Edge();
    mw.Add(ce);
    mw.Add(BRepBuilderAPI_MakeEdge(T2, V).Edge());
    mw.Add(BRepBuilderAPI_MakeEdge(V, T1).Edge());
    if (!mw.IsDone()) fail("the blend section could not be closed");
    BRepBuilderAPI_MakeFace mf(mw.Wire(), Standard_True);
    if (!mf.IsDone()) fail("the blend section is not planar");
    BRepPrimAPI_MakePrism mk(mf.Face(), along, Standard_True);
    if (!mk.IsDone()) fail("the blend could not be swept along its edge");
    Tracked tool;
    tool.shape = mk.Shape();
    const TopoDS_Shape blend = mk.Generated(ce).First();
    for (TopExp_Explorer ex(tool.shape, TopAbs_FACE); ex.More(); ex.Next()) {
      const TopoDS_Shape& g = ex.Current();
      if (g.IsSame(blend)) {
        add_origins(tool.faces, g, label(lab(base + ei)));
      } else if (g.IsSame(mk.FirstShape()) && !ends[0].IsNull()) {
        add_origins(tool.faces, g, origins_of(src.faces, ends[0]));
      } else if (g.IsSame(mk.LastShape()) && !ends[1].IsNull()) {
        add_origins(tool.faces, g, origins_of(src.faces, ends[1]));
      } else {
        add_origins(tool.faces, g, label(kInternalLabel));
      }
    }
    (convex ? cut_tools : fill_tools).push_back(std::move(tool));
  }
  Tracked cur = src;
  for (int pass = 0; pass < 2; ++pass) {
    std::vector<Tracked>& tools = pass == 0 ? cut_tools : fill_tools;
    if (tools.empty()) continue;
    std::vector<const Tracked*> tp;
    for (const Tracked& t : tools) tp.push_back(&t);
    cur = boolean_op(pass == 0 ? 1 : 0, {&cur}, tp);
  }
  // Only the swept sections' blend faces may be new: any other tool face on the result means
  // the section ran past its edge.
  TopTools_IndexedMapOfShape faces;
  TopExp::MapShapes(cur.shape, TopAbs_FACE, faces);
  for (int i = 1; i <= faces.Extent(); ++i) {
    const Origins& o = origins_of(cur.faces, faces(i));
    Origins kept;
    for (const Origin& g : o)
      if (!(g.kind == kLabel && g.a == kInternalLabel)) kept.push_back(g);
    if (kept.empty()) fail("a conic or curvature-continuous blend ran past its edge");
    if (Origins* m = cur.faces.ChangeSeek(faces(i))) *m = kept;
  }
  require_labels(cur, "the blend");
  return cur;
}

// ---- mid-surfaces

// op 40: ints [operand, neighbour label base, none label, n pairs, then per pair: face a,
// face c, face label]. Per pair of opposite parallel planar faces, the patch on the plane
// halfway between them where the two faces overlap. Its boundary edges take the face of the
// body across them, measured on face a.
Tracked do_mid_surface(const Exec& x, const KInstr& in) {
  const std::int64_t operand = in.ints[0];
  const std::int64_t neighbour_base = in.ints[1];
  const std::uint32_t none_label = lab(in.ints[2]);
  const std::int64_t n = in.ints[3];
  const Body& b = x.operand(operand);
  TopTools_IndexedMapOfShape selected;
  for (std::int64_t i = 0; i < n; ++i) {
    selected.Add(x.face(operand, in.ints[4 + 3 * i]));
    selected.Add(x.face(operand, in.ints[5 + 3 * i]));
  }
  TopoDS_Compound c;
  BRep_Builder bb;
  bb.MakeCompound(c);
  Tracked out;
  out.sheet = true;
  for (std::int64_t i = 0; i < n; ++i) {
    const TopoDS_Face fa = x.face(operand, in.ints[4 + 3 * i]);
    const TopoDS_Face fc = x.face(operand, in.ints[5 + 3 * i]);
    const std::uint32_t face_label = lab(in.ints[6 + 3 * i]);
    BRepAdaptor_Surface sa(fa), sc(fc);
    if (sa.GetType() != GeomAbs_Plane || sc.GetType() != GeomAbs_Plane)
      unsupported("mid-surfaces of curved faces");
    gp_Dir na = sa.Plane().Axis().Direction();
    gp_Dir nc = sc.Plane().Axis().Direction();
    if (!na.IsParallel(nc, 1e-9)) fail("a mid-surface pair must be opposite parallel faces");
    const double gap = gp_Vec(sa.Plane().Location(), sc.Plane().Location()).Dot(gp_Vec(na));
    if (std::abs(gap) < 1e-12) fail("a mid-surface pair must be opposite parallel faces");
    // Both faces moved onto the mid plane, then their overlap.
    gp_Trsf ta, tc;
    ta.SetTranslation(gp_Vec(na) * (0.5 * gap));
    tc.SetTranslation(gp_Vec(na) * (-0.5 * gap));
    const TopoDS_Shape ma = BRepBuilderAPI_Transform(fa, ta, Standard_True).Shape();
    const TopoDS_Shape mc = BRepBuilderAPI_Transform(fc, tc, Standard_True).Shape();
    BRepAlgoAPI_Common common(ma, mc);
    if (!common.IsDone()) fail("the mid-surface overlap could not be computed");
    ShapeUpgrade_UnifySameDomain unify(common.Shape(), Standard_True, Standard_True, Standard_False);
    unify.Build();
    TopoDS_Face mid;
    for (TopExp_Explorer ex(unify.Shape(), TopAbs_FACE); ex.More(); ex.Next()) {
      if (!mid.IsNull()) fail("a mid-surface pair overlaps in several pieces");
      mid = TopoDS::Face(ex.Current());
    }
    if (mid.IsNull()) fail("a mid-surface pair does not overlap");
    // The patch faces the way face a does.
    {
      BRepAdaptor_Surface sm(mid, Standard_True);
      gp_Dir nm = sm.Plane().Axis().Direction();
      if (mid.Orientation() == TopAbs_REVERSED) nm.Reverse();
      gp_Dir nf = na;
      if (fa.Orientation() == TopAbs_REVERSED) nf.Reverse();
      if (nm.Dot(nf) < 0) mid.Reverse();
    }
    add_origins(out.faces, mid, label(face_label));
    for (TopExp_Explorer ex(mid, TopAbs_EDGE); ex.More(); ex.Next()) {
      const TopoDS_Edge e = TopoDS::Edge(ex.Current());
      BRepAdaptor_Curve ce(e);
      const gp_Pnt m = ce.Value(0.5 * (ce.FirstParameter() + ce.LastParameter()))
                           .Translated(gp_Vec(na) * (-0.5 * gap));
      std::uint32_t l = none_label;
      const TopoDS_Vertex v = BRepBuilderAPI_MakeVertex(m).Vertex();
      for (std::size_t k = 0; k < b.faces.size(); ++k) {
        if (selected.Contains(b.faces[k])) continue;
        BRepExtrema_DistShapeShape d(v, b.faces[k]);
        if (d.IsDone() && d.Value() < 1e-7) {
          l = lab(neighbour_base + static_cast<std::int64_t>(k));
          break;
        }
      }
      add_origins(out.edges, e, label(l));
    }
    bb.Add(c, mid);
  }
  out.shape = c;
  return out;
}
}  // namespace

// Implemented in exact_sweep_loft.cc.
Profile offset_profile(const Profile& p, const KInstr& in);
Tracked build_planar_sheet(const Profile& p, std::uint32_t face_label);
Tracked split_keep(const Tracked& src, const KInstr& in, const Body& operand);
Tracked prism_to_body(const Profile& p, const KInstr& in, const Body& operand);
Tracked import_step(const KInstr& in, KResult& result);

// ------------------------------------------------------------------------- the interpreter

KResult kernel_execute(rust::Slice<const KInstr> program,
                       rust::Slice<const std::uint64_t> operands) {
  const auto guard = kernel_lock();
  Exec x{program, {}, {}, {}, {}, {}};
  // Bodies registered by earlier outputs are released when a later instruction fails.
  auto release_partial = [&x]() {
    for (const KBody& b : x.result.bodies) kernel_release(b.id);
    x.result.bodies.clear();
  };
  try {
    for (std::uint64_t id : operands) x.operands.push_back(registry_get(id));
    std::uint32_t outputs = 0;
    for (const KInstr& in : program) {
      Tracked t;
      Profile p;
      bool prof = false;
      const auto& I = in.ints;
      const auto& R = in.reals;
      switch (in.op) {
        case 1: t = track_operand(x.operand(I[0]), static_cast<std::uint32_t>(I[0])); break;
        case 2: p = build_profile(in); prof = true; t = p.tracked; break;
        case 3: t = build_prism(x.profile(I[0]), R[0], R[1], lab(I[1]), lab(I[2]), R.size() > 2 ? R[2] : 0.0, R.size() > 3 ? R[3] : 0.0); break;
        case 4: t = build_revolve(x.profile(I[0]), gp_Ax1(gp_Pnt(R[0], R[1], R[2]), gp_Dir(R[3], R[4], R[5])), R[6], lab(I[1]), lab(I[2])); break;
        case 5: {
          std::vector<gp_Pnt> path;
          for (std::size_t k = 0; k + 2 < R.size(); k += 3) path.emplace_back(R[k], R[k + 1], R[k + 2]);
          t = build_sweep(x.profile(I[0]), path, lab(I[1]), lab(I[2]));
          break;
        }
        case 6: {
          std::vector<const Profile*> secs;
          for (std::int64_t k = 0; k < I[2]; ++k) secs.push_back(&x.profile(I[3 + k]));
          t = build_loft(secs, lab(I[0]), lab(I[1]));
          break;
        }
        case 7: {
          std::vector<const Tracked*> args, tools;
          const std::int64_t na = I[1];
          for (std::int64_t k = 0; k < na; ++k) args.push_back(&x.slot(I[2 + k]));
          const std::int64_t nt = I[2 + na];
          for (std::int64_t k = 0; k < nt; ++k) tools.push_back(&x.slot(I[3 + na + k]));
          t = boolean_op(static_cast<int>(I[0]), args, tools);
          break;
        }
        case 8: t = do_fillet(x, in); break;
        case 9: t = do_chamfer(x, in); break;
        case 10: t = do_vertex_chamfer(x, in); break;
        case 11: t = do_shell(x, in); break;
        case 12: t = do_draft(x, in); break;
        case 13: t = do_transform(x.slot(I[0]), trsf_from(R.data()), I[1]); break;
        case 14: t = do_scale(x.slot(I[0]), R.data()); break;
        case 15: t = do_split(x.slot(I[0]), frame_from(R.data()), lab(I[1]), lab(I[2])); break;
        case 16: {
          t = cut_half_space(x.slot(I[0]), gp_Pln(gp_Pnt(R[0], R[1], R[2]), gp_Dir(R[3], R[4], R[5])), true, lab(I[1]));
          t.sheet = x.slot(I[0]).sheet;
          // A sheet's new boundary along the plane takes the edge label.
          if (t.sheet && I.size() > 2 && I[2] >= 0) {
            TopTools_IndexedMapOfShape es;
            TopExp::MapShapes(t.shape, TopAbs_EDGE, es);
            for (int k = 1; k <= es.Extent(); ++k)
              if (origins_of(t.edges, es(k)).empty()) add_origins(t.edges, es(k), label(lab(I[2])));
          }
          break;
        }
        case 17: t = do_box(in); break;
        case 18: t = import_step(in, x.result); break;
        case 19: t = do_defeature(x, in); break;
        case 20: t = do_remove_faces(x, in); break;
        case 21: t = do_offset_faces(x, in); break;
        case 22: {
          MergePolicy policy;
          policy.merge_collinear = I[1] != 0;
          for (std::int64_t k = 0; k < I[2]; ++k) policy.side_labels.push_back(static_cast<std::uint32_t>(I[3 + k]));
          std::sort(policy.side_labels.begin(), policy.side_labels.end());
          const Tracked& src = x.slot(I[0]);
          Tracked merged = merge_same_domain(src, policy);
          check_valid(merged.shape, "the operation");
          for (const auto& piece : pieces_of(merged.shape)) {
            KBody body = body_payload(merged, piece.first, piece.second || merged.sheet);
            body.output = outputs;
            x.result.bodies.push_back(std::move(body));
          }
          ++outputs;
          t = merged;
          break;
        }
        case 23: t = do_sew(x, in); break;
        case 24: t = do_thicken(x, in); break;
        case 26: t = build_planar_sheet(x.profile(I[0]), lab(I[1])); t.sheet = true; break;
        case 27: {
          TopoDS_Compound c;
          BRep_Builder b;
          b.MakeCompound(c);
          for (std::int64_t k = 0; k < I[0]; ++k) {
            const Tracked& s = x.slot(I[1 + k]);
            b.Add(c, s.shape);
            for (int j = 1; j <= s.faces.Extent(); ++j) add_origins(t.faces, s.faces.FindKey(j), s.faces(j));
            for (int j = 1; j <= s.edges.Extent(); ++j) add_origins(t.edges, s.edges.FindKey(j), s.edges(j));
            t.sheet = t.sheet || s.sheet;
          }
          t.shape = c;
          break;
        }
        case 28: p = offset_profile(x.profile(I[0]), in); prof = true; t = p.tracked; break;
        case 30: t = split_keep(x.slot(I[0]), in, x.operand(I[1])); break;
        case 37: t = do_extend(x, in); break;
        case 38: t = do_hinge_faces(x, in); break;
        case 39: t = do_offset_sheet(x, in); break;
        case 40: t = do_mid_surface(x, in); break;
        case 41: t = trim_to_next(x, in); break;
        case 43: t = do_section_blend(x, in); break;
        case 35: t = do_delete_hole(x, in); break;
        case 36: t = do_fill(x, in); break;
        case 44: t = do_boundary(x, in); break;
        case 34: t = do_ruled(x, in); t.sheet = true; break;
        case 33: t = do_full_round(x.slot(I[0]), lab(I[1]), R.data()); break;
        case 32: t = do_cut_with_sheet(x.slot(I[0]), x.slot(I[1]), I[2] != 0, lab(I[3])); break;
        case 31: t = prism_to_body(x.profile(I[0]), in, x.operand(I[3])); break;
        default: unsupported("unknown kernel instruction");
      }
      x.slots.push_back(std::move(t));
      x.profiles.push_back(std::move(p));
      x.is_profile.push_back(prof);
    }
    return std::move(x.result);
  } catch (const Standard_Failure& e) {
    release_partial();
    fail(std::string("OCCT raised ") + e.DynamicType()->Name() + ": " + e.GetMessageString());
  } catch (...) {
    release_partial();
    throw;
  }
}

// ------------------------------------------------------------------------- ray queries

rust::Vec<double> kernel_ray_hits(rust::Slice<const std::uint64_t> bodies,
                                  rust::Slice<const std::int64_t> face_filter,
                                  rust::Slice<const double> ray) {
  const auto guard = kernel_lock();
  // Returns (t, body, face, entering) quadruples sorted by t along the ray (both ways).
  try {
    rust::Vec<double> out;
    const gp_Lin line(gp_Pnt(ray[0], ray[1], ray[2]), gp_Dir(ray[3], ray[4], ray[5]));
    std::vector<std::array<double, 4>> hits;
    for (std::size_t bi = 0; bi < bodies.size(); ++bi) {
      auto b = registry_get(bodies[bi]);
      for (std::size_t fi = 0; fi < b->faces.size(); ++fi) {
        if (!face_filter.empty()) {
          bool want = false;
          for (std::size_t k = 0; k + 1 < face_filter.size(); k += 2)
            want |= face_filter[k] == static_cast<std::int64_t>(bi) &&
                    face_filter[k + 1] == static_cast<std::int64_t>(fi);
          if (!want) continue;
        }
        IntCurvesFace_ShapeIntersector it;
        it.Load(b->faces[fi], 1e-9);
        it.Perform(line, -1e6, 1e6);
        if (!it.IsDone()) continue;
        for (int k = 1; k <= it.NbPnt(); ++k) {
          const double t = it.WParameter(k);
          const double entering = it.Transition(k) == IntCurveSurface_In ? 1.0 : 0.0;
          hits.push_back({t, static_cast<double>(bi), static_cast<double>(fi), entering});
        }
      }
    }
    std::sort(hits.begin(), hits.end());
    for (const auto& h : hits)
      for (double v : h) out.push_back(v);
    return out;
  } catch (const Standard_Failure& e) {
    fail(std::string("OCCT raised ") + e.GetMessageString());
  }
}

rust::Vec<double> kernel_extent(rust::Slice<const std::uint64_t> bodies,
                                rust::Slice<const double> axis) {
  const auto guard = kernel_lock();
  // The range of (p - o)·d over the bodies, exact on their geometry.
  try {
    const gp_Pnt o(axis[0], axis[1], axis[2]);
    const gp_Dir d(axis[3], axis[4], axis[5]);
    gp_Ax3 frame(o, d);
    gp_Trsf to_local;
    to_local.SetTransformation(frame);
    double lo = 1e300, hi = -1e300;
    for (std::uint64_t id : bodies) {
      auto b = registry_get(id);
      TopoDS_Shape local = BRepBuilderAPI_Transform(b->shape, to_local, Standard_True).Shape();
      Bnd_Box box;
      BRepBndLib::AddOptimal(local, box, Standard_False, Standard_False);
      if (box.IsVoid()) continue;
      double x0, y0, z0, x1, y1, z1;
      box.Get(x0, y0, z0, x1, y1, z1);
      lo = std::min(lo, z0);
      hi = std::max(hi, z1);
    }
    rust::Vec<double> out;
    out.push_back(lo);
    out.push_back(hi);
    return out;
  } catch (const Standard_Failure& e) {
    fail(std::string("OCCT raised ") + e.GetMessageString());
  }
}

}  // namespace occt_kernel
}  // namespace runmat_geometry_io
