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
#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_GTransform.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeSolid.hxx>
#include <BRepBuilderAPI_Sewing.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepClass3d_SolidClassifier.hxx>
#include <BRepFilletAPI_MakeChamfer.hxx>
#include <BRepFilletAPI_MakeFillet.hxx>
#include <BRepGProp.hxx>
#include <BRepLib.hxx>
#include <BRepOffsetAPI_DraftAngle.hxx>
#include <BRepOffset_MakeOffset.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
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
#include <Geom_Plane.hxx>
#include <IntCurvesFace_ShapeIntersector.hxx>
#include <ShapeUpgrade_UnifySameDomain.hxx>
#include <Standard_Failure.hxx>
#include <Standard_Version.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Shell.hxx>
#include <TopoDS_Solid.hxx>
#include <gp_GTrsf.hxx>
#include <gp_Lin.hxx>
#include <gp_Pln.hxx>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <map>
#include <mutex>
#include <set>
#include <sstream>

namespace runmat_geometry_io {
namespace occt_kernel {

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

// Output bodies of a shape, in a deterministic order: solids, then free shells, then free
// faces gathered into one sheet.
std::vector<std::pair<TopoDS_Shape, bool>> pieces_of(const TopoDS_Shape& s) {
  std::vector<std::pair<TopoDS_Shape, bool>> out;
  TopTools_IndexedMapOfShape in_solid;
  for (TopExp_Explorer ex(s, TopAbs_SOLID); ex.More(); ex.Next()) {
    out.emplace_back(ex.Current(), false);
    TopExp::MapShapes(ex.Current(), TopAbs_FACE, in_solid);
  }
  for (TopExp_Explorer ex(s, TopAbs_SHELL, TopAbs_SOLID); ex.More(); ex.Next()) {
    out.emplace_back(ex.Current(), true);
    TopExp::MapShapes(ex.Current(), TopAbs_FACE, in_solid);
  }
  TopoDS_Compound loose;
  BRep_Builder b;
  b.MakeCompound(loose);
  bool any = false;
  for (TopExp_Explorer ex(s, TopAbs_FACE, TopAbs_SHELL); ex.More(); ex.Next()) {
    if (in_solid.Contains(ex.Current())) continue;
    b.Add(loose, ex.Current());
    any = true;
  }
  if (any) {
    BRepBuilderAPI_Sewing sew(1e-7);
    sew.Add(loose);
    sew.Perform();
    // Free faces that do not share edges stay separate sheets.
    TopoDS_Shape sewn = sew.SewedShape();
    if (sewn.ShapeType() == TopAbs_FACE) {
      TopoDS_Shell sh;
      b.MakeShell(sh);
      b.Add(sh, sewn);
      out.emplace_back(sh, true);
    } else {
      for (TopExp_Explorer ex(loose, TopAbs_FACE); ex.More(); ex.Next()) {
        TopoDS_Shell sh;
        b.MakeShell(sh);
        b.Add(sh, ex.Current());
        out.emplace_back(sh, true);
      }
    }
  }
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
    if (origins_of(t.faces, faces(i)).empty()) incomplete(std::string(what) + " made a face OCCT reports no history for");
  }
}

void check_valid(const TopoDS_Shape& s, const char* what) {
  if (s.IsNull()) fail(std::string(what) + " produced nothing");
  BRepCheck_Analyzer check(s, Standard_True);
  if (!check.IsValid()) fail(std::string(what) + " produced an invalid shape");
}

// ---- fillet

Tracked do_fillet(const Exec& x, const KInstr& in) {
  const Tracked& src = x.slot(in.ints[0]);
  const std::int64_t operand = in.ints[1];
  const std::int64_t edge_base = in.ints[2];
  const std::int64_t vertex_base = in.ints[3];
  const std::int64_t n = in.ints[4];
  const double radius = in.reals[0];
  const std::size_t nvar = static_cast<std::size_t>(in.reals[1]);
  BRepFilletAPI_MakeFillet mk(src.shape, ChFi3d_Rational);
  for (std::int64_t i = 0; i < n; ++i) {
    TopoDS_Edge e = x.edge(operand, in.ints[5 + i]);
    if (nvar > 0) {
      TColgp_Array1OfPnt2d ur(1, static_cast<int>(nvar));
      for (std::size_t k = 0; k < nvar; ++k)
        ur.SetValue(static_cast<int>(k + 1), gp_Pnt2d(in.reals[2 + 2 * k], in.reals[3 + 2 * k]));
      mk.Add(ur, e);
    } else {
      mk.Add(radius, e);
    }
  }
  mk.Build();
  if (!mk.IsDone()) fail("the fillet could not be built (radius too large, or the edges do not allow it)");
  Tracked out;
  out.shape = mk.Shape();
  propagate(src, mk, out);
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
  TopExp::MapShapes(out.shape, TopAbs_FACE, faces);
  for (std::size_t i = 0; i < b.edges.size(); ++i) {
    auto it = overrides.find(static_cast<std::int64_t>(i));
    const std::uint32_t l = it != overrides.end()
                                ? it->second
                                : static_cast<std::uint32_t>(edge_base + static_cast<std::int64_t>(i));
    for (const TopoDS_Shape& g : mk.Generated(b.edges[i])) {
      if (g.ShapeType() == TopAbs_FACE && faces.Contains(g)) add_origins(out.faces, g, label(l));
    }
  }
  for (std::size_t i = 0; i < b.vertices.size(); ++i) {
    for (const TopoDS_Shape& g : mk.Generated(b.vertices[i])) {
      if (g.ShapeType() == TopAbs_FACE && faces.Contains(g))
        add_origins(out.faces, g, label(static_cast<std::uint32_t>(vertex_base + static_cast<std::int64_t>(i))));
    }
  }
  require_labels(out, "the fillet");
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
    if (faces.Contains(f)) {
      add_origins(out.faces, f, origins_of(src.faces, f));
      out.kept.Add(f, 1);
    }
    for (const TopoDS_Shape& m : mk.Modified(f)) {
      if (faces.Contains(m)) add_origins(out.faces, m, origins_of(src.faces, f));
    }
    if (open.Contains(f)) continue;
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
  propagate(src, mk, out);
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
  Tracked out;
  out.shape = sp.Shape();
  out.sheet = src.sheet;
  propagate(src, sp, out);
  // Split faces: tagged by the side of the body they bound (opposite their outward normal).
  TopTools_IndexedMapOfShape faces;
  TopExp::MapShapes(out.shape, TopAbs_FACE, faces);
  for (const TopoDS_Shape& g : sp.Modified(mf.Face())) {
    for (TopExp_Explorer ex(out.shape, TopAbs_FACE); ex.More(); ex.Next()) {
      if (!ex.Current().IsSame(g)) continue;
      BRepAdaptor_Surface s(TopoDS::Face(ex.Current()));
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
  const double d = in.reals[0];
  BRepOffset_MakeOffset mk;
  mk.Initialize(src.shape, 0.0, 1e-6, BRepOffset_Skin, Standard_True, Standard_False,
                GeomAbs_Intersection, Standard_False, Standard_False);
  for (std::int64_t i = 0; i < n; ++i) mk.SetOffsetOnFace(x.face(operand, in.ints[3 + i]), d);
  mk.MakeOffsetShape();
  if (!mk.IsDone() || mk.Shape().IsNull()) fail("the faces could not be offset");
  Tracked out;
  out.shape = mk.Shape();
  TopTools_IndexedMapOfShape faces;
  TopExp::MapShapes(out.shape, TopAbs_FACE, faces);
  const Body& b = x.operand(operand);
  const BRepAlgo_Image& image = mk.OffsetFacesFromShapes();
  for (std::size_t i = 0; i < b.faces.size(); ++i) {
    const Origins& o = origins_of(src.faces, b.faces[i]);
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
      TopoDS_Shape m = sew.IsModified(f) ? sew.Modified(f) : f;
      for (TopExp_Explorer ex(m, TopAbs_FACE); ex.More(); ex.Next())
        if (faces.Contains(ex.Current())) add_origins(out.faces, ex.Current(), t.faces(k));
    }
    for (int k = 1; k <= t.edges.Extent(); ++k) {
      const TopoDS_Shape& e = t.edges.FindKey(k);
      TopoDS_Shape m = sew.IsModified(e) ? sew.Modified(e) : e;
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
