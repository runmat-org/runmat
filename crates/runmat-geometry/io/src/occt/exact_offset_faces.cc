// Offset faces (push-pull) for K1: direct.move_faces with an offset, or a translation of planar
// faces along their normals.
//
// The picked faces move along their normals and their neighbours extend or trim to meet them
// (BRepOffset_MakeOffset, faces left at distance 0 intersected with the moved ones). A neighbour
// tangent to a moved face can't be intersected with it: a round or fillet along the face's edge
// (a bent sheet's bend, a blended top edge) meets the face along a line, not across it. That is
// the usual case on parts read from STEP files. For those the round is lifted first
// (BRepAlgoAPI_Defeaturing extends the face and the round's other neighbour to a sharp edge), the
// face is moved, and the sharp edge is rounded again at the round's radius, so the round moves
// with the face, as it does when the part is rebuilt from its features. A round on the inside of
// a bent sheet can't be lifted (the sharp corner would cut through the sheet's outside): then a
// flat face is pushed or pulled as a sweep along its normal, fused to or cut from the body, and
// steps up from the rounds along it.
//
// Every step runs in millimetres (MmFrame): OCCT's offset and blend tolerances are tuned for
// millimetre models. When a step fails, the error says which one and why, in plain words.

#include "runmat-geometry-io/src/occt/kernel_ffi.rs.h"
#include "exact_kernel.hxx"

#include <BRepAdaptor_Surface.hxx>
#include <BRepAlgoAPI_Defeaturing.hxx>
#include <BRepAlgo_Image.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepFilletAPI_MakeFillet.hxx>
#include <BRepGProp_Face.hxx>
#include <BRepOffset_Error.hxx>
#include <BRepOffset_MakeOffset.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <BRep_Tool.hxx>
#include <Geom2d_Curve.hxx>
#include <GeomAbs_SurfaceType.hxx>
#include <Standard_Failure.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopoDS.hxx>
#include <gp_Cylinder.hxx>
#include <gp_Dir.hxx>
#include <gp_Pln.hxx>
#include <gp_Pnt2d.hxx>
#include <gp_Vec.hxx>

#include <cmath>
#include <memory>
#include <optional>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

namespace runmat_geometry_io {
namespace occt_kernel {

namespace {

// Two faces are tangent along an edge when their normals there are within about 1.1 degrees.
constexpr double kTangentCos = 0.9998;

std::string offset_error_words(BRepOffset_Error e) {
  switch (e) {
    case BRepOffset_BadNormalsOnGeometry:
      return "a face's surface has no defined normal at some point (a degenerate or pinched surface)";
    case BRepOffset_C0Geometry:
      return "a face's surface has a crease inside it (a surface with only positional continuity)";
    case BRepOffset_NullOffset:
      return "the distance is zero";
    case BRepOffset_NotConnectedShell:
      return "the body's faces are not joined into one closed shell";
    case BRepOffset_CannotTrimEdges:
      return "the neighbouring faces could not be trimmed to meet the moved faces";
    case BRepOffset_CannotFuseVertices:
      return "corners of the moved faces could not be joined";
    case BRepOffset_CannotExtentEdge:
      return "the neighbouring faces could not be extended to meet the moved faces";
    case BRepOffset_MixedConnectivity:
      return "an edge of a moved face is partly sharp and partly tangent";
    case BRepOffset_UserBreak:
      return "the operation was interrupted";
    default:
      return "the neighbouring faces could not be extended or trimmed to meet the moved faces";
  }
}

bool valid(const TopoDS_Shape& s) {
  if (s.IsNull()) return false;
  BRepCheck_Analyzer check(s, Standard_True);
  return check.IsValid() == Standard_True;
}

// The unit normal of `f` at the middle of its edge `e` (the face's orientation is ignored).
bool normal_at_edge(const TopoDS_Face& f, const TopoDS_Edge& e, gp_Vec& n) {
  Standard_Real a = 0, b = 0;
  Handle(Geom2d_Curve) c = BRep_Tool::CurveOnSurface(e, f, a, b);
  if (c.IsNull()) return false;
  const gp_Pnt2d uv = c->Value(0.5 * (a + b));
  BRepGProp_Face gf(f);
  gp_Pnt p;
  gf.Normal(uv.X(), uv.Y(), p, n);
  if (n.Magnitude() < 1e-12) return false;
  n.Normalize();
  return true;
}

bool tangent_along(const TopoDS_Face& f, const TopoDS_Face& g, const TopoDS_Edge& e) {
  gp_Vec a, b;
  if (!normal_at_edge(f, e, a) || !normal_at_edge(g, e, b)) return false;
  return std::abs(a.Dot(b)) > kTangentCos;
}

// A round along an edge of a moved face: the cylinder tangent to the moved face on one side and
// to `far` on the other.
struct Round {
  TopoDS_Face face;
  TopoDS_Face moved;  // the picked face it is tangent to
  TopoDS_Face far;    // the face across the round
  double radius = 0;
};

// The rounds tangent to the picked faces. A tangent neighbour that is not a cylinder joining
// two faces is reported: the move can't carry it.
std::vector<Round> rounds_along(const TopoDS_Shape& shape, const std::vector<TopoDS_Face>& picked) {
  TopTools_IndexedDataMapOfShapeListOfShape ef;
  TopExp::MapShapesAndUniqueAncestors(shape, TopAbs_EDGE, TopAbs_FACE, ef);
  TopTools_IndexedMapOfShape chosen;
  for (const TopoDS_Face& f : picked) chosen.Add(f);
  TopTools_IndexedMapOfShape seen;
  std::vector<Round> out;
  auto across = [&](const TopoDS_Shape& e, const TopoDS_Shape& f) -> TopoDS_Face {
    const TopTools_ListOfShape* fs = ef.Seek(e);
    if (fs == nullptr) return TopoDS_Face();
    for (const TopoDS_Shape& g : *fs)
      if (!g.IsSame(f)) return TopoDS::Face(g);
    return TopoDS_Face();
  };
  for (const TopoDS_Face& p : picked) {
    for (TopExp_Explorer ex(p, TopAbs_EDGE); ex.More(); ex.Next()) {
      const TopoDS_Edge e = TopoDS::Edge(ex.Current());
      if (BRep_Tool::Degenerated(e)) continue;
      const TopoDS_Face q = across(e, p);
      if (q.IsNull() || chosen.Contains(q) || seen.Contains(q)) continue;
      if (!tangent_along(p, q, e)) continue;
      seen.Add(q);
      BRepAdaptor_Surface s(q, Standard_False);
      if (s.GetType() != GeomAbs_Cylinder)
        fail("the faces could not be offset: a face along the edge of a moved face is tangent to it "
             "and is not a round of constant radius, so it can't move with the face");
      // The round's other tangent neighbour: the face it blends into.
      TopoDS_Face far;
      for (TopExp_Explorer qx(q, TopAbs_EDGE); qx.More(); qx.Next()) {
        const TopoDS_Edge qe = TopoDS::Edge(qx.Current());
        if (BRep_Tool::Degenerated(qe)) continue;
        const TopoDS_Face g = across(qe, q);
        if (g.IsNull() || g.IsSame(p) || !tangent_along(q, g, qe)) continue;
        if (chosen.Contains(g))
          fail("the faces could not be offset: a round joins two of the moved faces");
        if (!far.IsNull() && !far.IsSame(g))
          fail("the faces could not be offset: a round along a moved face blends into more than one "
               "face, so it can't be rebuilt after the move");
        far = g;
      }
      if (far.IsNull())
        fail("the faces could not be offset: a round along a moved face doesn't blend into another "
             "face");
      out.push_back({q, p, far, s.Cylinder().Radius()});
    }
  }
  return out;
}

// One run of OCCT's per-face offset: `faces` move by `dist`, every other face stays and is
// extended or trimmed. Null with `error` set when it fails or makes an invalid solid.
std::unique_ptr<BRepOffset_MakeOffset> run_offset(const TopoDS_Shape& shape,
                                                  const std::vector<TopoDS_Face>& faces,
                                                  const std::vector<double>& dist,
                                                  std::string& error) {
  auto mk = std::make_unique<BRepOffset_MakeOffset>();
  try {
    // Tolerance 1e-4 mm in the millimetre frame (the old metre-frame 1e-6 m was 1e-3 mm, looser
    // than OCCT's confusion and than many imported edges).
    mk->Initialize(shape, 0.0, 1e-4, BRepOffset_Skin, Standard_True, Standard_False, GeomAbs_Intersection,
                   Standard_False, Standard_False);
    for (std::size_t i = 0; i < faces.size(); ++i) mk->SetOffsetOnFace(faces[i], dist[i]);
    mk->MakeOffsetShape();
  } catch (const Standard_Failure& e) {
    error = std::string("OCCT stopped while moving the faces (") + e.GetMessageString() + ")";
    return nullptr;
  }
  if (!mk->IsDone() || mk->Shape().IsNull()) {
    error = offset_error_words(mk->Error());
    return nullptr;
  }
  if (!valid(mk->Shape())) {
    error = "the moved faces and their neighbours don't close into a valid solid";
    return nullptr;
  }
  return mk;
}

// The faces of `mk`'s result that `s` became (itself when the offset left it as it was).
std::vector<TopoDS_Face> images_of(const BRepOffset_MakeOffset& mk, const TopoDS_Shape& s,
                                   const TopTools_IndexedMapOfShape& present) {
  std::vector<TopoDS_Face> out;
  const BRepAlgo_Image& image = mk.OffsetFacesFromShapes();
  if (image.HasImage(s)) {
    TopTools_ListOfShape imgs;
    image.LastImage(s, imgs);
    for (const TopoDS_Shape& g : imgs)
      for (TopExp_Explorer ex(g, TopAbs_FACE); ex.More(); ex.Next())
        if (present.Contains(ex.Current())) out.push_back(TopoDS::Face(ex.Current()));
  } else if (present.Contains(s)) {
    out.push_back(TopoDS::Face(s));
  }
  return out;
}

// `in`'s face origins carried through `mk`; the faces in `moved` take `moved_origins` instead.
Tracked carry_offset(const Tracked& in, const BRepOffset_MakeOffset& mk,
                     const std::vector<TopoDS_Face>& moved, const std::vector<Origins>& moved_origins) {
  Tracked out;
  out.shape = mk.Shape();
  TopTools_IndexedMapOfShape present, before;
  TopExp::MapShapes(out.shape, TopAbs_FACE, present);
  TopExp::MapShapes(in.shape, TopAbs_FACE, before);
  for (int i = 1; i <= before.Extent(); ++i) {
    Origins o = origins_of(in.faces, before(i));
    for (std::size_t k = 0; k < moved.size(); ++k)
      if (moved[k].IsSame(before(i))) o = moved_origins[k];
    if (o.empty()) continue;
    for (const TopoDS_Face& g : images_of(mk, before(i), present)) add_origins(out.faces, g, o);
  }
  return out;
}

// The one face `f` became in a defeaturing (itself when untouched).
TopoDS_Face defeatured_image(BRepAlgoAPI_Defeaturing& df, const TopoDS_Face& f) {
  const TopTools_ListOfShape& m = df.Modified(f);
  if (m.IsEmpty()) return f;
  // Split in two: no single face to move or to round against.
  if (m.Extent() > 1) return TopoDS_Face();
  return TopoDS::Face(m.First());
}

TopoDS_Edge shared_edge(const TopoDS_Face& a, const TopoDS_Face& b) {
  TopTools_IndexedMapOfShape ea;
  TopExp::MapShapes(a, TopAbs_EDGE, ea);
  for (TopExp_Explorer ex(b, TopAbs_EDGE); ex.More(); ex.Next())
    if (ea.Contains(ex.Current())) return TopoDS::Edge(ex.Current());
  return TopoDS_Edge();
}

// The move with the rounds along the moved faces lifted, then rebuilt at their radii.
// Empty with `error` set when a step fails.
std::optional<Tracked> offset_with_rounds(const Tracked& work, const std::vector<TopoDS_Face>& faces,
                                          const std::vector<double>& dist,
                                          const std::vector<Origins>& moved_origins,
                                          const std::vector<Round>& rounds, std::string& error) {
  BRepAlgoAPI_Defeaturing df;
  df.SetShape(work.shape);
  for (const Round& r : rounds) df.AddFaceToRemove(r.face);
  df.SetRunParallel(Standard_False);
  df.SetToFillHistory(Standard_True);
  df.Build();
  if (!df.IsDone() || df.HasErrors()) {
    error = "the rounds along them could not be lifted to move with them";
    return std::nullopt;
  }
  for (const Round& r : rounds)
    if (!df.IsDeleted(r.face)) {
      error = "a round along them could not be lifted to move with them (on a bent sheet, the "
              "inside of a bend has no room for a sharp corner)";
      return std::nullopt;
    }
  Tracked sharp;
  sharp.shape = df.Shape();
  propagate_history(work, df.History(), sharp);

  std::vector<TopoDS_Face> sharp_faces;
  for (const TopoDS_Face& f : faces) sharp_faces.push_back(defeatured_image(df, f));
  for (const TopoDS_Face& f : sharp_faces)
    if (f.IsNull()) {
      error = "lifting the rounds split a moved face";
      return std::nullopt;
    }
  auto mk = run_offset(sharp.shape, sharp_faces, dist, error);
  if (!mk) return std::nullopt;
  Tracked moved = carry_offset(sharp, *mk, sharp_faces, moved_origins);

  // Each round again, on the sharp edge between the moved face and the face it blended into.
  TopTools_IndexedMapOfShape present;
  TopExp::MapShapes(moved.shape, TopAbs_FACE, present);
  BRepFilletAPI_MakeFillet mf(moved.shape, ChFi3d_Rational);
  std::vector<std::pair<TopoDS_Edge, const Round*>> edges;
  for (const Round& r : rounds) {
    const auto m = images_of(*mk, defeatured_image(df, r.moved), present);
    const auto g = images_of(*mk, defeatured_image(df, r.far), present);
    TopoDS_Edge e;
    for (const TopoDS_Face& a : m)
      for (const TopoDS_Face& b : g)
        if (e.IsNull()) e = shared_edge(a, b);
    if (e.IsNull()) {
      error = "after the move, a moved face no longer meets the face its round blended into, so the "
              "round can't be rebuilt";
      return std::nullopt;
    }
    mf.Add(r.radius, e);
    edges.emplace_back(e, &r);
  }
  try {
    mf.Build();
  } catch (const Standard_Failure&) {
  }
  if (!mf.IsDone() || !valid(mf.Shape())) {
    error = "the faces moved, but the rounds along them could not be rebuilt at their radii (a round "
            "may be larger than the faces next to it allow)";
    return std::nullopt;
  }
  Tracked out;
  out.shape = mf.Shape();
  propagate(moved, mf, out);
  TopTools_IndexedMapOfShape after;
  TopExp::MapShapes(out.shape, TopAbs_FACE, after);
  for (const auto& [e, r] : edges) {
    const Origins& o = origins_of(work.faces, r->face);
    for (const TopoDS_Shape& g : mf.Generated(e))
      if (g.ShapeType() == TopAbs_FACE && after.Contains(g)) add_origins(out.faces, g, o);
  }
  return out;
}

// Push-pull, when the rounds can't be lifted: the picked planar face swept along its normal by
// the distance, the sweep fused to the body (outward) or cut from it (inward). The sweep's sides
// rise from the face's edges, so a round along the face stays where it is and the moved face
// steps up from it by the distance. Faces made by the sweep take the moved face's origin (its
// far end) or the neighbour across the edge they rise from (its sides).
std::optional<Tracked> push_pull(const Tracked& work, const std::vector<TopoDS_Face>& faces,
                                 const std::vector<double>& dist,
                                 const std::vector<Origins>& moved_origins, std::string& error) {
  if (faces.size() != 1) {
    error = "several faces can't step past their rounds together";
    return std::nullopt;
  }
  const TopoDS_Face& p = faces[0];
  BRepAdaptor_Surface surface(p, Standard_False);
  if (surface.GetType() != GeomAbs_Plane) {
    error = "the face is curved, and only a flat face can step past its rounds";
    return std::nullopt;
  }
  gp_Dir n = surface.Plane().Axis().Direction();
  if (p.Orientation() == TopAbs_REVERSED) n.Reverse();
  BRepPrimAPI_MakePrism prism(p, gp_Vec(n) * dist[0], Standard_False);
  prism.Build();
  if (!prism.IsDone()) {
    error = "the face could not be swept along its normal";
    return std::nullopt;
  }
  TopTools_IndexedDataMapOfShapeListOfShape ef;
  TopExp::MapShapesAndUniqueAncestors(work.shape, TopAbs_EDGE, TopAbs_FACE, ef);
  Tracked tool;
  tool.shape = prism.Shape();
  TopTools_IndexedMapOfShape tool_faces;
  TopExp::MapShapes(tool.shape, TopAbs_FACE, tool_faces);
  for (int i = 1; i <= tool_faces.Extent(); ++i) add_origins(tool.faces, tool_faces(i), moved_origins[0]);
  for (TopExp_Explorer ex(p, TopAbs_EDGE); ex.More(); ex.Next()) {
    const TopTools_ListOfShape* fs = ef.Seek(ex.Current());
    if (fs == nullptr) continue;
    for (const TopoDS_Shape& g : *fs) {
      if (g.IsSame(p)) continue;
      const Origins& o = origins_of(work.faces, g);
      if (o.empty()) continue;
      for (const TopoDS_Shape& side : prism.Generated(ex.Current()))
        if (side.ShapeType() == TopAbs_FACE) {
          Origins& slot = tool.faces.ChangeFromKey(side);
          slot = o;
        }
    }
  }
  Tracked out;
  try {
    out = boolean_op(dist[0] > 0 ? 0 : 1, {&work}, {&tool});
  } catch (const std::exception& e) {
    error = std::string("the face's sweep could not be joined to the body (") + e.what() + ")";
    return std::nullopt;
  } catch (const Standard_Failure& e) {
    error = std::string("the face's sweep could not be joined to the body (") + e.GetMessageString() + ")";
    return std::nullopt;
  }
  if (!valid(out.shape)) {
    error = "the face's sweep and the body don't make a valid solid";
    return std::nullopt;
  }
  return out;
}

}  // namespace

Tracked offset_faces(const Tracked& src, const std::vector<TopoDS_Face>& faces,
                     const std::vector<double>& dist, const std::vector<Origins>& moved_origins) {
  const MmFrame mm(src, true);
  std::vector<TopoDS_Face> work_faces;
  std::vector<double> work_dist;
  for (std::size_t i = 0; i < faces.size(); ++i) {
    work_faces.push_back(TopoDS::Face(mm.image(faces[i])));
    work_dist.push_back(mm.len(dist[i]));
  }
  // A round tangent to a moved face can't be intersected with it: OCCT either fails or returns
  // a closed but wrong solid (on CTC-03, twice the volume). Rounds take the lift-and-rebuild way.
  const std::vector<Round> rounds = rounds_along(mm.work.shape, work_faces);
  std::string error;
  if (!rounds.empty()) {
    if (auto out = offset_with_rounds(mm.work, work_faces, work_dist, moved_origins, rounds, error))
      return mm.back(*out);
    std::string stepped;
    if (auto out = push_pull(mm.work, work_faces, work_dist, moved_origins, stepped)) return mm.back(*out);
    fail("the faces could not be offset: " + error + "; moving the face alone failed too: " + stepped);
  }
  auto mk = run_offset(mm.work.shape, work_faces, work_dist, error);
  if (!mk) fail("the faces could not be offset: " + error);
  return mm.back(carry_offset(mm.work, *mk, work_faces, moved_origins));
}

}  // namespace occt_kernel
}  // namespace runmat_geometry_io
