// K1 display tessellation: a registry body as triangles per summary face, a polyline per
// summary edge and a point per summary vertex, in the body's summary order (the same indices
// the Rust topology and the design layer's names use).
//
// BRepMesh meshes the whole shape at once, so neighbouring faces share their boundary
// discretization: the face meshes meet without cracks, and each edge polyline runs through the
// same points as the faces on either side. Nodes are shared within a face, so a renderer that
// averages normals over shared vertices shades each face smoothly and keeps creases at face
// boundaries.

#include "runmat-geometry-io/src/occt/kernel_ffi.rs.h"
#include "exact_kernel.hxx"

#include <BRepAdaptor_Curve.hxx>
#include <BRepLib_ToolTriangulatedShape.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepTools.hxx>
#include <BRep_Tool.hxx>
#include <GCPnts_TangentialDeflection.hxx>
#include <IMeshTools_Parameters.hxx>
#include <Poly_Polygon3D.hxx>
#include <Poly_PolygonOnTriangulation.hxx>
#include <Poly_Triangulation.hxx>
#include <Standard_Failure.hxx>
#include <TColStd_Array1OfInteger.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopLoc_Location.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopoDS.hxx>

namespace runmat_geometry_io {
namespace occt_kernel {

namespace {

void push_point(rust::Vec<double>& out, const gp_Pnt& p) {
  out.push_back(p.X());
  out.push_back(p.Y());
  out.push_back(p.Z());
}

KFaceMesh face_mesh(const TopoDS_Face& face) {
  KFaceMesh out;
  TopLoc_Location location;
  Handle(Poly_Triangulation) t = BRep_Tool::Triangulation(face, location);
  if (t.IsNull()) return out;
  const gp_Trsf& to_world = location.Transformation();
  for (Standard_Integer i = 1; i <= t->NbNodes(); ++i) {
    push_point(out.positions, t->Node(i).Transformed(to_world));
  }
  // The surface's own normal at each node (smooth shading on the true surface, not averaged
  // over triangles). The Rust side orients them by the triangles' winding.
  if (!t->HasNormals()) BRepLib_ToolTriangulatedShape::ComputeNormals(face, t);
  if (t->HasNormals()) {
    for (Standard_Integer i = 1; i <= t->NbNodes(); ++i) {
      const gp_Dir n = t->Normal(i).Transformed(to_world);
      out.normals.push_back(n.X());
      out.normals.push_back(n.Y());
      out.normals.push_back(n.Z());
    }
  }
  // Triangles wind counter-clockwise seen from outside: a reversed face flips them.
  const bool reversed = face.Orientation() == TopAbs_REVERSED;
  for (Standard_Integer i = 1; i <= t->NbTriangles(); ++i) {
    Standard_Integer a = 0, b = 0, c = 0;
    t->Triangle(i).Get(a, b, c);
    if (reversed) std::swap(b, c);
    out.triangles.push_back(static_cast<std::uint32_t>(a - 1));
    out.triangles.push_back(static_cast<std::uint32_t>(b - 1));
    out.triangles.push_back(static_cast<std::uint32_t>(c - 1));
  }
  return out;
}

// The edge through the triangulation of a face it bounds (so it runs through the face mesh's
// own boundary points); its curve discretized at the same deflection when no face has one.
KPolyline edge_polyline(const TopoDS_Edge& edge, const TopTools_ListOfShape& faces, double chord,
                        double angle) {
  KPolyline out;
  for (const TopoDS_Shape& f : faces) {
    TopLoc_Location location;
    Handle(Poly_Triangulation) t = BRep_Tool::Triangulation(TopoDS::Face(f), location);
    if (t.IsNull()) continue;
    Handle(Poly_PolygonOnTriangulation) p = BRep_Tool::PolygonOnTriangulation(edge, t, location);
    if (p.IsNull()) continue;
    const gp_Trsf& to_world = location.Transformation();
    const TColStd_Array1OfInteger& nodes = p->Nodes();
    for (Standard_Integer i = nodes.Lower(); i <= nodes.Upper(); ++i) {
      push_point(out.points, t->Node(nodes(i)).Transformed(to_world));
    }
    return out;
  }
  TopLoc_Location location;
  Handle(Poly_Polygon3D) p3 = BRep_Tool::Polygon3D(edge, location);
  if (!p3.IsNull()) {
    const gp_Trsf& to_world = location.Transformation();
    for (Standard_Integer i = 1; i <= p3->NbNodes(); ++i) {
      push_point(out.points, p3->Nodes()(i).Transformed(to_world));
    }
    return out;
  }
  BRepAdaptor_Curve curve(edge);
  GCPnts_TangentialDeflection d(curve, angle, chord);
  for (Standard_Integer i = 1; i <= d.NbPoints(); ++i) push_point(out.points, d.Value(i));
  return out;
}

// Whether a face of the body already holds a triangulation finer than `chord`. BRepMesh keeps a
// finer mesh, so a coarse display level asked for after the full display would get the full
// one; such a body is cleaned and meshed again at the coarser chord.
bool finer_mesh_than(const TopoDS_Shape& shape, double chord) {
  for (TopExp_Explorer ex(shape, TopAbs_FACE); ex.More(); ex.Next()) {
    TopLoc_Location loc;
    const Handle(Poly_Triangulation) tri = BRep_Tool::Triangulation(TopoDS::Face(ex.Current()), loc);
    if (!tri.IsNull() && tri->Deflection() < chord * 0.999) return true;
  }
  return false;
}

}  // namespace

KDisplay kernel_tessellate(std::uint64_t id, double chord, double angle) {
  const auto guard = kernel_lock();
  try {
    auto body = registry_get(id);
    IMeshTools_Parameters parameters;
    parameters.Deflection = chord;
    parameters.Angle = angle;
    parameters.DeflectionInterior = chord;
    parameters.AngleInterior = angle;
    parameters.Relative = Standard_False;
    parameters.InParallel = Standard_False;
    parameters.MinSize = chord * 0.1;
    parameters.ControlSurfaceDeflection = Standard_True;
    if (finer_mesh_than(body->shape, chord)) BRepTools::Clean(body->shape);
    BRepMesh_IncrementalMesh mesh(body->shape, parameters);
    if (!mesh.IsDone()) fail("OCCT display tessellation did not complete");

    KDisplay out;
    for (const TopoDS_Face& f : body->faces) out.faces.push_back(face_mesh(f));
    TopTools_IndexedDataMapOfShapeListOfShape edge_faces;
    TopExp::MapShapesAndUniqueAncestors(body->shape, TopAbs_EDGE, TopAbs_FACE, edge_faces);
    for (const TopoDS_Edge& e : body->edges) {
      const TopTools_ListOfShape* faces = edge_faces.Seek(e);
      static const TopTools_ListOfShape kNone;
      out.edges.push_back(edge_polyline(e, faces ? *faces : kNone, chord, angle));
    }
    for (const TopoDS_Vertex& v : body->vertices) push_point(out.vertices, BRep_Tool::Pnt(v));
    return out;
  } catch (const Standard_Failure& e) {
    fail(std::string("OCCT raised ") + e.GetMessageString());
  }
}

}  // namespace occt_kernel
}  // namespace runmat_geometry_io
