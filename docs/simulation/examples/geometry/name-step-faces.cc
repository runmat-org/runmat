// Writes authored names into an example STEP file: the part (product and solid) name and the
// names of selected faces, edges and vertices, as `ADVANCED_FACE('<name>', ...)`,
// `EDGE_CURVE('<name>', ...)` and `VERTEX_POINT('<name>', ...)`, so studies select by `name`
// (DEF-276). Faces are numbered in the kernel's face order in the file; edges and vertices in
// the kernel's unique edge and vertex order. The listing mode prints those numbers with each
// entity's location, so pick entities by location. The geometry is copied unchanged.
//
// Build: clang++ -std=c++17 -I/opt/homebrew/opt/opencascade/include/opencascade name-step-faces.cc \
//   -L/opt/homebrew/opt/opencascade/lib -Wl,-rpath,/opt/homebrew/opt/opencascade/lib \
//   -lTKDESTEP -lTKXSBase -lTKBRep -lTKG3d -lTKMath -lTKernel -lTKXCAF -lTKLCAF -lTKCAF -lTKCDF \
//   -lTKDE -lTKTopAlgo -o name-step-faces
// Run:   ./name-step-faces in.step out.step "Beam" 1="Fixed end" 2="Loaded end"
//        ./name-step-faces in.step out.step "Plate" 1="Plate" e1="South edge" v2="Base"
//        ./name-step-faces in.step  (lists faces: ordinal, centroid, normal, area; edges:
//                                    e<ordinal>, midpoint, length; vertices: v<ordinal>, point)
#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRep_Tool.hxx>
#include <BRepGProp.hxx>
#include <BRepLProp_SLProps.hxx>
#include <DESTEP_Parameters.hxx>
#include <GProp_GProps.hxx>
#include <STEPCAFControl_Reader.hxx>
#include <STEPCAFControl_Writer.hxx>
#include <TDataStd_Name.hxx>
#include <TDocStd_Document.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS.hxx>
#include <XCAFApp_Application.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>

#include <cstdio>
#include <iostream>
#include <map>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: name-step-faces in.step [out.step part-name N=name...]\n";
    return 1;
  }
  Handle(TDocStd_Document) doc;
  XCAFApp_Application::GetApplication()->NewDocument("MDTV-XCAF", doc);
  STEPCAFControl_Reader reader;
  reader.SetNameMode(true);
  if (reader.ReadFile(argv[1]) != IFSelect_RetDone || !reader.Transfer(doc)) {
    std::cerr << "cannot read " << argv[1] << '\n';
    return 2;
  }
  Handle(XCAFDoc_ShapeTool) shapes = XCAFDoc_DocumentTool::ShapeTool(doc->Main());
  TDF_LabelSequence free_shapes;
  shapes->GetFreeShapes(free_shapes);
  if (free_shapes.Length() != 1 || shapes->IsAssembly(free_shapes.Value(1))) {
    std::cerr << "expected one part\n";
    return 2;
  }
  const TDF_Label part = free_shapes.Value(1);
  const TopoDS_Shape shape = shapes->GetShape(part);
  std::vector<TopoDS_Face> faces;
  for (TopExp_Explorer explorer(shape, TopAbs_FACE); explorer.More(); explorer.Next()) {
    faces.push_back(TopoDS::Face(explorer.Current()));
  }
  TopTools_IndexedMapOfShape edges;
  TopTools_IndexedMapOfShape vertices;
  TopExp::MapShapes(shape, TopAbs_EDGE, edges);
  TopExp::MapShapes(shape, TopAbs_VERTEX, vertices);
  if (argc == 2) {
    for (std::size_t index = 0; index < faces.size(); ++index) {
      GProp_GProps props;
      BRepGProp::SurfaceProperties(faces[index], props);
      const gp_Pnt c = props.CentreOfMass();
      BRepAdaptor_Surface surface(faces[index]);
      const double u = 0.5 * (surface.FirstUParameter() + surface.LastUParameter());
      const double v = 0.5 * (surface.FirstVParameter() + surface.LastVParameter());
      BRepLProp_SLProps local(surface, u, v, 1, 1e-7);
      gp_Dir n(0, 0, 1);
      if (local.IsNormalDefined()) {
        n = local.Normal();
        if (faces[index].Orientation() == TopAbs_REVERSED) n.Reverse();
      }
      std::printf("%3zu centroid (%9.3f %9.3f %9.3f) normal (%6.3f %6.3f %6.3f) area %10.3f\n",
                  index + 1, c.X(), c.Y(), c.Z(), n.X(), n.Y(), n.Z(), props.Mass());
    }
    for (int index = 1; index <= edges.Extent(); ++index) {
      const TopoDS_Edge edge = TopoDS::Edge(edges(index));
      BRepAdaptor_Curve curve(edge);
      const gp_Pnt m = curve.Value(0.5 * (curve.FirstParameter() + curve.LastParameter()));
      GProp_GProps props;
      BRepGProp::LinearProperties(edge, props);
      std::printf("e%-2d midpoint (%9.3f %9.3f %9.3f) length %10.3f\n", index, m.X(), m.Y(),
                  m.Z(), props.Mass());
    }
    for (int index = 1; index <= vertices.Extent(); ++index) {
      const gp_Pnt p = BRep_Tool::Pnt(TopoDS::Vertex(vertices(index)));
      std::printf("v%-2d point (%9.3f %9.3f %9.3f)\n", index, p.X(), p.Y(), p.Z());
    }
    return 0;
  }
  if (argc < 4) {
    std::cerr << "naming needs out.step and a part name\n";
    return 1;
  }
  TDataStd_Name::Set(part, argv[3]);
  // A part whose shape is one solid is named by its product; solids inside a compound are
  // named individually.
  for (TopExp_Explorer solids(shape, TopAbs_SOLID); solids.More(); solids.Next()) {
    if (solids.Current().IsSame(shape)) continue;
    TDF_Label label = shapes->AddSubShape(part, solids.Current());
    if (!label.IsNull()) TDataStd_Name::Set(label, argv[3]);
  }
  for (int arg = 4; arg < argc; ++arg) {
    const std::string spec = argv[arg];
    const auto equals = spec.find('=');
    const char kind = spec.empty() ? '\0' : spec[0];
    const std::size_t start = (kind == 'e' || kind == 'v') ? 1 : 0;
    const std::size_t count = kind == 'e' ? static_cast<std::size_t>(edges.Extent())
                              : kind == 'v' ? static_cast<std::size_t>(vertices.Extent())
                                            : faces.size();
    const std::size_t ordinal =
        equals == std::string::npos ? 0 : std::stoul(spec.substr(start, equals - start));
    if (ordinal == 0 || ordinal > count) {
      std::cerr << "bad entity spec " << spec << '\n';
      return 1;
    }
    const TopoDS_Shape entity = kind == 'e'   ? edges(static_cast<int>(ordinal))
                                : kind == 'v' ? vertices(static_cast<int>(ordinal))
                                              : TopoDS_Shape(faces[ordinal - 1]);
    TDF_Label label = shapes->AddSubShape(part, entity);
    if (label.IsNull()) {
      std::cerr << "cannot label " << spec.substr(0, equals) << '\n';
      return 2;
    }
    TDataStd_Name::Set(label, spec.substr(equals + 1).c_str());
  }
  DESTEP_Parameters params;
  params.InitFromStatic();
  params.WriteSubshapeNames = true;
  STEPCAFControl_Writer writer;
  writer.SetNameMode(true);
  writer.SetColorMode(true);
  if (!writer.Transfer(doc, params, STEPControl_AsIs) || writer.Write(argv[2]) != IFSelect_RetDone) {
    std::cerr << "cannot write " << argv[2] << '\n';
    return 2;
  }
  return 0;
}
