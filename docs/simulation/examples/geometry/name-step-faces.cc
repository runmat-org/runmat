// Writes authored names into an example STEP file: the part (product and solid) name and the
// names of selected faces, as `ADVANCED_FACE('<name>', ...)`, so studies select by `name`
// (DEF-276). Faces are numbered as RunMat's inspection numbers them (`face_000001` is face 1:
// the kernel's face order in the file). The geometry is copied unchanged.
//
// Build: clang++ -std=c++17 -I/opt/homebrew/opt/opencascade/include/opencascade name-step-faces.cc \
//   -L/opt/homebrew/opt/opencascade/lib -Wl,-rpath,/opt/homebrew/opt/opencascade/lib \
//   -lTKDESTEP -lTKXSBase -lTKBRep -lTKG3d -lTKMath -lTKernel -lTKXCAF -lTKLCAF -lTKCAF -lTKCDF \
//   -lTKDE -lTKTopAlgo -o name-step-faces
// Run:   ./name-step-faces in.step out.step "Beam" 1="Fixed end" 2="Loaded end"
//        ./name-step-faces in.step            (lists faces: ordinal, centroid, normal, area)
#include <BRepAdaptor_Surface.hxx>
#include <BRepGProp.hxx>
#include <BRepLProp_SLProps.hxx>
#include <DESTEP_Parameters.hxx>
#include <GProp_GProps.hxx>
#include <STEPCAFControl_Reader.hxx>
#include <STEPCAFControl_Writer.hxx>
#include <TDataStd_Name.hxx>
#include <TDocStd_Document.hxx>
#include <TopExp_Explorer.hxx>
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
    const std::size_t ordinal = std::stoul(spec.substr(0, equals));
    if (equals == std::string::npos || ordinal == 0 || ordinal > faces.size()) {
      std::cerr << "bad face spec " << spec << '\n';
      return 1;
    }
    TDF_Label label = shapes->AddSubShape(part, faces[ordinal - 1]);
    if (label.IsNull()) {
      std::cerr << "cannot label face " << ordinal << '\n';
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
