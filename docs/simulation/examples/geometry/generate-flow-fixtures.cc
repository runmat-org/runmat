// Exact single-body flow validation fixtures, authored for RunMat. Dimensions are metres.
// Build: clang++ -std=c++17 -I/opt/homebrew/opt/opencascade/include/opencascade
//   generate-flow-fixtures.cc -L/opt/homebrew/opt/opencascade/lib
//   -Wl,-rpath,/opt/homebrew/opt/opencascade/lib -lTKPrim -lTKDESTEP -lTKXSBase
//   -lTKTopAlgo -lTKBRep -lTKG3d -lTKMath -lTKernel -lTKXCAF -lTKLCAF -lTKCAF -lTKCDF -lTKDE
//   -o generate-flow-fixtures
// Run: ./generate-flow-fixtures <output-directory>/
//
// Each fixture is one named part whose boundary faces carry authored names, so the studies
// select them with `select name`. Box faces are named in the Open CASCADE BRepPrimAPI_MakeBox
// order: x = 0, x = max, y = 0, y = max, z = 0, z = max. Cylinder faces are named in the
// BRepPrimAPI_MakeCylinder order: lateral surface, end at the axis tip, end at the axis origin.
#include <BRepCheck_Analyzer.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <DESTEP_Parameters.hxx>
#include <Interface_Static.hxx>
#include <STEPCAFControl_Writer.hxx>
#include <STEPControl_Controller.hxx>
#include <TDataStd_Name.hxx>
#include <TDocStd_Document.hxx>
#include <TopExp_Explorer.hxx>
#include <XCAFApp_Application.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>
#include <gp_Ax2.hxx>
#include <gp_Dir.hxx>
#include <gp_Pnt.hxx>
#include <iostream>
#include <string>
#include <vector>

namespace {
// Writes one named part; `face_names` name its faces in the kernel's face order.
bool write(const TopoDS_Shape& shape, const std::string& part_name,
           const std::vector<std::string>& face_names, const std::string& path) {
  if (!BRepCheck_Analyzer(shape).IsValid()) return false;
  Handle(TDocStd_Document) doc;
  XCAFApp_Application::GetApplication()->NewDocument("MDTV-XCAF", doc);
  Handle(XCAFDoc_ShapeTool) shapes = XCAFDoc_DocumentTool::ShapeTool(doc->Main());
  const TDF_Label part = shapes->AddShape(shape, Standard_False);
  TDataStd_Name::Set(part, part_name.c_str());
  std::size_t index = 0;
  for (TopExp_Explorer faces(shape, TopAbs_FACE); faces.More(); faces.Next(), ++index) {
    if (index >= face_names.size()) return false;
    const TDF_Label face = shapes->AddSubShape(part, faces.Current());
    if (face.IsNull()) return false;
    TDataStd_Name::Set(face, face_names[index].c_str());
  }
  if (index != face_names.size()) return false;
  // Shapes are built in metres: the document unit and the declared STEP unit must both be
  // metres, or the writer labels metre values as millimetres.
  XCAFDoc_DocumentTool::SetLengthUnit(doc, 1.0, UnitsMethods_LengthUnit_Meter);
  DESTEP_Parameters params;
  params.InitFromStatic();
  params.WriteSubshapeNames = true;
  params.WriteUnit = UnitsMethods_LengthUnit_Meter;
  STEPCAFControl_Writer writer;
  writer.SetNameMode(true);
  if (!writer.Transfer(doc, params, STEPControl_AsIs)) return false;
  if (writer.Write(path.c_str()) != IFSelect_RetDone) return false;
  std::cout << "wrote " << path << '\n';
  return true;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 1;
  const std::string dir = argv[1];
  // Shapes are built in metres: the session unit and the declared STEP unit must both be metres,
  // or the writer labels metre values as millimetres.
  STEPControl_Controller::Init();
  if (!Interface_Static::SetCVal("xstep.cascade.unit", "M")) return 5;
  if (!Interface_Static::SetCVal("write.step.unit", "M")) return 5;
  // Lid-driven cavity: unit square section, thin spanwise slab closed by symmetry planes.
  if (!write(BRepPrimAPI_MakeBox(1.0, 1.0, 0.05).Shape(), "Cavity",
             {"Left wall", "Right wall", "Bottom wall", "Lid", "Back", "Front"},
             dir + "cavity-slab.step"))
    return 2;
  // Plane channel: 1 m long, 0.1 m between the walls, spanwise slab closed by symmetry planes.
  if (!write(BRepPrimAPI_MakeBox(1.0, 0.1, 0.02).Shape(), "Channel",
             {"Inlet", "Outlet", "Lower wall", "Upper wall", "Back", "Front"},
             dir + "plane-channel.step"))
    return 3;
  // Long plane channel: 3 m long (L/h = 30), same section. The uniform-inlet entrance adds about
  // 0.8 h of equivalent length to the developed drop, under 3% of the drop at this length.
  if (!write(BRepPrimAPI_MakeBox(3.0, 0.1, 0.02).Shape(), "Channel",
             {"Inlet", "Outlet", "Lower wall", "Upper wall", "Back", "Front"},
             dir + "plane-channel-long.step"))
    return 6;
  // Round pipe: 0.05 m radius, 0.5 m long along +x.
  const gp_Ax2 axis(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0));
  if (!write(BRepPrimAPI_MakeCylinder(axis, 0.05, 0.5).Shape(), "Pipe", {"Wall", "Outlet", "Inlet"},
             dir + "pipe.step"))
    return 4;
  return 0;
}
