// Exact single-body flow validation fixtures, authored for RunMat. Dimensions are metres.
// Build: clang++ -std=c++17 -I/opt/homebrew/opt/opencascade/include/opencascade
//   generate-flow-fixtures.cc -L/opt/homebrew/opt/opencascade/lib
//   -Wl,-rpath,/opt/homebrew/opt/opencascade/lib -lTKPrim -lTKDESTEP -lTKXSBase
//   -lTKTopAlgo -lTKBRep -lTKG3d -lTKMath -lTKernel -o generate-flow-fixtures
// Run: ./generate-flow-fixtures <output-directory>/
//
// Box faces keep the Open CASCADE BRepPrimAPI_MakeBox order, which the studies select by id:
// face_000001 x = 0, face_000002 x = max, face_000003 y = 0, face_000004 y = max,
// face_000005 z = 0, face_000006 z = max.
#include <BRepCheck_Analyzer.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <STEPControl_Writer.hxx>
#include <gp_Ax2.hxx>
#include <gp_Dir.hxx>
#include <gp_Pnt.hxx>
#include <iostream>
#include <string>

namespace {
bool write(const TopoDS_Shape& shape, const std::string& path) {
  if (!BRepCheck_Analyzer(shape).IsValid()) return false;
  STEPControl_Writer writer;
  if (writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone) return false;
  if (writer.Write(path.c_str()) != IFSelect_RetDone) return false;
  std::cout << "wrote " << path << '\n';
  return true;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 1;
  const std::string dir = argv[1];
  // Lid-driven cavity: unit square section, thin spanwise slab closed by symmetry planes.
  if (!write(BRepPrimAPI_MakeBox(1.0, 1.0, 0.05).Shape(), dir + "cavity-slab.step")) return 2;
  // Plane channel: 1 m long, 0.1 m between the walls, spanwise slab closed by symmetry planes.
  if (!write(BRepPrimAPI_MakeBox(1.0, 0.1, 0.02).Shape(), dir + "plane-channel.step")) return 3;
  // Round pipe: 0.05 m radius, 0.5 m long along +x.
  const gp_Ax2 axis(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0));
  if (!write(BRepPrimAPI_MakeCylinder(axis, 0.05, 0.5).Shape(), dir + "pipe.step")) return 4;
  return 0;
}
