// Exact instructional tuning fork, authored for RunMat. Dimensions are millimetres.
// Build: clang++ -std=c++17 -I/opt/homebrew/opt/opencascade/include/opencascade
//   generate-tuning-fork.cc -L/opt/homebrew/opt/opencascade/lib
//   -Wl,-rpath,/opt/homebrew/opt/opencascade/lib -lTKPrim -lTKDESTEP
//   -lTKXSBase -lTKTopAlgo -lTKBRep -lTKG3d -lTKMath -lTKernel -o generate-tuning-fork
// Run: ./generate-tuning-fork tuning-fork.step
#include <BRepBuilderAPI_MakePolygon.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <gp_Vec.hxx>
#include <STEPControl_Writer.hxx>
#include <gp_Pnt.hxx>
#include <iostream>
int main(int argc, char** argv) {
  if (argc != 2) return 1;
  // One closed profile and one extrusion give the exact fork without Boolean seams.
  BRepBuilderAPI_MakePolygon outline;
  const double profile[][2] = {
      {-4, 0}, {4, 0}, {4, 25}, {18, 25}, {18, 100}, {10, 100},
      {10, 35}, {-10, 35}, {-10, 100}, {-18, 100}, {-18, 25}, {-4, 25}};
  for (const auto& point : profile) outline.Add(gp_Pnt(point[0], -3, point[1]));
  outline.Close();
  const auto face = BRepBuilderAPI_MakeFace(outline.Wire()).Face();
  const auto joined = BRepPrimAPI_MakePrism(face, gp_Vec(0, 6, 0)).Shape();
  if (!BRepCheck_Analyzer(joined).IsValid()) return 2;
  STEPControl_Writer writer;
  if (writer.Transfer(joined, STEPControl_AsIs) != IFSelect_RetDone) return 3;
  if (writer.Write(argv[1]) != IFSelect_RetDone) return 4;
  std::cout << "Wrote exact instructional fork: " << argv[1] << '\n';
}
