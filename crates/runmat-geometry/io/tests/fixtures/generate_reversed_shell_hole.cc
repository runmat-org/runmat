// Writes reversed_shell_hole.brep: a 10 mm box with a 2 mm radius through hole whose shell is
// stored reversed in its solid (faces reversed in the shell), the layout STEP readers produce.
// Build: clang++ -std=c++17 -I$OCCT/include/opencascade generate_reversed_shell_hole.cc \
//   -L$OCCT/lib -lTKernel -lTKMath -lTKBRep -lTKTopAlgo -lTKPrim -lTKBO -lTKG3d -lTKGeomBase
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepTools.hxx>
#include <BRep_Builder.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <gp_Ax2.hxx>

int main() {
  const TopoDS_Shape box = BRepPrimAPI_MakeBox(10, 10, 10).Shape();
  const TopoDS_Shape hole =
      BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(5, 5, -1), gp_Dir(0, 0, 1)), 2, 12).Shape();
  const TopoDS_Shape cut = BRepAlgoAPI_Cut(box, hole).Shape();
  BRep_Builder builder;
  TopoDS_Shell shell;
  builder.MakeShell(shell);
  for (TopExp_Explorer faces(cut, TopAbs_FACE); faces.More(); faces.Next()) {
    builder.Add(shell, faces.Current().Reversed());
  }
  shell.Closed(Standard_True);
  TopoDS_Solid solid;
  builder.MakeSolid(solid);
  builder.Add(solid, shell.Reversed());
  if (!BRepCheck_Analyzer(solid).IsValid()) {
    return 1;
  }
  BRepTools::Write(solid, "reversed_shell_hole.brep");
  return 0;
}
