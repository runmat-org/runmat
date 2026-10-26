// Writes a 0.5 x 0.1 x 0.02 box whose IGES header declares millimetres: metre values under a
// millimetre label, for the authored-units override witness.
#include <BRepPrimAPI_MakeBox.hxx>
#include <IGESControl_Controller.hxx>
#include <IGESControl_Writer.hxx>
#include <Interface_Static.hxx>
int main(int argc, char** argv) {
  if (argc != 2) return 1;
  IGESControl_Controller::Init();
  Interface_Static::SetIVal("write.iges.brep.mode", 1);
  IGESControl_Writer writer("MM", 1);
  if (!writer.AddShape(BRepPrimAPI_MakeBox(0.5, 0.1, 0.02).Shape())) return 2;
  writer.ComputeModel();
  return writer.Write(argv[1]) ? 0 : 3;
}
