// Multi-body STEP fixtures for shared-interface (conformal) import. Dimensions in millimetres.
// Every body carries a product (or solid) name and the faces a test selects carry authored names
// (ADVANCED_FACE('<name>', ...)).
//
//   touching_parts_assembly.step  A 10 mm block ("Block") standing on a 30 x 30 x 5 mm plate
//                                 ("Plate") as two separate products: the block's base covers part
//                                 of the plate's top face, as a CAD assembly exports it.
//   part_in_air_assembly.step     A 10 mm cube ("Part") inside a 30 mm air box ("Air") written as
//                                 two products; the air carries one cavity whose faces coincide
//                                 with the part's faces.
//   two_cavity_air_assembly.step  Two cubes ("Left", "Right") in one air box: the air solid has two
//                                 disjoint cavities.
//   stacked_blocks_assembly.step  Two 20 x 20 x 10 mm blocks stacked along z as separate products:
//                                 "Copper" on top (its top face "Hot face") and "Steel" below (its
//                                 bottom face "Cold face"); they touch over their full 20 x 20 face.
//   overlapping_parts_assembly.step  Two 10 mm cubes ("First", "Second") that interpenetrate by
//                                 5 mm, which shared interfaces must reject.
//   free_wireframe_part.step      One product: a 10 mm cube plus a free construction line that
//                                 starts at one of its corners (no face uses the line).
//   named_cavity_part.step        One product: a 20 mm block ("Insulator") with a closed 6 mm cavity
//                                 (BREP_WITH_VOIDS); the 6 cavity faces are named "Electrode" and
//                                 the bottom face "Ground plane".
//   pocket_coolant_assembly.step  A 30 x 30 x 10 mm plate ("Plate") with a 10 x 10 x 5 mm pocket
//                                 open at its top, and the coolant ("Coolant") filling the pocket,
//                                 as two products: the coolant's floor and walls coincide with the
//                                 pocket's faces and its top ("Coolant top") is flush with the
//                                 plate's top face, which surrounds it.
//   channel_coolant_assembly.step A 40 x 20 x 10 mm plate ("Plate") with a 10 x 4 mm channel along
//                                 x, and the coolant ("Coolant") filling the channel, as two
//                                 products: the coolant's ends ("Inlet", "Outlet") are flush with
//                                 the plate's end faces, which surround them. The channel faces
//                                 are "Channel wall" on the coolant and "Plate wall" on the plate;
//                                 the plate's top is "Heater".
//   fused_compound.step           One product whose shape is the General Fuse of a cube and its
//                                 air box: two solids that already share the cavity faces, named by
//                                 their solid entities ("Part", "Air").
//
// Build: clang++ -std=c++17 -I/opt/homebrew/opt/opencascade/include/opencascade \
//   generate_shared_interface_fixtures.cc -L/opt/homebrew/opt/opencascade/lib \
//   -Wl,-rpath,/opt/homebrew/opt/opencascade/lib -lTKPrim -lTKBO -lTKDESTEP -lTKXSBase \
//   -lTKTopAlgo -lTKBRep -lTKG3d -lTKMath -lTKernel -lTKXCAF -lTKLCAF -lTKCAF -lTKCDF -lTKDE \
//   -o generate_shared_interface_fixtures
// Run: ./generate_shared_interface_fixtures <out-dir/>
#include <BRepAlgoAPI_BuilderAlgo.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepBuilderAPI_Copy.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRep_Builder.hxx>
#include <TopoDS_Compound.hxx>
#include <BRepGProp.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <DESTEP_Parameters.hxx>
#include <GProp_GProps.hxx>
#include <STEPCAFControl_Writer.hxx>
#include <TDataStd_Name.hxx>
#include <TDocStd_Document.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <XCAFApp_Application.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>

#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct FaceRule {
  std::string name;
  std::function<bool(const gp_Pnt&)> test;
};
struct Part {
  std::string name;
  TopoDS_Shape shape;
  std::vector<FaceRule> faces;
};

gp_Pnt face_centroid(const TopoDS_Shape& face) {
  GProp_GProps props;
  BRepGProp::SurfaceProperties(face, props);
  return props.CentreOfMass();
}

double volume(const TopoDS_Shape& shape) {
  GProp_GProps props;
  BRepGProp::VolumeProperties(shape, props);
  return props.Mass();
}

void name_faces(const Handle(XCAFDoc_ShapeTool)& shapes, const TDF_Label& owner,
                const TopoDS_Shape& shape, const std::vector<FaceRule>& rules) {
  for (const FaceRule& rule : rules) {
    int matched = 0;
    for (TopExp_Explorer faces(shape, TopAbs_FACE); faces.More(); faces.Next()) {
      if (!rule.test(face_centroid(faces.Current()))) continue;
      TDF_Label label = shapes->AddSubShape(owner, faces.Current());
      if (label.IsNull()) continue;
      TDataStd_Name::Set(label, rule.name.c_str());
      ++matched;
    }
    if (matched == 0) {
      std::cerr << "face rule matched nothing: " << rule.name << '\n';
      std::exit(2);
    }
  }
}

bool write(const Handle(TDocStd_Document)& doc, const std::string& path) {
  DESTEP_Parameters params;
  params.InitFromStatic();
  params.WriteSubshapeNames = true;
  STEPCAFControl_Writer writer;
  writer.SetNameMode(true);
  if (!writer.Transfer(doc, params, STEPControl_AsIs)) return false;
  if (writer.Write(path.c_str()) != IFSelect_RetDone) return false;
  std::cout << "wrote " << path << '\n';
  return true;
}

Handle(TDocStd_Document) new_document() {
  Handle(TDocStd_Document) doc;
  XCAFApp_Application::GetApplication()->NewDocument("MDTV-XCAF", doc);
  return doc;
}

// Separate products under one assembly, each owning its own copy of its geometry.
bool write_assembly(const std::string& path, const std::string& assembly_name,
                    const std::vector<Part>& parts) {
  Handle(TDocStd_Document) doc = new_document();
  Handle(XCAFDoc_ShapeTool) shapes = XCAFDoc_DocumentTool::ShapeTool(doc->Main());
  TDF_Label assembly = shapes->NewShape();
  TDataStd_Name::Set(assembly, assembly_name.c_str());
  for (const Part& part : parts) {
    TDF_Label label = shapes->AddShape(part.shape, Standard_False);
    TDataStd_Name::Set(label, part.name.c_str());
    TDF_Label component = shapes->AddComponent(assembly, label, TopLoc_Location());
    TDataStd_Name::Set(component, part.name.c_str());
    name_faces(shapes, label, part.shape, part.faces);
  }
  shapes->UpdateAssemblies();
  return write(doc, path);
}

TopoDS_Shape box(double x, double y, double z, double dx, double dy, double dz) {
  return BRepPrimAPI_MakeBox(gp_Pnt(x, y, z), dx, dy, dz).Shape();
}

// A standalone copy of a shape, as a CAD tool writes each part: a boolean result reuses its tool's
// faces, which would otherwise make two parts share one STEP face.
TopoDS_Shape own(const TopoDS_Shape& shape) { return BRepBuilderAPI_Copy(shape).Shape(); }

bool on_air_boundary(const gp_Pnt& c) {
  return std::abs(c.X()) > 14.9 || std::abs(c.Y()) > 14.9 || std::abs(c.Z()) > 14.9;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: generate_shared_interface_fixtures <out-dir/>\n";
    return 1;
  }
  const std::string dir = argv[1];

  const TopoDS_Shape plate = box(-15, -15, -5, 30, 30, 5);
  const TopoDS_Shape block = box(-5, -5, 0, 10, 10, 10);
  bool ok = write_assembly(
      dir + "touching_parts_assembly.step", "Stack",
      {{"Plate", plate, {{"Base", [](const gp_Pnt& c) { return c.Z() < -4.9; }}}},
       {"Block", block, {{"Top load", [](const gp_Pnt& c) { return c.Z() > 9.9; }}}}});

  const TopoDS_Shape air_box = box(-15, -15, -15, 30, 30, 30);
  const TopoDS_Shape part = box(-5, -5, -5, 10, 10, 10);
  const TopoDS_Shape air = BRepAlgoAPI_Cut(air_box, part).Shape();
  ok = ok && write_assembly(dir + "part_in_air_assembly.step", "Cell",
                            {{"Part", part, {{"Part top", [](const gp_Pnt& c) {
                                                return c.Z() > 4.9 && c.Z() < 5.1;
                                              }}}},
                             {"Air", air, {{"Outer boundary", on_air_boundary}}}});

  const TopoDS_Shape left = box(-12, -4, -4, 8, 8, 8);
  const TopoDS_Shape right = box(4, -4, -4, 8, 8, 8);
  const TopoDS_Shape two_cavity_air =
      BRepAlgoAPI_Cut(BRepAlgoAPI_Cut(air_box, left).Shape(), right).Shape();
  ok = ok && write_assembly(dir + "two_cavity_air_assembly.step", "Cell",
                            {{"Left", left, {}},
                             {"Right", right, {}},
                             {"Air", two_cavity_air, {{"Outer boundary", on_air_boundary}}}});

  ok = ok && write_assembly(
      dir + "stacked_blocks_assembly.step", "Stack",
      {{"Copper", box(0, 0, 10, 20, 20, 10), {{"Hot face", [](const gp_Pnt& c) { return c.Z() > 19.9; }}}},
       {"Steel", box(0, 0, 0, 20, 20, 10), {{"Cold face", [](const gp_Pnt& c) { return c.Z() < 0.1; }}}}});

  ok = ok && write_assembly(dir + "overlapping_parts_assembly.step", "Clash",
                            {{"First", box(0, 0, 0, 10, 10, 10), {}},
                             {"Second", box(5, 0, 0, 10, 10, 10), {}}});

  {
    const TopoDS_Shape pocket = box(-5, -5, 0, 10, 10, 5);
    ok = ok && write_assembly(
        dir + "pocket_coolant_assembly.step", "Cold plate",
        {{"Plate", own(BRepAlgoAPI_Cut(box(-15, -15, -5, 30, 30, 10), pocket).Shape()),
          {{"Plate base", [](const gp_Pnt& c) { return c.Z() < -4.9; }}}},
         {"Coolant", own(pocket), {{"Coolant top", [](const gp_Pnt& c) { return c.Z() > 4.9; }}}}});
    const TopoDS_Shape channel = box(0, 5, 3, 40, 10, 4);
    auto in_channel = [](const gp_Pnt& c) {
      return c.X() > 0.1 && c.X() < 39.9 && c.Y() > 4.9 && c.Y() < 15.1 && c.Z() > 2.9 && c.Z() < 7.1;
    };
    ok = ok && write_assembly(
        dir + "channel_coolant_assembly.step", "Cold plate",
        {{"Plate", own(BRepAlgoAPI_Cut(box(0, 0, 0, 40, 20, 10), channel).Shape()),
          {{"Heater", [](const gp_Pnt& c) { return c.Z() > 9.9; }}, {"Plate wall", in_channel}}},
         {"Coolant", own(channel),
          {{"Inlet", [](const gp_Pnt& c) { return c.X() < 0.1; }},
           {"Outlet", [](const gp_Pnt& c) { return c.X() > 39.9; }},
           {"Channel wall", in_channel}}}});
  }

  {
    BRep_Builder builder;
    TopoDS_Compound part;
    builder.MakeCompound(part);
    builder.Add(part, box(0, 0, 0, 10, 10, 10));
    builder.Add(part, BRepBuilderAPI_MakeEdge(gp_Pnt(10, 10, 10), gp_Pnt(20, 25, 30)).Edge());
    Handle(TDocStd_Document) doc = new_document();
    Handle(XCAFDoc_ShapeTool) shapes = XCAFDoc_DocumentTool::ShapeTool(doc->Main());
    TDataStd_Name::Set(shapes->AddShape(part, Standard_False), "Cube");
    ok = ok && write(doc, dir + "free_wireframe_part.step");
  }

  {
    const TopoDS_Shape block = BRepAlgoAPI_Cut(box(-10, -10, -10, 20, 20, 20), box(-3, -3, -3, 6, 6, 6)).Shape();
    Handle(TDocStd_Document) doc = new_document();
    Handle(XCAFDoc_ShapeTool) shapes = XCAFDoc_DocumentTool::ShapeTool(doc->Main());
    TDF_Label label = shapes->AddShape(block, Standard_False);
    TDataStd_Name::Set(label, "Insulator");
    name_faces(shapes, label, block,
               {{"Electrode", [](const gp_Pnt& c) { return std::abs(c.X()) < 3.1 && std::abs(c.Y()) < 3.1 && std::abs(c.Z()) < 3.1; }},
                {"Ground plane", [](const gp_Pnt& c) { return c.Z() < -9.9; }}});
    ok = ok && write(doc, dir + "named_cavity_part.step");
  }

  // One product: the General Fuse of the cube and the air box shares the cube faces.
  BRepAlgoAPI_BuilderAlgo fuse;
  TopTools_ListOfShape arguments;
  arguments.Append(air_box);
  arguments.Append(part);
  fuse.SetArguments(arguments);
  fuse.Build();
  if (!fuse.IsDone() || fuse.HasErrors()) {
    std::cerr << "general fuse failed\n";
    return 2;
  }
  Handle(TDocStd_Document) doc = new_document();
  Handle(XCAFDoc_ShapeTool) shapes = XCAFDoc_DocumentTool::ShapeTool(doc->Main());
  TDF_Label cell = shapes->AddShape(fuse.Shape(), Standard_False);
  TDataStd_Name::Set(cell, "Cell");
  for (TopExp_Explorer solids(fuse.Shape(), TopAbs_SOLID); solids.More(); solids.Next()) {
    TDF_Label label = shapes->AddSubShape(cell, solids.Current());
    TDataStd_Name::Set(label, volume(solids.Current()) < 2000.0 ? "Part" : "Air");
  }
  name_faces(shapes, cell, fuse.Shape(), {{"Outer boundary", on_air_boundary}});
  ok = ok && write(doc, dir + "fused_compound.step");
  return ok ? 0 : 2;
}
