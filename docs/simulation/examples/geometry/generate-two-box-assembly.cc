// Authored structural-static fixture. Dimensions are metres; names select geometry, not ordinals.
// Build with the same OCCT libraries as generate-flow-fixtures.cc.
// Run: generate-two-box-assembly output.step
// Normalize writer whitespace: perl -pi -e 's/[ \t]+$//' output.step
// Verify unchanged mechanics and imported names: generate-two-box-assembly --verify old.step new.step
#include <APIHeaderSection_MakeHeader.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepGProp.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRep_Tool.hxx>
#include <DESTEP_Parameters.hxx>
#include <GProp_GProps.hxx>
#include <Interface_HArray1OfHAsciiString.hxx>
#include <Interface_Static.hxx>
#include <STEPCAFControl_Reader.hxx>
#include <STEPCAFControl_Writer.hxx>
#include <STEPControl_Controller.hxx>
#include <StepShape_AdvancedFace.hxx>
#include <TDataStd_Name.hxx>
#include <TDocStd_Document.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS.hxx>
#include <TransferBRep.hxx>
#include <XSControl_WorkSession.hxx>
#include <XSControl_TransferReader.hxx>
#include <XCAFApp_Application.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value, const char* diagnostic) {
  if (!value) throw std::runtime_error(diagnostic);
}
bool near(double a, double b) { return std::abs(a - b) < 1e-9; }
Handle(TDocStd_Document) document() {
  Handle(TDocStd_Document) doc;
  XCAFApp_Application::GetApplication()->NewDocument("MDTV-XCAF", doc);
  return doc;
}
gp_Pnt center(const TopoDS_Shape& shape, bool volume = false) {
  GProp_GProps properties;
  if (volume) BRepGProp::VolumeProperties(shape, properties);
  else BRepGProp::SurfaceProperties(shape, properties);
  return properties.CentreOfMass();
}
std::string name(const TDF_Label& label) {
  Handle(TDataStd_Name) value;
  return label.FindAttribute(TDataStd_Name::GetID(), value)
      ? TCollection_AsciiString(value->Get(), '?').ToCString() : "";
}
void generate(const char* output) {
  auto doc = document();
  auto shapes = XCAFDoc_DocumentTool::ShapeTool(doc->Main());
  const TDF_Label assembly = shapes->NewShape();
  TDataStd_Name::Set(assembly, "Structural static assembly");
  for (bool right : {false, true}) {
    // Construct independently: sharing a part definition also shares face names.
    const TopoDS_Shape box = BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape();
    const TDF_Label part = shapes->AddShape(box, Standard_False);
    TDataStd_Name::Set(part, right ? "right_body" : "left_body");
    int supports = 0, loads = 0;
    for (TopExp_Explorer faces(box, TopAbs_FACE); faces.More(); faces.Next()) {
      const gp_Pnt c = center(faces.Current());
      const char* face_name = nullptr;
      if (near(c.X(), 0.0)) { face_name = right ? "right_support" : "left_support"; ++supports; }
      if (!right && near(c.X(), 1.0)) { face_name = "left_load_face"; ++loads; }
      if (face_name) TDataStd_Name::Set(shapes->AddSubShape(part, faces.Current()), face_name);
    }
    require(supports == 1 && loads == (right ? 0 : 1), "ambiguous geometric face rule");
    gp_Trsf placement; placement.SetTranslation(gp_Vec(right ? 10.0 : 0.0, 0.0, 0.0));
    shapes->AddComponent(assembly, part, TopLoc_Location(placement));
  }
  shapes->UpdateAssemblies();
  XCAFDoc_DocumentTool::SetLengthUnit(doc, 1.0, UnitsMethods_LengthUnit_Meter);
  DESTEP_Parameters parameters; parameters.InitFromStatic();
  parameters.WriteUnit = UnitsMethods_LengthUnit_Meter;
  parameters.WriteSubshapeNames = true;
  STEPCAFControl_Writer writer; writer.SetNameMode(true);
  require(writer.Transfer(doc, parameters, STEPControl_AsIs), "STEP transfer failed");
  const auto model = writer.ChangeWriter().Model();
  APIHeaderSection_MakeHeader header(model);
  auto authors = new Interface_HArray1OfHAsciiString(1, 1);
  authors->SetValue(1, new TCollection_HAsciiString("RunMat fixture generator"));
  header.SetAuthor(authors); header.SetOrganization(authors);
  header.SetName(new TCollection_HAsciiString("two-box-assembly.step"));
  header.SetTimeStamp(new TCollection_HAsciiString("2000-01-01T00:00:00"));
  header.Apply(model);
  require(writer.Write(output) == IFSelect_RetDone, "STEP write failed");
}

// Import through OCCT, compare world-space geometry, then inspect actual imported definition names.
void verify(const char* input, bool named) {
  auto doc = document();
  STEPCAFControl_Reader reader; reader.SetNameMode(true);
  require(reader.ReadFile(input) == IFSelect_RetDone, "STEP read failed");
  reader.ChangeReader().SetSystemLengthUnit(1000.0); // OCCT unit in mm: 1000 mm = one metre.
  require(reader.Transfer(doc), "STEP import failed");
  auto shapes = XCAFDoc_DocumentTool::ShapeTool(doc->Main());
  TDF_LabelSequence roots; shapes->GetFreeShapes(roots);
  require(roots.Length() == 1, "expected one assembly root");
  const TopoDS_Shape whole = shapes->GetShape(roots.Value(1));
  require(BRepCheck_Analyzer(whole).IsValid(), "invalid imported BREP");
  std::vector<TopoDS_Shape> solids;
  for (TopExp_Explorer s(whole, TopAbs_SOLID); s.More(); s.Next()) solids.push_back(s.Current());
  require(solids.size() == 2, "expected two solids");
  std::sort(solids.begin(), solids.end(), [](const auto& a, const auto& b) { return center(a, true).X() < center(b, true).X(); });
  for (size_t i = 0; i < solids.size(); ++i) {
    const auto& solid = solids[i];
    GProp_GProps volume, surface; BRepGProp::VolumeProperties(solid, volume); BRepGProp::SurfaceProperties(solid, surface);
    require(near(volume.Mass(), 6.0) && near(surface.Mass(), 22.0), "changed volume or surface area");
    const gp_Pnt c = volume.CentreOfMass();
    require(near(c.X(), i * 10.0 + 0.5) && near(c.Y(), 1) && near(c.Z(), 1.5), "changed solid placement");
    for (auto [kind, count] : {std::pair{TopAbs_FACE, 6}, {TopAbs_EDGE, 12}, {TopAbs_VERTEX, 8}}) {
      TopTools_IndexedMapOfShape entities; TopExp::MapShapes(solid, kind, entities);
      require(entities.Extent() == count, "changed topology inventory");
    }
    for (TopExp_Explorer v(solid, TopAbs_VERTEX); v.More(); v.Next()) {
      const gp_Pnt p = BRep_Tool::Pnt(TopoDS::Vertex(v.Current()));
      require((near(p.X(), i * 10.0) || near(p.X(), i * 10.0 + 1)) &&
          (near(p.Y(), 0) || near(p.Y(), 2)) && (near(p.Z(), 0) || near(p.Z(), 3)), "changed box vertices");
    }
    for (TopExp_Explorer f(solid, TopAbs_FACE); f.More(); f.Next())
      require(BRepAdaptor_Surface(TopoDS::Face(f.Current())).GetType() == GeomAbs_Plane, "changed planar box geometry");
    std::cout << input << " solid x=" << i * 10 << ".." << i * 10 + 1 << " y=0..2 z=0..3 volume=6 area=22 faces=6 edges=12 vertices=8\n";
  }
  if (!named) return;
  std::map<std::string, int> names;
  TDF_LabelSequence labels; shapes->GetShapes(labels);
  for (int i = 1; i <= labels.Length(); ++i) {
    const TDF_Label part = labels.Value(i);
    if (!shapes->IsSimpleShape(part) || shapes->IsReference(part)) continue;
    const std::string body = name(part); ++names[body];
    require(body == "left_body" || body == "right_body", "unexpected part definition");
    // STEPCAF can omit subshape labels. Resolve authored ADVANCED_FACE names
    // through the actual OCCT transfer map, as RunMat's importer does.
    const auto session = reader.Reader().WS();
    const auto model = session->Model();
    for (int j = 1; j <= model->NbEntities(); ++j) {
      auto entity = Handle(StepShape_AdvancedFace)::DownCast(model->Value(j));
      if (entity.IsNull() || entity->Name().IsNull()) continue;
      const std::string face = entity->Name()->ToCString();
      if (face != "left_support" && face != "left_load_face" && face != "right_support") continue;
      const auto shape = TransferBRep::ShapeResult(session->TransferReader()->TransientProcess(), entity);
      bool owned = false;
      for (TopExp_Explorer f(shapes->GetShape(part), TopAbs_FACE); f.More(); f.Next())
        owned = owned || f.Current().IsSame(shape);
      if (!owned) continue;
      require(shape.ShapeType() == TopAbs_FACE, "named boundary is not a face");
      GProp_GProps area; BRepGProp::SurfaceProperties(shape, area);
      const auto c = area.CentreOfMass();
      require(near(area.Mass(), 6) && near(c.X(), face == "left_load_face" ? 1 : 0) && near(c.Y(), 1) && near(c.Z(), 1.5), "wrong named end face");
      require((body == "right_body") == (face == "right_support"), "boundary belongs to wrong solid");
      ++names[face];
    }
  }
  for (const char* expected : {"left_body", "right_body", "left_support", "left_load_face", "right_support"}) {
    require(names[expected] == 1, "imported name missing or ambiguous");
    std::cout << expected << ": exactly one imported entity\n";
  }
}
} // namespace

int main(int argc, char** argv) {
  try {
    STEPControl_Controller::Init();
    require(Interface_Static::SetCVal("xstep.cascade.unit", "M"), "cannot set metre session unit");
    require(Interface_Static::SetCVal("write.step.unit", "M"), "cannot set metre STEP unit");
    if (argc == 2) generate(argv[1]);
    else if (argc == 4 && std::string(argv[1]) == "--verify") { verify(argv[2], false); verify(argv[3], true); }
    else throw std::runtime_error("usage: generate-two-box-assembly output.step | --verify old.step new.step");
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
