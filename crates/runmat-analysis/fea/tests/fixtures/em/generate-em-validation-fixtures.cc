// Exact STEP fixtures for RunMat's electromagnetic analytic checks (coax, solenoid, parallel plate,
// coaxial capacitor, straight wire). Dimensions in millimetres. Same writer as the website EM
// fixtures (oct-13-2026/website/physics/studies/electromagnetic/generate-em-fixtures.cc): touching
// bodies are fragmented by General Fuse and written as an assembly of named parts, with names on
// the faces studies select (`select name`).
// Build: clang++ -std=c++17 -I/opt/homebrew/opt/opencascade/include/opencascade generate-em-validation-fixtures.cc \
//   -L/opt/homebrew/opt/opencascade/lib -Wl,-rpath,/opt/homebrew/opt/opencascade/lib \
//   -lTKPrim -lTKBO -lTKDESTEP -lTKXSBase -lTKTopAlgo -lTKBRep -lTKG3d -lTKMath -lTKernel \
//   -lTKXCAF -lTKLCAF -lTKCAF -lTKCDF -lTKDE -lTKService -lTKV3d -lTKFillet -lTKShHealing \
//   -o generate-em-validation-fixtures
// Run: ./generate-em-validation-fixtures <out-dir>/
#include <BRepAdaptor_Surface.hxx>
#include <BRepAlgoAPI_BuilderAlgo.hxx>
#include <BRep_Builder.hxx>
#include <TopoDS_Compound.hxx>
#include <BRepLProp_SLProps.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepBuilderAPI_Copy.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakePolygon.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepFilletAPI_MakeFillet.hxx>
#include <BRepGProp.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <BRepPrimAPI_MakeRevol.hxx>
#include <DESTEP_Parameters.hxx>
#include <GProp_GProps.hxx>
#include <Quantity_Color.hxx>
#include <STEPCAFControl_Writer.hxx>
#include <TDataStd_Name.hxx>
#include <TDocStd_Document.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS.hxx>
#include <XCAFApp_Application.hxx>
#include <XCAFDoc_ColorTool.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>
#include <gp_Ax1.hxx>
#include <gp_Ax2.hxx>
#include <gp_Trsf.hxx>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Body { TopoDS_Shape shape; std::string name; Quantity_Color color; };
struct FaceRule { std::string body; std::string name; std::function<bool(const gp_Pnt&, const gp_Dir&)> test; };

gp_Pnt centroid(const TopoDS_Shape& s, bool volume) {
  GProp_GProps p; if (volume) BRepGProp::VolumeProperties(s, p); else BRepGProp::SurfaceProperties(s, p);
  return p.CentreOfMass();
}
double volume_of(const TopoDS_Shape& s) { GProp_GProps p; BRepGProp::VolumeProperties(s, p); return p.Mass(); }

gp_Pnt face_point(const TopoDS_Face& f) {
  BRepAdaptor_Surface s(f);
  return s.Value(0.5 * (s.FirstUParameter() + s.LastUParameter()), 0.5 * (s.FirstVParameter() + s.LastVParameter()));
}

gp_Dir face_normal(const TopoDS_Face& f) {
  BRepAdaptor_Surface s(f);
  const double u = 0.5 * (s.FirstUParameter() + s.LastUParameter());
  const double v = 0.5 * (s.FirstVParameter() + s.LastVParameter());
  BRepLProp_SLProps props(s, u, v, 1, 1e-7);
  if (!props.IsNormalDefined()) return gp_Dir(0, 0, 1);
  gp_Dir n = props.Normal();
  if (f.Orientation() == TopAbs_REVERSED) n.Reverse();
  return n;
}

// Writes an assembly of named parts, the way a CAD tool exports touching bodies. The bodies are
// fragmented first (General Fuse) so each part's boundary is imprinted where it touches another
// part: coincident faces line up exactly, and RunMat can join them into a conformal mesh
// (DEF-281). Each fragment is owned by the smallest input body that contains it, so an enclosing
// air box keeps only the space outside the parts. Face rules name faces of each part.
bool write_cell(const std::string& path, std::vector<Body> bodies, const std::vector<FaceRule>& faces) {
  TopoDS_Shape cell = bodies.front().shape;
  if (bodies.size() > 1) {
    BRepAlgoAPI_BuilderAlgo fuse; TopTools_ListOfShape args;
    for (const auto& b : bodies) args.Append(b.shape);
    fuse.SetArguments(args); fuse.SetRunParallel(false); fuse.Build();
    if (!fuse.IsDone() || fuse.HasErrors()) { std::cerr << "fuse failed: " << path << '\n'; return false; }
    cell = fuse.Shape();
  }
  // Group fragments by owning body.
  std::vector<TopoDS_Compound> parts(bodies.size());
  std::vector<int> part_solids(bodies.size(), 0);
  BRep_Builder builder;
  for (auto& part : parts) builder.MakeCompound(part);
  for (TopExp_Explorer e(cell, TopAbs_SOLID); e.More(); e.Next()) {
    const TopoDS_Shape solid = e.Current();
    int owner = -1; double owner_volume = 1e300;
    for (size_t i = 0; i < bodies.size(); ++i) {
      const TopoDS_Shape rest = BRepAlgoAPI_Cut(solid, bodies[i].shape).Shape();
      if (volume_of(rest) < 1e-6 * volume_of(solid) && volume_of(bodies[i].shape) < owner_volume) {
        owner = static_cast<int>(i); owner_volume = volume_of(bodies[i].shape);
      }
    }
    if (owner < 0) { std::cerr << "unowned fragment in " << path << '\n'; return false; }
    builder.Add(parts[owner], solid); ++part_solids[owner];
  }

  Handle(TDocStd_Document) doc; XCAFApp_Application::GetApplication()->NewDocument("MDTV-XCAF", doc);
  Handle(XCAFDoc_ShapeTool) shapes = XCAFDoc_DocumentTool::ShapeTool(doc->Main());
  Handle(XCAFDoc_ColorTool) colors = XCAFDoc_DocumentTool::ColorTool(doc->Main());
  TDF_Label assembly = shapes->NewShape();
  TDataStd_Name::Set(assembly, "Cell");
  for (size_t i = 0; i < bodies.size(); ++i) {
    if (part_solids[i] == 0) { std::cerr << "empty part " << bodies[i].name << '\n'; return false; }
    // Each part owns its own copy of the imprinted geometry, as separate CAD bodies do; shared
    // topology between products would make their face names ambiguous.
    const TopoDS_Shape fragment = part_solids[i] == 1 ? TopoDS_Shape(TopExp_Explorer(parts[i], TopAbs_SOLID).Current())
                                                     : TopoDS_Shape(parts[i]);
    const TopoDS_Shape part = BRepBuilderAPI_Copy(fragment).Shape();
    TDF_Label label = shapes->AddShape(part, Standard_False);
    TDataStd_Name::Set(label, bodies[i].name.c_str());
    colors->SetColor(label, bodies[i].color, XCAFDoc_ColorSurf);
    TDF_Label component = shapes->AddComponent(assembly, label, TopLoc_Location());
    TDataStd_Name::Set(component, bodies[i].name.c_str());
    for (const auto& rule : faces) {
      if (rule.body != bodies[i].name) continue;
      int matched = 0;
      for (TopExp_Explorer f(part, TopAbs_FACE); f.More(); f.Next()) {
        const TopoDS_Face face = TopoDS::Face(f.Current());
        if (!rule.test(face_point(face), face_normal(face))) continue;
        TDF_Label face_label = shapes->AddSubShape(label, face);
        if (face_label.IsNull()) continue;
        TDataStd_Name::Set(face_label, rule.name.c_str());
        ++matched;
      }
      if (matched == 0) { std::cerr << "face rule matched nothing: " << rule.name << '\n'; return false; }
      std::cout << "  " << rule.name << ": " << matched << " face(s)\n";
    }
  }
  shapes->UpdateAssemblies();
  DESTEP_Parameters params; params.InitFromStatic(); params.WriteSubshapeNames = true;
  STEPCAFControl_Writer writer; writer.SetNameMode(true); writer.SetColorMode(true);
  if (!writer.Transfer(doc, params, STEPControl_AsIs)) return false;
  if (writer.Write(path.c_str()) != IFSelect_RetDone) return false;
  std::cout << "wrote " << path << " (" << bodies.size() << " parts)\n";
  return true;
}

}  // namespace

namespace {


const Quantity_Color kCopper(0.85, 0.50, 0.25, Quantity_TOC_RGB);
const Quantity_Color kAir(0.80, 0.90, 1.00, Quantity_TOC_RGB);
const Quantity_Color kEpoxy(0.70, 0.55, 0.20, Quantity_TOC_RGB);

TopoDS_Shape zcylinder(double radius, double z0, double length) {
  return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, z0), gp_Dir(0, 0, 1)), radius, length).Shape();
}
TopoDS_Shape annulus(double inner, double outer, double z0, double length) {
  return BRepAlgoAPI_Cut(zcylinder(outer, z0, length), zcylinder(inner, z0 - 1, length + 2)).Shape();
}
double radial(const gp_Pnt& c) { return std::hypot(c.X(), c.Y()); }

// Coaxial line: copper conductor a = 2 mm inside an air annulus to b = 10 mm, 40 mm long.
bool coax(const std::string& dir) {
  std::vector<FaceRule> faces = {
      {"Conductor", "Conductor in", [](const gp_Pnt& c, const gp_Dir&) { return c.Z() < 0.01; }},
      {"Conductor", "Conductor out", [](const gp_Pnt& c, const gp_Dir&) { return c.Z() > 39.99; }},
      {"Air", "Air in", [](const gp_Pnt& c, const gp_Dir&) { return c.Z() < 0.01; }},
      {"Air", "Air out", [](const gp_Pnt& c, const gp_Dir&) { return c.Z() > 39.99; }},
      {"Air", "Outer shell", [](const gp_Pnt& c, const gp_Dir&) { return radial(c) > 9.9 && c.Z() > 0.01 && c.Z() < 39.99; }},
  };
  return write_cell(dir + "coax.step", {{zcylinder(2, 0, 40), "Conductor", kCopper}, {zcylinder(10, 0, 40), "Air", kAir}}, faces);
}

// Thick solenoid: winding 20 to 24 mm radius, 100 mm long, centred, in an air cylinder
// 150 mm radius, 400 mm long.
bool solenoid(const std::string& dir) {
  std::vector<FaceRule> faces = {
      {"Air", "Outer boundary", [](const gp_Pnt& c, const gp_Dir&) { return radial(c) > 149.9 || std::abs(c.Z()) > 199.9; }},
  };
  return write_cell(dir + "solenoid.step", {{annulus(20, 24, -50, 100), "Winding", kCopper}, {zcylinder(150, -200, 400), "Air", kAir}}, faces);
}

// Parallel-plate capacitor dielectric, 50 x 50 x 2 mm.
bool parallel_plate(const std::string& dir) {
  std::vector<FaceRule> faces = {
      {"Dielectric", "Top plate", [](const gp_Pnt& c, const gp_Dir& n) { return c.Z() > 1.99 && n.Z() > 0.9; }},
      {"Dielectric", "Bottom plate", [](const gp_Pnt& c, const gp_Dir& n) { return c.Z() < 0.01 && n.Z() < -0.9; }},
  };
  return write_cell(dir + "parallel-plate.step", {{BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 50, 50, 2).Shape(), "Dielectric", kEpoxy}}, faces);
}

// Coaxial capacitor dielectric: annulus a = 2 mm to b = 10 mm, 20 mm long.
bool coax_dielectric(const std::string& dir) {
  std::vector<FaceRule> faces = {
      {"Dielectric", "Inner conductor", [](const gp_Pnt& c, const gp_Dir&) { return radial(c) < 2.1 && c.Z() > 0.01 && c.Z() < 19.99; }},
      {"Dielectric", "Outer conductor", [](const gp_Pnt& c, const gp_Dir&) { return radial(c) > 9.9 && c.Z() > 0.01 && c.Z() < 19.99; }},
  };
  return write_cell(dir + "coax-dielectric.step", {{annulus(2, 10, 0, 20), "Dielectric", kEpoxy}}, faces);
}

// Straight copper wire, radius 1 mm, 100 mm long.
bool wire(const std::string& dir) {
  std::vector<FaceRule> faces = {
      {"Wire", "Wire in", [](const gp_Pnt& c, const gp_Dir&) { return c.Z() < 0.01; }},
      {"Wire", "Wire out", [](const gp_Pnt& c, const gp_Dir&) { return c.Z() > 99.99; }},
  };
  return write_cell(dir + "wire.step", {{zcylinder(1, 0, 100), "Wire", kCopper}}, faces);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) { std::cerr << "usage: generate-em-validation-fixtures <out-dir/>\n"; return 1; }
  const std::string dir = argv[1];
  bool ok = coax(dir) && solenoid(dir) && parallel_plate(dir) && coax_dielectric(dir) && wire(dir);
  return ok ? 0 : 2;
}
