// Exact multi-body flow fixtures for conjugate heat transfer and fluid-structure interaction,
// authored for RunMat. Dimensions are metres. Touching bodies are written as an assembly of named
// parts whose coincident faces line up exactly (General Fuse imprint), so the mesh is conformal
// across the fluid and the solid; every selected face carries an authored name.
// Build: clang++ -std=c++17 -I/opt/homebrew/opt/opencascade/include/opencascade
//   generate-coupled-fixtures.cc -L/opt/homebrew/opt/opencascade/lib
//   -Wl,-rpath,/opt/homebrew/opt/opencascade/lib -lTKPrim -lTKBO -lTKDESTEP -lTKXSBase
//   -lTKTopAlgo -lTKBRep -lTKG3d -lTKMath -lTKernel -lTKXCAF -lTKLCAF -lTKCAF -lTKCDF -lTKDE
//   -lTKService -lTKV3d -lTKFillet -lTKShHealing -o generate-coupled-fixtures
// Run: ./generate-coupled-fixtures <output-directory>/
#include <BRepAdaptor_Surface.hxx>
#include <Interface_Static.hxx>
#include <STEPControl_Controller.hxx>
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
  // Shapes are built in metres: the document unit and the declared STEP unit must both be metres,
  // or the writer labels metre values as millimetres.
  XCAFDoc_DocumentTool::SetLengthUnit(doc, 1.0, UnitsMethods_LengthUnit_Meter);
  DESTEP_Parameters params; params.InitFromStatic(); params.WriteSubshapeNames = true;
  params.WriteUnit = UnitsMethods_LengthUnit_Meter;
  STEPCAFControl_Writer writer; writer.SetNameMode(true); writer.SetColorMode(true);
  if (!writer.Transfer(doc, params, STEPControl_AsIs)) return false;
  if (writer.Write(path.c_str()) != IFSelect_RetDone) return false;
  std::cout << "wrote " << path << " (" << bodies.size() << " parts)\n";
  return true;
}

}  // namespace

namespace {

const Quantity_Color kWater(0.55, 0.75, 1.00, Quantity_TOC_RGB);
const Quantity_Color kAluminum(0.75, 0.76, 0.78, Quantity_TOC_RGB);
const Quantity_Color kRubber(0.20, 0.20, 0.22, Quantity_TOC_RGB);

bool near(double a, double b) { return std::abs(a - b) < 1e-7; }

// Conjugate plate-channel reference: a 0.3 m long, 10 mm aluminium plate under a 10 mm coolant
// channel, in a 5 mm spanwise slab closed by symmetry planes. Heated from below; adiabatic lid.
bool plate_channel(const std::string& dir) {
  const double L = 0.3, t = 0.01, h = 0.01, w = 0.005;
  TopoDS_Shape plate = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), L, t, w).Shape();
  TopoDS_Shape channel = BRepPrimAPI_MakeBox(gp_Pnt(0, t, 0), L, h, w).Shape();
  std::vector<FaceRule> faces = {
      {"Plate", "Heater", [=](const gp_Pnt& c, const gp_Dir&) { return near(c.Y(), 0); }},
      {"Plate", "Wetted wall", [=](const gp_Pnt& c, const gp_Dir&) { return near(c.Y(), t); }},
      {"Plate", "Plate ends", [=](const gp_Pnt& c, const gp_Dir&) { return near(c.X(), 0) || near(c.X(), L); }},
      {"Plate", "Plate sides", [=](const gp_Pnt& c, const gp_Dir&) { return near(c.Z(), 0) || near(c.Z(), w); }},
      {"Coolant", "Channel wall", [=](const gp_Pnt& c, const gp_Dir&) { return near(c.Y(), t); }},
      {"Coolant", "Lid", [=](const gp_Pnt& c, const gp_Dir&) { return near(c.Y(), t + h); }},
      {"Coolant", "Inlet", [=](const gp_Pnt& c, const gp_Dir&) { return near(c.X(), 0); }},
      {"Coolant", "Outlet", [=](const gp_Pnt& c, const gp_Dir&) { return near(c.X(), L); }},
      {"Coolant", "Channel sides", [=](const gp_Pnt& c, const gp_Dir&) { return near(c.Z(), 0) || near(c.Z(), w); }},
  };
  return write_cell(dir + "cht-plate-channel.step",
                    {{plate, "Plate", kAluminum}, {channel, "Coolant", kWater}}, faces);
}

// Compliant tube reference: water in a 10 mm radius bore through a 4 mm thick elastomer tube,
// 0.3 m long. The tube ends are clamped; its outer surface is free.
bool compliant_tube(const std::string& dir) {
  const double a = 0.01, b = 0.014, L = 0.3;
  const gp_Ax2 axis(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0));
  TopoDS_Shape core = BRepPrimAPI_MakeCylinder(axis, a, L).Shape();
  TopoDS_Shape tube = BRepAlgoAPI_Cut(BRepPrimAPI_MakeCylinder(axis, b, L).Shape(), core).Shape();
  std::vector<FaceRule> faces = {
      {"Water", "Inlet", [=](const gp_Pnt& c, const gp_Dir&) { return near(c.X(), 0); }},
      {"Water", "Outlet", [=](const gp_Pnt& c, const gp_Dir&) { return near(c.X(), L); }},
      {"Water", "Bore", [=](const gp_Pnt& c, const gp_Dir&) { return near(std::hypot(c.Y(), c.Z()), a); }},
      {"Tube", "Tube bore", [=](const gp_Pnt& c, const gp_Dir&) { return near(std::hypot(c.Y(), c.Z()), a); }},
      {"Tube", "Tube ends", [=](const gp_Pnt& c, const gp_Dir&) { return near(c.X(), 0) || near(c.X(), L); }},
      {"Tube", "Tube outer", [=](const gp_Pnt& c, const gp_Dir&) { return near(std::hypot(c.Y(), c.Z()), b); }},
  };
  return write_cell(dir + "compliant-tube.step", {{core, "Water", kWater}, {tube, "Tube", kRubber}}, faces);
}

// Cold plate: a 200 x 100 x 12 mm aluminium plate with two 10 x 4 mm coolant passages along x,
// centred 2 mm below the heated top face.
bool cold_plate(const std::string& dir) {
  const double L = 0.2, W = 0.1, T = 0.012, cw = 0.01, ch = 0.004, y0 = 0.004;
  TopoDS_Shape block = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), L, T, W).Shape();
  TopoDS_Shape passages = BRepAlgoAPI_Fuse(
      BRepPrimAPI_MakeBox(gp_Pnt(0, y0, 0.025 - cw / 2), L, ch, cw).Shape(),
      BRepPrimAPI_MakeBox(gp_Pnt(0, y0, 0.075 - cw / 2), L, ch, cw).Shape()).Shape();
  TopoDS_Shape plate = BRepAlgoAPI_Cut(block, passages).Shape();
  auto in_passage = [=](const gp_Pnt& c) {
    return c.Y() > y0 - 1e-7 && c.Y() < y0 + ch + 1e-7 &&
           (std::abs(c.Z() - 0.025) < cw / 2 + 1e-7 || std::abs(c.Z() - 0.075) < cw / 2 + 1e-7);
  };
  std::vector<FaceRule> faces = {
      {"Coolant", "Inlet", [=](const gp_Pnt& c, const gp_Dir&) { return near(c.X(), 0); }},
      {"Coolant", "Outlet", [=](const gp_Pnt& c, const gp_Dir&) { return near(c.X(), L); }},
      {"Coolant", "Channel wall", [=](const gp_Pnt& c, const gp_Dir&) { return !near(c.X(), 0) && !near(c.X(), L); }},
      {"Plate", "Plate wall", [=](const gp_Pnt& c, const gp_Dir&) { return in_passage(c) && !near(c.X(), 0) && !near(c.X(), L); }},
      {"Plate", "Heater", [=](const gp_Pnt& c, const gp_Dir&) { return near(c.Y(), T); }},
  };
  return write_cell(dir + "cold-plate.step", {{plate, "Plate", kAluminum}, {passages, "Coolant", kWater}}, faces);
}

// Flexible duct: water in a 20 x 20 mm square passage, 0.2 m long, inside a 3 mm thick rubber
// wall. The wall's two end faces are mounted.
bool flexible_duct(const std::string& dir) {
  const double L = 0.2, a = 0.02, t = 0.003;
  TopoDS_Shape water = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), L, a, a).Shape();
  TopoDS_Shape wall = BRepAlgoAPI_Cut(BRepPrimAPI_MakeBox(gp_Pnt(0, -t, -t), L, a + 2 * t, a + 2 * t).Shape(), water).Shape();
  auto bore = [=](const gp_Pnt& c) {
    return (near(c.Y(), 0) || near(c.Y(), a) || near(c.Z(), 0) || near(c.Z(), a)) && !near(c.X(), 0) && !near(c.X(), L);
  };
  std::vector<FaceRule> faces = {
      {"Water", "Inlet", [=](const gp_Pnt& c, const gp_Dir&) { return near(c.X(), 0); }},
      {"Water", "Outlet", [=](const gp_Pnt& c, const gp_Dir&) { return near(c.X(), L); }},
      {"Water", "Fluid wall", [=](const gp_Pnt& c, const gp_Dir&) { return bore(c); }},
      {"Wall", "Structural wall", [=](const gp_Pnt& c, const gp_Dir&) { return bore(c); }},
      {"Wall", "Mount", [=](const gp_Pnt& c, const gp_Dir&) { return near(c.X(), 0) || near(c.X(), L); }},
  };
  return write_cell(dir + "flexible-duct.step", {{water, "Water", kWater}, {wall, "Wall", kRubber}}, faces);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) { std::cerr << "usage: generate-coupled-fixtures <out-dir/>\n"; return 1; }
  const std::string dir = argv[1];
  STEPControl_Controller::Init();
  if (!Interface_Static::SetCVal("xstep.cascade.unit", "M")) return 3;
  if (!Interface_Static::SetCVal("write.step.unit", "M")) return 3;
  return plate_channel(dir) && compliant_tube(dir) && cold_plate(dir) && flexible_duct(dir) ? 0 : 2;
}
