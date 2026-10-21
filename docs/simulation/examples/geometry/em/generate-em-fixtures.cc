// Exact STEP fixtures for RunMat's electromagnetic examples and analytic checks. Dimensions in
// millimetres. Every body and every face a study selects carries an authored name, so studies use
// `select name`. Bodies that touch are fragmented by General Fuse, so their boundaries are imprinted
// where they meet and coincident faces line up exactly; studies import them with
// `interfaces = shared` for a conformal mesh across materials and air.
//
// Each fragment has exactly one owner: the smallest input body that contains it. The generator
// checks this (every fragment lies wholly inside or wholly outside each body, no two fragments
// coincide, part volumes add up to the fused cell) and re-reads every written file to confirm
// each part holds exactly the solids it was given, before reporting success.
//
// Build: clang++ -std=c++17 -I/opt/homebrew/opt/opencascade/include/opencascade generate-em-fixtures.cc \
//   -L/opt/homebrew/opt/opencascade/lib -Wl,-rpath,/opt/homebrew/opt/opencascade/lib \
//   -lTKPrim -lTKBO -lTKDESTEP -lTKXSBase -lTKTopAlgo -lTKBRep -lTKG3d -lTKMath -lTKernel \
//   -lTKXCAF -lTKLCAF -lTKCAF -lTKCDF -lTKDE -lTKService -lTKV3d -lTKFillet -lTKShHealing \
//   -o generate-em-fixtures
// Run: ./generate-em-fixtures <out-dir>/
#include <BRepAdaptor_Surface.hxx>
#include <BRepAlgoAPI_BuilderAlgo.hxx>
#include <BRepAlgoAPI_Common.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepBuilderAPI_Copy.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakePolygon.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepFilletAPI_MakeFillet.hxx>
#include <BRepGProp.hxx>
#include <BRepLProp_SLProps.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <BRepPrimAPI_MakeRevol.hxx>
#include <BRep_Builder.hxx>
#include <DESTEP_Parameters.hxx>
#include <GProp_GProps.hxx>
#include <Quantity_Color.hxx>
#include <STEPCAFControl_Reader.hxx>
#include <STEPCAFControl_Writer.hxx>
#include <TDF_LabelSequence.hxx>
#include <TDataStd_Name.hxx>
#include <TDocStd_Document.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <XCAFApp_Application.hxx>
#include <XCAFDoc_ColorTool.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>
#include <cmath>
#include <functional>
#include <gp_Ax1.hxx>
#include <gp_Ax2.hxx>
#include <gp_Trsf.hxx>
#include <iostream>
#include <string>
#include <vector>
#include <TCollection_AsciiString.hxx>
#include <algorithm>
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

// Volume of `fragment` inside `body`; fails loudly instead of reading a failed Boolean as empty.
bool common_volume(const TopoDS_Shape& fragment, const TopoDS_Shape& body, double& out) {
  BRepAlgoAPI_Common common(fragment, body);
  if (!common.IsDone() || common.HasErrors()) return false;
  out = volume_of(common.Shape());
  return true;
}

int count_solids(const TopoDS_Shape& s) {
  int n = 0;
  for (TopExp_Explorer e(s, TopAbs_SOLID); e.More(); e.Next()) ++n;
  return n;
}

// Re-read a written cell and confirm each named part holds exactly its solids and volume.
bool verify_written(const std::string& path, const std::vector<std::string>& names,
                    const std::vector<int>& solids, const std::vector<double>& volumes) {
  STEPCAFControl_Reader reader; reader.SetNameMode(true);
  if (reader.ReadFile(path.c_str()) != IFSelect_RetDone) { std::cerr << "re-read failed: " << path << '\n'; return false; }
  Handle(TDocStd_Document) doc; XCAFApp_Application::GetApplication()->NewDocument("MDTV-XCAF", doc);
  if (!reader.Transfer(doc)) { std::cerr << "re-transfer failed: " << path << '\n'; return false; }
  Handle(XCAFDoc_ShapeTool) shapes = XCAFDoc_DocumentTool::ShapeTool(doc->Main());
  TDF_LabelSequence free_shapes; shapes->GetFreeShapes(free_shapes);
  if (free_shapes.Length() != 1) { std::cerr << path << ": expected one root, found " << free_shapes.Length() << '\n'; return false; }
  int total = 0;
  for (size_t i = 0; i < names.size(); ++i) {
    TDF_LabelSequence components; shapes->GetComponents(free_shapes.Value(1), components);
    int found = 0;
    for (int c = 1; c <= components.Length(); ++c) {
      TDF_Label referred; if (!shapes->GetReferredShape(components.Value(c), referred)) continue;
      Handle(TDataStd_Name) name; if (!referred.FindAttribute(TDataStd_Name::GetID(), name)) continue;
      if (TCollection_AsciiString(name->Get()).ToCString() != names[i]) continue;
      ++found;
      const TopoDS_Shape part = shapes->GetShape(referred);
      const int n = count_solids(part); const double vol = volume_of(part);
      if (n != solids[i] || std::abs(vol - volumes[i]) > 1e-6 * std::max(1.0, volumes[i])) {
        std::cerr << path << ": part " << names[i] << " re-read as " << n << " solid(s), volume " << vol
                  << "; wrote " << solids[i] << ", " << volumes[i] << '\n';
        return false;
      }
      total += n;
    }
    if (found != 1) { std::cerr << path << ": part " << names[i] << " appears " << found << " times\n"; return false; }
  }
  if (total != count_solids(shapes->GetShape(free_shapes.Value(1)))) {
    std::cerr << path << ": root holds solids outside the named parts\n"; return false;
  }
  return true;
}

// Writes an assembly of named parts, the way a CAD tool exports touching bodies. The bodies are
// fragmented first (General Fuse) so each part's boundary is imprinted where it touches another
// part. Each fragment is owned by the smallest input body that contains it, so an enclosing air
// body keeps only the space outside the parts. Face rules name faces of each part.
bool write_cell(const std::string& path, std::vector<Body> bodies, const std::vector<FaceRule>& faces) {
  TopoDS_Shape cell = bodies.front().shape;
  if (bodies.size() > 1) {
    BRepAlgoAPI_BuilderAlgo fuse; TopTools_ListOfShape args;
    for (const auto& b : bodies) args.Append(b.shape);
    fuse.SetArguments(args); fuse.SetRunParallel(false); fuse.Build();
    if (!fuse.IsDone() || fuse.HasErrors()) { std::cerr << "fuse failed: " << path << '\n'; return false; }
    cell = fuse.Shape();
  }
  for (size_t i = 0; i < bodies.size(); ++i)
    for (size_t j = i + 1; j < bodies.size(); ++j)
      if (bodies[i].name == bodies[j].name) { std::cerr << path << ": duplicate part name " << bodies[i].name << '\n'; return false; }
  // Group fragments by owning body; every fragment lies wholly inside or wholly outside each body.
  std::vector<TopoDS_Compound> parts(bodies.size());
  std::vector<int> part_solids(bodies.size(), 0);
  std::vector<double> part_volumes(bodies.size(), 0.0);
  std::vector<std::pair<double, gp_Pnt>> seen;
  BRep_Builder builder;
  for (auto& part : parts) builder.MakeCompound(part);
  double cell_volume = 0;
  // The General Fuse result can reference one split solid from several places; visit each
  // solid once (by shared topology), or the same fragment would be written into a part twice.
  TopTools_IndexedMapOfShape unique_solids;
  TopExp::MapShapes(cell, TopAbs_SOLID, unique_solids);
  for (int index = 1; index <= unique_solids.Extent(); ++index) {
    const TopoDS_Shape solid = unique_solids(index);
    const double v = volume_of(solid);
    const gp_Pnt c = centroid(solid, true);
    if (v <= 0) { std::cerr << path << ": empty fragment\n"; return false; }
    for (const auto& [sv, sc] : seen)
      if (std::abs(sv - v) <= 1e-9 * v && sc.Distance(c) <= 1e-6) { std::cerr << path << ": duplicate fragment\n"; return false; }
    seen.emplace_back(v, c);
    cell_volume += v;
    int owner = -1; double owner_volume = 1e300;
    for (size_t i = 0; i < bodies.size(); ++i) {
      double inside = 0;
      if (!common_volume(solid, bodies[i].shape, inside)) { std::cerr << path << ": Boolean failed classifying a fragment\n"; return false; }
      const double fraction = inside / v;
      if (fraction > 1e-6 && fraction < 1 - 1e-6) { std::cerr << path << ": fragment straddles " << bodies[i].name << '\n'; return false; }
      if (fraction >= 1 - 1e-6) {
        const double bv = volume_of(bodies[i].shape);
        if (std::abs(bv - owner_volume) <= 1e-9 * bv) { std::cerr << path << ": fragment has two equal owners\n"; return false; }
        if (bv < owner_volume) { owner = static_cast<int>(i); owner_volume = bv; }
      }
    }
    if (owner < 0) { std::cerr << "unowned fragment in " << path << '\n'; return false; }
    builder.Add(parts[owner], solid); ++part_solids[owner]; part_volumes[owner] += v;
  }
  double owned = 0; for (double v : part_volumes) owned += v;
  if (std::abs(owned - cell_volume) > 1e-9 * cell_volume) { std::cerr << path << ": owned volume differs from the cell\n"; return false; }

  Handle(TDocStd_Document) doc; XCAFApp_Application::GetApplication()->NewDocument("MDTV-XCAF", doc);
  Handle(XCAFDoc_ShapeTool) shapes = XCAFDoc_DocumentTool::ShapeTool(doc->Main());
  Handle(XCAFDoc_ColorTool) colors = XCAFDoc_DocumentTool::ColorTool(doc->Main());
  TDF_Label assembly = shapes->NewShape();
  TDataStd_Name::Set(assembly, "Cell");
  std::vector<std::string> names;
  for (size_t i = 0; i < bodies.size(); ++i) {
    if (part_solids[i] == 0) { std::cerr << "empty part " << bodies[i].name << '\n'; return false; }
    names.push_back(bodies[i].name);
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
  if (!verify_written(path, names, part_solids, part_volumes)) return false;
  std::cout << "wrote " << path << " (" << bodies.size() << " parts, verified)\n";
  return true;
}

}  // namespace

namespace {

const Quantity_Color kCopper(0.85, 0.50, 0.25, Quantity_TOC_RGB);
const Quantity_Color kSteel(0.55, 0.57, 0.60, Quantity_TOC_RGB);
const Quantity_Color kIron(0.35, 0.36, 0.38, Quantity_TOC_RGB);
const Quantity_Color kAir(0.80, 0.90, 1.00, Quantity_TOC_RGB);
const Quantity_Color kEpoxy(0.70, 0.55, 0.20, Quantity_TOC_RGB);

// A spur gear: 24 trapezoid teeth on a 50 mm root circle, 60 mm tip, 12 mm face width, 20 mm bore.
TopoDS_Shape gear() {
  const int teeth = 24; const double root = 25, tip = 30, width = 12;
  BRepBuilderAPI_MakePolygon outline;
  for (int i = 0; i < teeth; ++i) {
    const double pitch = 2 * M_PI / teeth, a = i * pitch;
    const double angles[4] = {a - 0.30 * pitch, a - 0.17 * pitch, a + 0.17 * pitch, a + 0.30 * pitch};
    const double radii[4] = {root, tip, tip, root};
    for (int k = 0; k < 4; ++k) outline.Add(gp_Pnt(radii[k] * std::cos(angles[k]), radii[k] * std::sin(angles[k]), -width / 2));
  }
  outline.Close();
  TopoDS_Shape blank = BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(outline.Wire()).Face(), gp_Vec(0, 0, width)).Shape();
  TopoDS_Shape bore = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -width), gp_Dir(0, 0, 1)), 10, 2 * width).Shape();
  return BRepAlgoAPI_Cut(blank, bore).Shape();
}

// A single-turn rectangular-section copper inductor around the gear (38 to 46 mm radius,
// 10 mm tall) with a 4 mm gap; its two gap faces are the terminals. Built as a ring minus a slot,
// so both terminal faces come from one Boolean and carry no placement.
TopoDS_Shape coil() {
  const double r_in = 38, r_out = 46, h = 10;
  TopoDS_Shape outer = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -h / 2), gp_Dir(0, 0, 1)), r_out, h).Shape();
  TopoDS_Shape inner = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -h), gp_Dir(0, 0, 1)), r_in, 2 * h).Shape();
  TopoDS_Shape slot = BRepPrimAPI_MakeBox(gp_Pnt(r_in - 2, -2, -h), r_out - r_in + 4, 4, 2 * h).Shape();
  return BRepAlgoAPI_Cut(BRepAlgoAPI_Cut(outer, inner).Shape(), slot).Shape();
}

bool induction_cell(const std::string& dir) {
  TopoDS_Shape air = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -80), gp_Dir(0, 0, 1)), 120, 160).Shape();
  std::vector<FaceRule> faces = {
      {"Coil", "Coil terminal in", [](const gp_Pnt& c, const gp_Dir& n) { return c.X() > 30 && std::abs(c.Y()) < 3 && n.Y() < -0.9; }},
      {"Coil", "Coil terminal out", [](const gp_Pnt& c, const gp_Dir& n) { return c.X() > 30 && std::abs(c.Y()) < 3 && n.Y() > 0.9; }},
      {"Air", "Outer boundary", [](const gp_Pnt& c, const gp_Dir&) { return std::hypot(c.X(), c.Y()) > 100 || std::abs(c.Z()) > 79; }},
  };
  return write_cell(dir + "induction-gear.step",
                    {{gear(), "Gear", kSteel}, {coil(), "Coil", kCopper}, {air, "Air", kAir}}, faces);
}

// A Hall-effect current sensor: a 40 x 40 mm laminated iron core (8 mm square limbs, 15 mm deep)
// with a 2 mm air gap in one limb, around a 6 x 20 mm copper primary bar.
bool current_sensor(const std::string& dir) {
  TopoDS_Shape outer = BRepPrimAPI_MakeBox(gp_Pnt(-20, -20, -7.5), 40, 40, 15).Shape();
  TopoDS_Shape window = BRepPrimAPI_MakeBox(gp_Pnt(-12, -12, -10), 24, 24, 20).Shape();
  TopoDS_Shape gap = BRepPrimAPI_MakeBox(gp_Pnt(12, -1, -10), 8, 2, 20).Shape();
  TopoDS_Shape core = BRepAlgoAPI_Cut(BRepAlgoAPI_Cut(outer, window).Shape(), gap).Shape();
  TopoDS_Shape primary = BRepPrimAPI_MakeBox(gp_Pnt(-3, -10, -40), 6, 20, 80).Shape();
  TopoDS_Shape air = BRepPrimAPI_MakeBox(gp_Pnt(-60, -60, -40), 120, 120, 80).Shape();
  std::vector<FaceRule> faces = {
      {"Primary", "Primary in", [](const gp_Pnt& c, const gp_Dir&) { return c.Z() < -39.9; }},
      {"Primary", "Primary out", [](const gp_Pnt& c, const gp_Dir&) { return c.Z() > 39.9; }},
      {"Air", "Outer boundary", [](const gp_Pnt& c, const gp_Dir&) { return std::abs(c.X()) > 59.9 || std::abs(c.Y()) > 59.9 || std::abs(c.Z()) > 39.9; }},
  };
  return write_cell(dir + "current-sensor.step",
                    {{core, "Core", kIron}, {primary, "Primary", kCopper}, {air, "Air", kAir}}, faces);
}

// A cast-epoxy high-voltage bushing block, 80 x 40 x 40 mm, with a 30 x 10 x 10 mm electrode
// cavity; the electrode edges are sharp or rounded with `fillet` mm.
bool hv_electrode(const std::string& dir, double fillet, const std::string& file) {
  TopoDS_Shape block = BRepPrimAPI_MakeBox(gp_Pnt(-40, -20, -20), 80, 40, 40).Shape();
  TopoDS_Shape electrode = BRepPrimAPI_MakeBox(gp_Pnt(-15, -5, 0), 30, 10, 10).Shape();
  if (fillet > 0) {
    BRepFilletAPI_MakeFillet rounding(electrode);
    for (TopExp_Explorer e(electrode, TopAbs_EDGE); e.More(); e.Next()) rounding.Add(fillet, TopoDS::Edge(e.Current()));
    electrode = rounding.Shape();
  }
  TopoDS_Shape insulator = BRepAlgoAPI_Cut(block, electrode).Shape();
  std::vector<FaceRule> faces = {
      {"Insulator", "Electrode", [](const gp_Pnt& c, const gp_Dir&) { return std::abs(c.X()) < 15.1 && std::abs(c.Y()) < 5.1 && c.Z() > -0.1 && c.Z() < 10.1; }},
      {"Insulator", "Ground plane", [](const gp_Pnt& c, const gp_Dir&) { return c.Z() < -19.9; }},
  };
  return write_cell(dir + file, {{insulator, "Insulator", kEpoxy}}, faces);
}


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


// Parallel-plate strip line: a copper strip 10 mm wide, 20 mm long and 2 mm thick under a 1 mm
// air gap to a grounded return plane (the top face). Current flows along y.
bool strip_line(const std::string& dir) {
  std::vector<FaceRule> faces = {
      {"Strip", "Strip in", [](const gp_Pnt& c, const gp_Dir&) { return c.Y() < 0.01; }},
      {"Strip", "Strip out", [](const gp_Pnt& c, const gp_Dir&) { return c.Y() > 19.99; }},
      {"Strip", "Strip floor", [](const gp_Pnt& c, const gp_Dir&) { return c.Z() < 0.01; }},
      {"Air", "Air in", [](const gp_Pnt& c, const gp_Dir&) { return c.Y() < 0.01; }},
      {"Air", "Air out", [](const gp_Pnt& c, const gp_Dir&) { return c.Y() > 19.99; }},
      {"Air", "Return plane", [](const gp_Pnt& c, const gp_Dir&) { return c.Z() > 2.99; }},
  };
  return write_cell(dir + "strip-line.step",
                    {{BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 10, 20, 2).Shape(), "Strip", kCopper},
                     {BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 2), 10, 20, 1).Shape(), "Air", kAir}},
                    faces);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) { std::cerr << "usage: generate-em-fixtures <out-dir/>\n"; return 1; }
  const std::string dir = argv[1];
  bool ok = induction_cell(dir) && current_sensor(dir) && hv_electrode(dir, 0, "hv-electrode-sharp.step") &&
            hv_electrode(dir, 2, "hv-electrode-rounded.step") && coax(dir) && solenoid(dir) &&
            parallel_plate(dir) && coax_dielectric(dir) && wire(dir) && strip_line(dir);
  return ok ? 0 : 2;
}
