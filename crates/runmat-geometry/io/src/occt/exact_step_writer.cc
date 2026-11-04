// AP242 (or AP214) export for K1 through XCAF: product names, assembly structure with
// placements, part and face colours, face names and materials.
//
// Document encoding (one instruction per record):
//   op 1 header:     strs [name], ints [format: 0 AP242, 1 AP214]
//   op 2 part:       strs [name, material ("" none)], ints [body count, body indices into the
//                    call's bodies...], reals [has colour, r, g, b]
//   op 3 face name:  ints [part, body in part, face], strs [name]
//   op 4 face colour: ints [part, body in part, face], reals [r, g, b]
//   op 5 occurrence: strs [name], ints [part (-1: a sub-assembly), parent (-1: the top)],
//                    reals [transform: rows of R, then t]
// Lengths are written in metres (the kernel's unit).

#include "runmat-geometry-io/src/occt/kernel_ffi.rs.h"
#include "exact_kernel.hxx"

#include <BRep_Builder.hxx>
#include <Interface_Static.hxx>
#include <Quantity_Color.hxx>
#include <STEPCAFControl_Writer.hxx>
#include <STEPControl_Writer.hxx>
#include <Standard_Failure.hxx>
#include <TCollection_AsciiString.hxx>
#include <TCollection_ExtendedString.hxx>
#include <TCollection_HAsciiString.hxx>
#include <TDF_Label.hxx>
#include <TDataStd_Name.hxx>
#include <TDocStd_Document.hxx>
#include <TopLoc_Location.hxx>
#include <TopoDS_Compound.hxx>
#include <XCAFApp_Application.hxx>
#include <XCAFDoc_ColorTool.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_MaterialTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>

#include <mutex>
#include <sstream>

namespace runmat_geometry_io {
namespace occt_kernel {
namespace {

void set_name(const TDF_Label& l, const std::string& name) {
  TDataStd_Name::Set(l, TCollection_ExtendedString(name.c_str(), Standard_True));
}

// Restores a global STEP setting on scope exit.
struct StaticGuard {
  std::string key;
  std::string old;
  StaticGuard(const char* k, const char* v) : key(k) {
    const char* o = Interface_Static::CVal(k);
    old = o ? o : "";
    Interface_Static::SetCVal(k, v);
  }
  ~StaticGuard() { Interface_Static::SetCVal(key.c_str(), old.c_str()); }
};
}  // namespace

rust::Vec<std::uint8_t> kernel_export_step(rust::Slice<const KInstr> document,
                                           rust::Slice<const std::uint64_t> ids) {
  try {
    static std::mutex step_mutex;  // the writer's settings are process-global
    std::lock_guard<std::mutex> lock(step_mutex);
    std::vector<std::shared_ptr<const Body>> bodies;
    for (auto id : ids) bodies.push_back(registry_get(id));
    Handle(XCAFApp_Application) app = XCAFApp_Application::GetApplication();
    Handle(TDocStd_Document) doc;
    app->NewDocument("MDTV-XCAF", doc);
    Handle(XCAFDoc_ShapeTool) st = XCAFDoc_DocumentTool::ShapeTool(doc->Main());
    Handle(XCAFDoc_ColorTool) ct = XCAFDoc_DocumentTool::ColorTool(doc->Main());
    Handle(XCAFDoc_MaterialTool) mt = XCAFDoc_DocumentTool::MaterialTool(doc->Main());
    XCAFDoc_DocumentTool::SetLengthUnit(doc, 1.0);
    std::string name = "runmat";
    int format = 0;
    struct Part {
      TDF_Label label;
      std::vector<std::shared_ptr<const Body>> bodies;
    };
    std::vector<Part> parts;
    std::vector<TDF_Label> occurrences;
    std::vector<TDF_Label> assemblies;  // per occurrence: its assembly label when a sub-assembly
    bool any_occurrence = false;
    for (const KInstr& in : document) if (in.op == 5) any_occurrence = true;
    TDF_Label top;
    if (any_occurrence) top = st->NewShape();
    for (const KInstr& in : document) {
      switch (in.op) {
        case 1:
          if (!in.strs.empty()) name = std::string(in.strs[0]);
          if (!in.ints.empty()) format = static_cast<int>(in.ints[0]);
          if (any_occurrence) set_name(top, name);
          break;
        case 2: {
          Part p;
          const std::int64_t n = in.ints[0];
          BRep_Builder b;
          TopoDS_Shape shape;
          if (n == 1) {
            p.bodies.push_back(bodies.at(static_cast<std::size_t>(in.ints[1])));
            shape = p.bodies[0]->shape;
          } else {
            TopoDS_Compound c;
            b.MakeCompound(c);
            for (std::int64_t i = 0; i < n; ++i) {
              p.bodies.push_back(bodies.at(static_cast<std::size_t>(in.ints[1 + i])));
              b.Add(c, p.bodies.back()->shape);
            }
            shape = c;
          }
          p.label = st->AddShape(shape, Standard_False);
          set_name(p.label, in.strs.empty() ? "part" : std::string(in.strs[0]));
          if (in.reals.size() >= 4 && in.reals[0] != 0.0)
            ct->SetColor(p.label, Quantity_Color(in.reals[1], in.reals[2], in.reals[3], Quantity_TOC_RGB), XCAFDoc_ColorSurf);
          if (in.strs.size() > 1 && !std::string(in.strs[1]).empty()) {
            Handle(TCollection_HAsciiString) mname = new TCollection_HAsciiString(std::string(in.strs[1]).c_str());
            Handle(TCollection_HAsciiString) empty = new TCollection_HAsciiString("");
            TDF_Label ml = mt->AddMaterial(mname, empty, 0.0, empty, empty);
            mt->SetMaterial(p.label, ml);
          }
          parts.push_back(p);
          break;
        }
        case 3:
        case 4: {
          const Part& p = parts.at(static_cast<std::size_t>(in.ints[0]));
          const Body& b = *p.bodies.at(static_cast<std::size_t>(in.ints[1]));
          const TopoDS_Face& f = b.faces.at(static_cast<std::size_t>(in.ints[2]));
          TDF_Label sub = st->AddSubShape(p.label, f);
          if (sub.IsNull()) fail("a named face is not part of its body");
          if (in.op == 3) set_name(sub, std::string(in.strs.at(0)));
          else ct->SetColor(sub, Quantity_Color(in.reals[0], in.reals[1], in.reals[2], Quantity_TOC_RGB), XCAFDoc_ColorSurf);
          break;
        }
        case 5: {
          const std::int64_t part = in.ints[0];
          const std::int64_t parent = in.ints[1];
          TDF_Label into = parent < 0 ? top : assemblies.at(static_cast<std::size_t>(parent));
          if (into.IsNull()) fail("an occurrence's parent is not a sub-assembly");
          TDF_Label what;
          TDF_Label own_assembly;
          if (part < 0) {
            own_assembly = st->NewShape();
            set_name(own_assembly, std::string(in.strs.at(0)));
            what = own_assembly;
          } else {
            what = parts.at(static_cast<std::size_t>(part)).label;
          }
          TopLoc_Location loc(trsf_from(in.reals.data()));
          TDF_Label comp = st->AddComponent(into, what, loc);
          if (comp.IsNull()) fail("an occurrence could not be placed");
          set_name(comp, std::string(in.strs.at(0)));
          occurrences.push_back(comp);
          assemblies.push_back(own_assembly);
          break;
        }
        default:
          fail("unknown export record");
      }
    }
    st->UpdateAssemblies();
    StaticGuard schema("write.step.schema", format == 1 ? "AP214IS" : "AP242DIS");
    StaticGuard unit("write.step.unit", "M");
    StaticGuard product("write.step.product.name", name.c_str());
    STEPCAFControl_Writer w;
    w.SetNameMode(Standard_True);
    w.SetColorMode(Standard_True);
    w.SetLayerMode(Standard_True);
    w.SetMaterialMode(Standard_True);
    if (!w.Transfer(doc, STEPControl_AsIs)) fail("the document could not be written as STEP");
    std::ostringstream os;
    if (w.ChangeWriter().WriteStream(os) != IFSelect_RetDone) fail("the STEP stream could not be written");
    const std::string bytes = os.str();
    rust::Vec<std::uint8_t> out;
    out.reserve(bytes.size());
    for (unsigned char c : bytes) out.push_back(c);
    return out;
  } catch (const Standard_Failure& e) {
    fail(std::string("OCCT raised ") + e.GetMessageString());
  }
}

}  // namespace occt_kernel
}  // namespace runmat_geometry_io
