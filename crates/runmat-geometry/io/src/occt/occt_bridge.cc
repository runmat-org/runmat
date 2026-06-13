#include "runmat-geometry-io/src/occt/ffi.rs.h"

#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepTools.hxx>
#include <IGESControl_Reader.hxx>
#include <IGESCAFControl_Reader.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <IMeshTools_Parameters.hxx>
#include <Message_ProgressIndicator.hxx>
#include <Message_ProgressRange.hxx>
#include <Poly_Triangle.hxx>
#include <Poly_Triangulation.hxx>
#include <Quantity_Color.hxx>
#include <STEPCAFControl_Reader.hxx>
#include <STEPControl_Reader.hxx>
#include <TCollection_AsciiString.hxx>
#include <TCollection_ExtendedString.hxx>
#include <TCollection_HAsciiString.hxx>
#include <TColStd_HSequenceOfExtendedString.hxx>
#include <TDF_Label.hxx>
#include <TDF_LabelSequence.hxx>
#include <TDF_Tool.hxx>
#include <TDataStd_Name.hxx>
#include <TDataStd_TreeNode.hxx>
#include <TDocStd_Document.hxx>
#include <TopAbs_Orientation.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp_Explorer.hxx>
#include <TopLoc_Location.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>
#include <XCAFApp_Application.hxx>
#include <XCAFDoc.hxx>
#include <XCAFDoc_ColorTool.hxx>
#include <XCAFDoc_ColorType.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_LayerTool.hxx>
#include <XCAFDoc_MaterialTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>
#include <gp_Pnt.hxx>
#include <gp_Trsf.hxx>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace runmat_geometry_io {
namespace occt_backend {
namespace {

std::string str_from_rust(rust::Str value) {
  return std::string(value.data(), value.size());
}

std::string bytes_to_string(rust::Slice<const std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

std::string format_name(OcctCadFormat format) {
  switch (format) {
    case OcctCadFormat::Step:
      return "step";
    case OcctCadFormat::Iges:
      return "iges";
    case OcctCadFormat::Brep:
      return "brep";
  }
  throw std::runtime_error("unsupported OCCT CAD format");
}

struct CadDocument {
  TopoDS_Shape shape;
  Handle(TDocStd_Document) document;
  Handle(XCAFDoc_ShapeTool) shape_tool;
  Handle(XCAFDoc_ColorTool) color_tool;
  Handle(XCAFDoc_LayerTool) layer_tool;
  Handle(XCAFDoc_MaterialTool) material_tool;
  bool has_xcaf = false;
};

void check_cancelled(const OcctImportOptions& options) {
  if (options.cancel_token_id != 0 && occt_import_cancelled(options.cancel_token_id)) {
    throw std::runtime_error("OCCT CAD import cancelled");
  }
}

class RunmatCancelProgressIndicator final : public Message_ProgressIndicator {
public:
  explicit RunmatCancelProgressIndicator(const OcctImportOptions& options)
    : options_(options) {}

protected:
  Standard_Boolean UserBreak() override {
    return options_.cancel_token_id != 0 && occt_import_cancelled(options_.cancel_token_id);
  }

  void Show(const Message_ProgressScope&, const Standard_Boolean) override {}

private:
  OcctImportOptions options_;
};

Handle(TDocStd_Document) new_xcaf_document() {
  Handle(XCAFApp_Application) app = XCAFApp_Application::GetApplication();
  Handle(TDocStd_Document) document;
  app->NewDocument("MDTV-XCAF", document);
  if (document.IsNull()) {
    throw std::runtime_error("OCCT XCAF application returned a null document");
  }
  return document;
}

CadDocument finish_xcaf_document(const Handle(TDocStd_Document)& document) {
  CadDocument result;
  result.document = document;
  result.shape_tool = XCAFDoc_DocumentTool::ShapeTool(document->Main());
  result.color_tool = XCAFDoc_DocumentTool::ColorTool(document->Main());
  result.layer_tool = XCAFDoc_DocumentTool::LayerTool(document->Main());
  result.material_tool = XCAFDoc_DocumentTool::MaterialTool(document->Main());
  result.has_xcaf = !result.shape_tool.IsNull();

  TDF_LabelSequence free_shapes;
  result.shape_tool->GetFreeShapes(free_shapes);
  if (free_shapes.Length() <= 0) {
    throw std::runtime_error("OCCT XCAF reader produced no free shapes");
  }
  result.shape = XCAFDoc_ShapeTool::GetOneShape(free_shapes);
  if (result.shape.IsNull()) {
    throw std::runtime_error("OCCT XCAF reader returned a null shape");
  }
  return result;
}

CadDocument read_step_shape(const std::string& path,
                            std::istream& stream,
                            const OcctImportOptions& options) {
  check_cancelled(options);
  STEPCAFControl_Reader reader;
  reader.SetNameMode(Standard_True);
  reader.SetColorMode(Standard_True);
  reader.SetLayerMode(Standard_True);
  reader.SetPropsMode(Standard_True);
  reader.SetMatMode(Standard_True);
  IFSelect_ReturnStatus status = reader.ReadStream(path.c_str(), stream);
  check_cancelled(options);
  if (status != IFSelect_RetDone) {
    throw std::runtime_error("OCCT STEP reader did not return IFSelect_RetDone");
  }
  Handle(TDocStd_Document) document = new_xcaf_document();
  RunmatCancelProgressIndicator transfer_progress(options);
  if (!reader.Transfer(document, transfer_progress.Start())) {
    check_cancelled(options);
    throw std::runtime_error("OCCT STEP XCAF transfer failed");
  }
  check_cancelled(options);
  return finish_xcaf_document(document);
}

CadDocument read_iges_shape(const std::string& path,
                            std::istream& stream,
                            const OcctImportOptions& options) {
  check_cancelled(options);
  IGESCAFControl_Reader reader;
  reader.SetNameMode(Standard_True);
  reader.SetColorMode(Standard_True);
  reader.SetLayerMode(Standard_True);
  IFSelect_ReturnStatus status = reader.ReadStream(path.c_str(), stream);
  check_cancelled(options);
  if (status != IFSelect_RetDone) {
    throw std::runtime_error("OCCT IGES reader did not return IFSelect_RetDone");
  }
  Handle(TDocStd_Document) document = new_xcaf_document();
  RunmatCancelProgressIndicator transfer_progress(options);
  if (!reader.Transfer(document, transfer_progress.Start())) {
    check_cancelled(options);
    throw std::runtime_error("OCCT IGES XCAF transfer failed");
  }
  check_cancelled(options);
  return finish_xcaf_document(document);
}

CadDocument read_brep_shape(std::istream& stream, const OcctImportOptions& options) {
  check_cancelled(options);
  TopoDS_Shape shape;
  BRep_Builder builder;
  RunmatCancelProgressIndicator read_progress(options);
  BRepTools::Read(shape, stream, builder, read_progress.Start());
  check_cancelled(options);
  if (shape.IsNull()) {
    throw std::runtime_error("OCCT BREP reader returned a null shape");
  }
  CadDocument result;
  result.shape = shape;
  return result;
}

CadDocument read_shape(const std::string& path,
                       rust::Slice<const std::uint8_t> bytes,
                       OcctCadFormat format,
                       const OcctImportOptions& options) {
  std::string payload = bytes_to_string(bytes);
  std::istringstream stream(payload);
  switch (format) {
    case OcctCadFormat::Step:
      return read_step_shape(path, stream, options);
    case OcctCadFormat::Iges:
      return read_iges_shape(path, stream, options);
    case OcctCadFormat::Brep:
      return read_brep_shape(stream, options);
  }
  throw std::runtime_error("unsupported OCCT CAD format");
}

std::string face_name(std::uint64_t face_id) {
  return "Face " + std::to_string(face_id + 1);
}

std::string ascii_string(const TCollection_AsciiString& value) {
  return std::string(value.ToCString());
}

std::string extended_string(const TCollection_ExtendedString& value) {
  TCollection_AsciiString ascii(value, '?');
  return ascii_string(ascii);
}

std::string label_entry(const TDF_Label& label) {
  if (label.IsNull()) {
    return "";
  }
  TCollection_AsciiString entry;
  TDF_Tool::Entry(label, entry);
  return ascii_string(entry);
}

std::string label_name(const TDF_Label& label) {
  if (label.IsNull()) {
    return "";
  }
  Handle(TDataStd_Name) name;
  if (!label.FindAttribute(TDataStd_Name::GetID(), name) || name.IsNull()) {
    return "";
  }
  return extended_string(name->Get());
}

std::string label_display_name(const TDF_Label& label) {
  std::string name = label_name(label);
  if (!name.empty()) {
    return name;
  }
  std::string entry = label_entry(label);
  if (!entry.empty()) {
    return "CAD " + entry;
  }
  return "Unnamed CAD Node";
}

std::string label_kind(const TDF_Label& label) {
  if (label.IsNull()) {
    return "unknown";
  }
  if (XCAFDoc_ShapeTool::IsAssembly(label)) {
    return "assembly";
  }
  if (XCAFDoc_ShapeTool::IsComponent(label)) {
    return "component";
  }
  if (XCAFDoc_ShapeTool::IsReference(label)) {
    return "reference";
  }
  if (XCAFDoc_ShapeTool::IsSubShape(label)) {
    TopoDS_Shape shape = XCAFDoc_ShapeTool::GetShape(label);
    if (!shape.IsNull() && shape.ShapeType() == TopAbs_FACE) {
      return "face";
    }
    return "subshape";
  }
  if (XCAFDoc_ShapeTool::IsSimpleShape(label)) {
    return "body";
  }
  if (XCAFDoc_ShapeTool::IsCompound(label)) {
    return "compound";
  }
  if (XCAFDoc_ShapeTool::IsShape(label)) {
    return "shape";
  }
  return "unknown";
}

bool label_has_name(const TDF_Label& label) {
  Handle(TDataStd_Name) name;
  return !label.IsNull() && label.FindAttribute(TDataStd_Name::GetID(), name) && !name.IsNull() &&
         !extended_string(name->Get()).empty();
}

bool label_has_material(const TDF_Label& label) {
  Handle(TDataStd_TreeNode) node;
  return !label.IsNull() && label.FindAttribute(XCAFDoc::MaterialRefGUID(), node) &&
         !node.IsNull() && node->HasFather();
}

bool label_has_semantic_value(const CadDocument& document, const TDF_Label& label) {
  if (label.IsNull()) {
    return false;
  }
  if (label_has_name(label) || label_has_material(label)) {
    return true;
  }
  Quantity_Color color;
  if (XCAFDoc_ColorTool::GetColor(label, XCAFDoc_ColorSurf, color) ||
      XCAFDoc_ColorTool::GetColor(label, XCAFDoc_ColorGen, color) ||
      XCAFDoc_ColorTool::GetColor(label, XCAFDoc_ColorCurv, color)) {
    return true;
  }
  if (!document.layer_tool.IsNull()) {
    Handle(TColStd_HSequenceOfExtendedString) layers = document.layer_tool->GetLayers(label);
    if (!layers.IsNull() && layers->Length() > 0) {
      return true;
    }
  }
  return false;
}

std::vector<TDF_Label> semantic_owner_path(const CadDocument& document, TDF_Label label) {
  std::vector<TDF_Label> reverse_path;
  for (TDF_Label current = label; !current.IsNull() && !current.IsRoot();
       current = current.Father()) {
    if (XCAFDoc_ShapeTool::IsShape(current) || label_has_semantic_value(document, current)) {
      reverse_path.push_back(current);
    }
  }
  std::reverse(reverse_path.begin(), reverse_path.end());
  return reverse_path;
}

void append_component_owner_path(const CadDocument& document,
                                 const TopoDS_Shape& shape,
                                 std::vector<TDF_Label>& owners) {
  if (!document.has_xcaf || document.shape_tool.IsNull()) {
    return;
  }
  std::set<std::string> seen;
  for (const TDF_Label& owner : owners) {
    const std::string entry = label_entry(owner);
    if (!entry.empty()) {
      seen.insert(entry);
    }
  }

  TDF_LabelSequence component_path;
  if (!document.shape_tool->FindComponent(shape, component_path)) {
    return;
  }
  for (Standard_Integer index = 1; index <= component_path.Length(); ++index) {
    const TDF_Label label = component_path.Value(index);
    const std::string entry = label_entry(label);
    if (!entry.empty() && seen.insert(entry).second) {
      owners.push_back(label);
    }
  }
}

void append_label_ref(OcctFaceSemanticPayload& semantic, const TDF_Label& label) {
  const std::string entry = label_entry(label);
  if (entry.empty()) {
    return;
  }
  semantic.owner_entries.push_back(entry);
  semantic.owner_names.push_back(label_name(label));
  semantic.owner_kinds.push_back(label_kind(label));
}

void append_layers(OcctFaceSemanticPayload& semantic,
                   const CadDocument& document,
                   const std::vector<TDF_Label>& labels) {
  if (document.layer_tool.IsNull()) {
    return;
  }
  std::set<std::string> seen;
  for (auto it = labels.rbegin(); it != labels.rend(); ++it) {
    Handle(TColStd_HSequenceOfExtendedString) layers = document.layer_tool->GetLayers(*it);
    if (layers.IsNull()) {
      continue;
    }
    for (Standard_Integer index = 1; index <= layers->Length(); ++index) {
      const std::string name = extended_string(layers->Value(index));
      if (!name.empty() && seen.insert(name).second) {
        semantic.layer_names.push_back(name);
      }
    }
  }
}

bool find_color_on_label(const TDF_Label& label, Quantity_Color& color, std::string& color_type) {
  if (XCAFDoc_ColorTool::GetColor(label, XCAFDoc_ColorSurf, color)) {
    color_type = "surface";
    return true;
  }
  if (XCAFDoc_ColorTool::GetColor(label, XCAFDoc_ColorGen, color)) {
    color_type = "generic";
    return true;
  }
  if (XCAFDoc_ColorTool::GetColor(label, XCAFDoc_ColorCurv, color)) {
    color_type = "curve";
    return true;
  }
  return false;
}

void append_color(OcctFaceSemanticPayload& semantic, const std::vector<TDF_Label>& labels) {
  Quantity_Color color;
  std::string color_type;
  for (auto it = labels.rbegin(); it != labels.rend(); ++it) {
    if (find_color_on_label(*it, color, color_type)) {
      semantic.color_type = color_type;
      semantic.color_hex_rgba = ascii_string(Quantity_Color::ColorToHex(color)) + "FF";
      return;
    }
  }
}

std::string real_string(Standard_Real value) {
  std::ostringstream stream;
  stream << std::setprecision(17) << value;
  return stream.str();
}

void append_material(OcctFaceSemanticPayload& semantic, const std::vector<TDF_Label>& labels) {
  for (auto it = labels.rbegin(); it != labels.rend(); ++it) {
    Handle(TDataStd_TreeNode) node;
    if (!it->FindAttribute(XCAFDoc::MaterialRefGUID(), node) || node.IsNull() ||
        !node->HasFather()) {
      continue;
    }
    const TDF_Label material_label = node->Father()->Label();
    Handle(TCollection_HAsciiString) name;
    Handle(TCollection_HAsciiString) description;
    Handle(TCollection_HAsciiString) density_name;
    Handle(TCollection_HAsciiString) density_value_type;
    Standard_Real density = 0.0;
    if (!XCAFDoc_MaterialTool::GetMaterial(
            material_label, name, description, density, density_name, density_value_type)) {
      continue;
    }
    semantic.material_label_entry = label_entry(material_label);
    semantic.material_name = name.IsNull() ? "" : std::string(name->ToCString());
    semantic.material_description =
        description.IsNull() ? "" : std::string(description->ToCString());
    semantic.material_density = density == 0.0 ? "" : real_string(density);
    semantic.material_density_name =
        density_name.IsNull() ? "" : std::string(density_name->ToCString());
    semantic.material_density_value_type =
        density_value_type.IsNull() ? "" : std::string(density_value_type->ToCString());
    return;
  }
}

TDF_Label face_label(const CadDocument& document, const TopoDS_Face& face) {
  if (!document.has_xcaf || document.shape_tool.IsNull()) {
    return TDF_Label();
  }
  TDF_Label label;
  if (document.shape_tool->Search(face, label, Standard_True, Standard_True, Standard_True) &&
      !label.IsNull()) {
    return label;
  }
  label = document.shape_tool->FindMainShape(face);
  return label;
}

OcctFaceSemanticPayload face_semantics(const CadDocument& document,
                                       const TopoDS_Face& face,
                                       std::uint64_t face_id) {
  OcctFaceSemanticPayload semantic;
  semantic.face_id = face_id;
  TDF_Label label = face_label(document, face);
  if (label.IsNull()) {
    return semantic;
  }

  semantic.label_entry = label_entry(label);
  semantic.label_name = label_name(label);
  semantic.label_kind = label_kind(label);
  std::vector<TDF_Label> owners = semantic_owner_path(document, label);
  if (owners.empty()) {
    owners.push_back(label);
  }
  append_component_owner_path(document, face, owners);
  for (const TDF_Label& owner : owners) {
    append_label_ref(semantic, owner);
  }
  append_layers(semantic, document, owners);
  append_color(semantic, owners);
  append_material(semantic, owners);
  return semantic;
}

bool has_face_semantics(const OcctFaceSemanticPayload& semantic) {
  return !semantic.label_entry.empty() || !semantic.owner_entries.empty() ||
         !semantic.layer_names.empty() || !semantic.color_hex_rgba.empty() ||
         !semantic.material_name.empty();
}

void append_assembly_label(OcctImportPayload& result,
                           const Handle(XCAFDoc_ShapeTool)& shape_tool,
                           const TDF_Label& label,
                           const std::string& parent_node_id,
                           std::set<std::string>& active_path,
                           std::uint32_t depth,
                           const OcctImportOptions& options) {
  check_cancelled(options);
  if (label.IsNull() || depth > 256) {
    return;
  }
  const std::string node_id = label_entry(label);
  if (node_id.empty() || active_path.count(node_id) != 0) {
    return;
  }
  active_path.insert(node_id);

  OcctAssemblyNodePayload node;
  node.node_id = node_id;
  node.parent_node_id = parent_node_id;
  node.label = label_display_name(label);
  result.assembly_nodes.push_back(node);

  TDF_Label child_owner = label;
  TDF_Label referred;
  if (XCAFDoc_ShapeTool::IsReference(label) &&
      XCAFDoc_ShapeTool::GetReferredShape(label, referred) && !referred.IsNull()) {
    child_owner = referred;
  }
  TDF_LabelSequence components;
  if (XCAFDoc_ShapeTool::IsAssembly(child_owner) &&
      XCAFDoc_ShapeTool::GetComponents(child_owner, components, Standard_False)) {
    for (Standard_Integer index = 1; index <= components.Length(); ++index) {
      check_cancelled(options);
      append_assembly_label(
          result, shape_tool, components.Value(index), node_id, active_path, depth + 1, options);
    }
  }

  active_path.erase(node_id);
}

void append_assembly_tree(OcctImportPayload& result,
                          const CadDocument& document,
                          const OcctImportOptions& options) {
  if (!document.has_xcaf || document.shape_tool.IsNull()) {
    return;
  }
  TDF_LabelSequence free_shapes;
  document.shape_tool->GetFreeShapes(free_shapes);
  std::set<std::string> active_path;
  for (Standard_Integer index = 1; index <= free_shapes.Length(); ++index) {
    check_cancelled(options);
    append_assembly_label(
        result, document.shape_tool, free_shapes.Value(index), "", active_path, 0, options);
  }
}

void push_triangle(OcctImportPayload& result,
                   std::uint32_t a,
                   std::uint32_t b,
                   std::uint32_t c,
                   std::uint64_t face_id) {
  result.triangles.push_back(a);
  result.triangles.push_back(b);
  result.triangles.push_back(c);
  result.triangle_face_ids.push_back(face_id);
}

void append_face_mesh(OcctImportPayload& result,
                      const TopoDS_Face& face,
                      std::uint64_t face_id,
                      const std::string& resolved_face_name,
                      const OcctImportOptions& options) {
  check_cancelled(options);
  TopLoc_Location location;
  Handle(Poly_Triangulation) triangulation = BRep_Tool::Triangulation(face, location);
  result.face_ids.push_back(face_id);
  result.face_names.push_back(resolved_face_name.empty() ? face_name(face_id) : resolved_face_name);

  if (static_cast<std::uint64_t>(result.triangle_face_ids.size()) >= options.max_triangles &&
      options.truncate_at_max_triangles) {
    result.truncated = true;
    return;
  }

  if (triangulation.IsNull()) {
    result.warnings.push_back("OCCT face " + std::to_string(face_id + 1) +
                              " had no triangulation after meshing");
    return;
  }

  const Standard_Integer triangle_count = triangulation->NbTriangles();
  const std::uint64_t current_count =
      static_cast<std::uint64_t>(result.triangle_face_ids.size());
  if (current_count >= options.max_triangles) {
    if (options.truncate_at_max_triangles) {
      result.truncated = true;
      return;
    }
    throw std::runtime_error("OCCT tessellation exceeded max_triangles");
  }

  const std::uint64_t remaining =
      options.max_triangles == std::numeric_limits<std::uint64_t>::max()
          ? std::numeric_limits<std::uint64_t>::max()
          : options.max_triangles - current_count;
  std::uint64_t emit_count = static_cast<std::uint64_t>(triangle_count);
  if (emit_count > remaining) {
    if (options.truncate_at_max_triangles) {
      emit_count = remaining;
      result.truncated = true;
    } else {
      throw std::runtime_error("OCCT tessellation exceeded max_triangles");
    }
  }
  if (emit_count == 0) {
    return;
  }

  std::vector<Poly_Triangle> triangles;
  triangles.reserve(static_cast<std::size_t>(emit_count));
  std::vector<Standard_Integer> node_ids;
  node_ids.reserve(static_cast<std::size_t>(emit_count) * 3);
  for (Standard_Integer i = 1; i <= triangle_count; ++i) {
    if ((i & 1023) == 1) {
      check_cancelled(options);
    }
    if (static_cast<std::uint64_t>(triangles.size()) >= emit_count) {
      break;
    }
    const Poly_Triangle triangle = triangulation->Triangle(i);
    triangles.push_back(triangle);
    node_ids.push_back(triangle.Value(1));
    node_ids.push_back(triangle.Value(2));
    node_ids.push_back(triangle.Value(3));
  }

  std::sort(node_ids.begin(), node_ids.end());
  node_ids.erase(std::unique(node_ids.begin(), node_ids.end()), node_ids.end());

  const gp_Trsf transform = location.Transformation();
  std::unordered_map<Standard_Integer, std::uint32_t> global_index_by_local_node;
  global_index_by_local_node.reserve(node_ids.size());
  for (std::size_t index = 0; index < node_ids.size(); ++index) {
    if ((index & 1023) == 0) {
      check_cancelled(options);
    }
    const Standard_Integer node_id = node_ids[index];
    if (node_id <= 0 || node_id > triangulation->NbNodes()) {
      throw std::runtime_error("OCCT returned a triangle node index outside the node buffer");
    }
    const std::size_t global_vertex_offset = result.vertices.size() / 3;
    if (global_vertex_offset >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
      throw std::runtime_error("OCCT tessellation exceeded u32 vertex indexing capacity");
    }
    gp_Pnt point = triangulation->Node(node_id);
    point.Transform(transform);
    global_index_by_local_node.emplace(node_id, static_cast<std::uint32_t>(global_vertex_offset));
    result.vertices.push_back(point.X());
    result.vertices.push_back(point.Y());
    result.vertices.push_back(point.Z());
  }

  const bool forward = face.Orientation() == TopAbs_FORWARD;
  for (std::size_t index = 0; index < triangles.size(); ++index) {
    if ((index & 1023) == 0) {
      check_cancelled(options);
    }
    const Poly_Triangle& triangle = triangles[index];
    const std::uint32_t n1 = global_index_by_local_node.at(triangle.Value(1));
    const std::uint32_t n2 = global_index_by_local_node.at(triangle.Value(2));
    const std::uint32_t n3 = global_index_by_local_node.at(triangle.Value(3));
    if (forward) {
      push_triangle(result, n1, n2, n3, face_id);
    } else {
      push_triangle(result, n3, n2, n1, face_id);
    }
  }
}

} // namespace

OcctImportPayload import_cad_bytes(rust::Str path,
                                   rust::Slice<const std::uint8_t> bytes,
                                   OcctCadFormat format,
                                   OcctImportOptions options) {
  const std::string path_string = str_from_rust(path);
  check_cancelled(options);
  CadDocument document = read_shape(path_string, bytes, format, options);
  check_cancelled(options);

  IMeshTools_Parameters mesh_parameters;
  mesh_parameters.Deflection = options.linear_deflection;
  mesh_parameters.Relative = options.relative_deflection;
  mesh_parameters.Angle = options.angular_deflection;
  mesh_parameters.InParallel = Standard_False;
  if (!options.truncate_at_max_triangles) {
    RunmatCancelProgressIndicator mesh_progress(options);
    BRepMesh_IncrementalMesh mesh(document.shape,
                                  mesh_parameters,
                                  mesh_progress.Start());
    check_cancelled(options);
    if (!mesh.IsDone()) {
      throw std::runtime_error("OCCT tessellation did not complete");
    }
  }

  OcctImportPayload result;
  result.backend = "occt-native";
  result.format_name = format_name(format);
  result.truncated = false;
  result.triangle_budget = options.max_triangles;
  append_assembly_tree(result, document, options);

  std::uint64_t face_id = 0;
  for (TopExp_Explorer explorer(document.shape, TopAbs_FACE); explorer.More(); explorer.Next()) {
    check_cancelled(options);
    const TopoDS_Face face = TopoDS::Face(explorer.Current());
    OcctFaceSemanticPayload semantic = face_semantics(document, face, face_id);
    const std::string resolved_face_name = semantic.label_name.empty()
                                               ? face_name(face_id)
                                               : static_cast<std::string>(semantic.label_name);
    if (options.truncate_at_max_triangles &&
        static_cast<std::uint64_t>(result.triangle_face_ids.size()) < options.max_triangles) {
      RunmatCancelProgressIndicator face_mesh_progress(options);
      BRepMesh_IncrementalMesh face_mesh(face,
                                         mesh_parameters,
                                         face_mesh_progress.Start());
      check_cancelled(options);
      if (!face_mesh.IsDone()) {
        result.warnings.push_back("OCCT preview tessellation did not complete for face " +
                                  std::to_string(face_id + 1));
      }
    }
    append_face_mesh(result, face, face_id, resolved_face_name, options);
    if (has_face_semantics(semantic)) {
      result.face_semantics.push_back(semantic);
    }
    ++face_id;
  }

  if (face_id == 0) {
    throw std::runtime_error("OCCT import produced no faces");
  }
  if (result.triangle_face_ids.empty() && !result.truncated) {
    throw std::runtime_error("OCCT import produced no tessellated triangles");
  }
  if (result.truncated) {
    result.warnings.push_back(
        "OCCT preview tessellation was truncated at the requested triangle budget of " +
        std::to_string(options.max_triangles) + " triangles");
  }

  return result;
}

} // namespace occt_backend
} // namespace runmat_geometry_io
