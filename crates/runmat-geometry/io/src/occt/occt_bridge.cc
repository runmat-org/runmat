#include "runmat-geometry-io/src/occt/ffi.rs.h"
#include "exact_assembly.hxx"
#include "exact_healing.hxx"
#include "exact_shared_interfaces.hxx"
#include "exact_names.hxx"
#include "exact_xcaf_healing.hxx"

#include <BRep_Builder.hxx>
#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepClass3d.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepGProp.hxx>
#include <GProp_GProps.hxx>
#include <BRep_Tool.hxx>
#include <BRepLib_CheckCurveOnSurface.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepTools.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <gp_Trsf.hxx>
#include <BRepTools_WireExplorer.hxx>
#include <Geom2d_Curve.hxx>
#include <IGESControl_Reader.hxx>
#include <IGESCAFControl_Reader.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <IMeshTools_Parameters.hxx>
#include <Message_ProgressIndicator.hxx>
#include <Message_ProgressRange.hxx>
#include <Standard_Version.hxx>
#include <Poly_Triangle.hxx>
#include <Poly_Triangulation.hxx>
#include <Quantity_Color.hxx>
#include <ShapeAnalysis_Edge.hxx>
#include <TopoDS_Compound.hxx>
#include <StepShape_AdvancedFace.hxx>
#include <StepShape_EdgeCurve.hxx>
#include <StepShape_TopologicalRepresentationItem.hxx>
#include <StepShape_VertexPoint.hxx>
#include <TransferBRep.hxx>
#include <XSControl_TransferReader.hxx>
#include <Transfer_TransientProcess.hxx>
#include <Interface_InterfaceModel.hxx>
#include <STEPCAFControl_Reader.hxx>
#include <StepData_StepModel.hxx>
#include <XSControl_Reader.hxx>
#include <XSControl_WorkSession.hxx>
#include <STEPControl_Reader.hxx>
#include <STEPConstruct_UnitContext.hxx>
#include <StepData_Factors.hxx>
#include <StepData_StepModel.hxx>
#include <StepRepr_GlobalUnitAssignedContext.hxx>
#include <StepGeom_GeometricRepresentationContextAndGlobalUnitAssignedContext.hxx>
#include <StepGeom_GeomRepContextAndGlobUnitAssCtxAndGlobUncertaintyAssCtx.hxx>
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
#include <TopExp.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopLoc_Location.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Iterator.hxx>
#include <TopoDS_Shell.hxx>
#include <TopoDS_Solid.hxx>
#include <TopoDS_Vertex.hxx>
#include <TopoDS_Wire.hxx>
#include <TopoDS_Shape.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <XCAFApp_Application.hxx>
#include <XCAFDoc.hxx>
#include <XCAFDoc_ColorTool.hxx>
#include <XCAFDoc_ColorType.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_LayerTool.hxx>
#include <XCAFDoc_MaterialTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>
#include <gp_Pnt.hxx>
#include <gp_Mat.hxx>
#include <gp_Vec.hxx>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace runmat_geometry_io {
namespace occt_backend {

rust::String exact_kernel_abi() {
  return rust::String(std::string("occt/") + OCC_VERSION_COMPLETE + "/brep-v3");
}
namespace {

std::string str_from_rust(rust::Str value) {
  return std::string(value.data(), value.size());
}

std::string bytes_to_string(rust::Slice<const std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

std::array<std::uint8_t, 32> geometry_digest(const std::string& bytes) {
  rust::Vec<std::uint8_t> digest = occt_geometry_digest(
      rust::Slice<const std::uint8_t>(
          reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
  if (digest.size() != 32) {
    throw std::runtime_error("OCCT geometry digest has an invalid length");
  }
  std::array<std::uint8_t, 32> result;
  std::copy(digest.begin(), digest.end(), result.begin());
  return result;
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

// XCAF application registration, transfer and Close share kernel state. A
// recursive lock permits exception cleanup while the transfer boundary is held.
std::recursive_timed_mutex& cad_exchange_mutex() {
  static std::recursive_timed_mutex mutex;
  return mutex;
}

struct CadDocument {
  TopoDS_Shape shape;
  Handle(TDocStd_Document) document;
  Handle(XCAFDoc_ShapeTool) shape_tool;
  Handle(XCAFDoc_ColorTool) color_tool;
  Handle(XCAFDoc_LayerTool) layer_tool;
  Handle(XCAFDoc_MaterialTool) material_tool;
  bool has_xcaf = false;

  CadDocument() = default;
  CadDocument(const CadDocument&) = delete;
  CadDocument& operator=(const CadDocument&) = delete;

  CadDocument(CadDocument&& other) noexcept {
    *this = std::move(other);
  }

  CadDocument& operator=(CadDocument&& other) noexcept {
    if (this != &other) {
      close_document();
      shape = std::move(other.shape);
      document = std::move(other.document);
      shape_tool = std::move(other.shape_tool);
      color_tool = std::move(other.color_tool);
      layer_tool = std::move(other.layer_tool);
      material_tool = std::move(other.material_tool);
      has_xcaf = other.has_xcaf;

      other.document.Nullify();
      other.shape_tool.Nullify();
      other.color_tool.Nullify();
      other.layer_tool.Nullify();
      other.material_tool.Nullify();
      other.has_xcaf = false;
    }
    return *this;
  }

  ~CadDocument() {
    close_document();
  }

private:
  void close_document() {
    if (document.IsNull()) {
      return;
    }
    shape_tool.Nullify();
    color_tool.Nullify();
    layer_tool.Nullify();
    material_tool.Nullify();
    std::lock_guard<std::recursive_timed_mutex> lock(cad_exchange_mutex());
    Handle(XCAFApp_Application) app = XCAFApp_Application::GetApplication();
    app->Close(document);
    document.Nullify();
    has_xcaf = false;
  }
};

struct CadPreviewSession {
  std::string path;
  OcctCadFormat format;
  CadDocument document;
  IMeshTools_Parameters mesh_parameters;
  std::vector<TopoDS_Face> faces;
  std::uint64_t face_cursor = 0;
};

std::mutex& preview_sessions_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<std::uint64_t, std::unique_ptr<CadPreviewSession>>& preview_sessions() {
  static std::unordered_map<std::uint64_t, std::unique_ptr<CadPreviewSession>> sessions;
  return sessions;
}

std::atomic<std::uint64_t>& next_preview_session_id() {
  static std::atomic<std::uint64_t> next_id{1};
  return next_id;
}

void check_cancelled(const OcctImportOptions& options) {
  if (options.cancel_token_id != 0 && occt_import_cancelled(options.cancel_token_id)) {
    throw std::runtime_error("OCCT CAD import cancelled");
  }
}

std::string label_entry(const TDF_Label& label);

IMeshTools_Parameters mesh_parameters_from_options(const OcctImportOptions& options) {
  IMeshTools_Parameters mesh_parameters;
  mesh_parameters.Deflection = options.linear_deflection;
  mesh_parameters.Relative = options.relative_deflection;
  mesh_parameters.Angle = options.angular_deflection;
  mesh_parameters.InParallel = Standard_False;
  return mesh_parameters;
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

// OCCT exchange/XCAF transfer uses process-wide application and unit state.
// Serialize this unsafe boundary only; independent shape evaluation and meshing
// keep their own parallelism. Waiting still observes the import's cancellation.
std::unique_lock<std::recursive_timed_mutex> lock_cad_exchange(const OcctImportOptions& options) {
  std::unique_lock<std::recursive_timed_mutex> lock(cad_exchange_mutex(), std::defer_lock);
  while (!lock.try_lock_for(std::chrono::milliseconds(25))) {
    check_cancelled(options);
  }
  check_cancelled(options);
  return lock;
}

CadDocument finish_xcaf_document(CadDocument result) {
  const Handle(TDocStd_Document)& document = result.document;
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

// OCCT's sub-shape name reading misses faces, edges and vertices of a solid's inner shells
// (BREP_WITH_VOIDS cavities, such as an electrode cavity in an insulator). Every named
// ADVANCED_FACE, EDGE_CURVE and VERTEX_POINT in the file is mapped through the transfer result to
// its kernel shape and labeled in the part that owns it, unless OCCT already labeled it.
static TopAbs_ShapeEnum named_step_subshape_kind(const Handle(Standard_Transient)& entity) {
  if (entity->IsKind(STANDARD_TYPE(StepShape_AdvancedFace))) return TopAbs_FACE;
  if (entity->IsKind(STANDARD_TYPE(StepShape_EdgeCurve))) return TopAbs_EDGE;
  if (entity->IsKind(STANDARD_TYPE(StepShape_VertexPoint))) return TopAbs_VERTEX;
  return TopAbs_SHAPE;
}

static void label_named_step_subshapes(const Handle(XSControl_WorkSession)& session,
                                       CadDocument& document) {
  if (session.IsNull() || !document.has_xcaf || document.shape_tool.IsNull()) {
    return;
  }
  const Handle(Interface_InterfaceModel) model = session->Model();
  const Handle(XSControl_TransferReader) transfer = session->TransferReader();
  if (model.IsNull() || transfer.IsNull()) {
    return;
  }
  std::vector<std::pair<TopoDS_Shape, std::string>> named;
  for (Standard_Integer index = 1; index <= model->NbEntities(); ++index) {
    const Handle(StepShape_TopologicalRepresentationItem) item =
        Handle(StepShape_TopologicalRepresentationItem)::DownCast(model->Value(index));
    if (item.IsNull() || item->Name().IsNull() || item->Name()->Length() == 0) {
      continue;
    }
    const TopAbs_ShapeEnum kind = named_step_subshape_kind(item);
    if (kind == TopAbs_SHAPE) {
      continue;
    }
    const TopoDS_Shape shape = TransferBRep::ShapeResult(transfer->TransientProcess(), item);
    if (shape.IsNull() || shape.ShapeType() != kind) {
      continue;
    }
    named.emplace_back(shape, std::string(item->Name()->ToCString()));
  }
  if (named.empty()) {
    return;
  }
  TDF_LabelSequence labels;
  document.shape_tool->GetShapes(labels);
  for (Standard_Integer index = 1; index <= labels.Length(); ++index) {
    const TDF_Label& definition = labels.Value(index);
    if (!XCAFDoc_ShapeTool::IsSimpleShape(definition)) {
      continue;
    }
    TopTools_IndexedMapOfShape subshapes;
    const TopoDS_Shape definition_shape = XCAFDoc_ShapeTool::GetShape(definition);
    for (const TopAbs_ShapeEnum kind : {TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX}) {
      TopExp::MapShapes(definition_shape, kind, subshapes);
    }
    std::unordered_map<const TopoDS_TShape*, Standard_Integer> by_tshape;
    for (Standard_Integer subshape = 1; subshape <= subshapes.Extent(); ++subshape) {
      by_tshape.emplace(subshapes(subshape).TShape().get(), subshape);
    }
    for (const auto& [shape, name] : named) {
      const auto found = by_tshape.find(shape.TShape().get());
      if (found == by_tshape.end()) {
        continue;
      }
      const TopoDS_Shape& subshape = subshapes(found->second);
      TDF_Label existing;
      if (document.shape_tool->FindSubShape(definition, subshape, existing) &&
          !exact_label_name(existing).empty()) {
        continue;
      }
      const TDF_Label label = existing.IsNull()
                                  ? document.shape_tool->AddSubShape(definition, subshape)
                                  : existing;
      if (!label.IsNull()) {
        TDataStd_Name::Set(label, TCollection_ExtendedString(name.c_str(), Standard_True));
      }
    }
  }
}

// Inspect the reader's source entities before transfer. The destination document
// unit is a separate preference and is never evidence of a file declaration.
void verify_step_source_units(const STEPControl_Reader& reader,
                              rust::Slice<const std::uint8_t> bytes,
                              const OcctImportOptions& options) {
  const auto model = reader.StepModel();
  if (model.IsNull()) throw std::runtime_error("OCCT STEP reader has no source model");
  std::vector<OcctStepContextUnitPayload> contexts;
  StepData_Factors factors;
  // CascadeUnit is expressed in millimetres. A local 1000-mm reference makes
  // ComputeFactors return metres per source coordinate, independent of transfer.
  factors.SetCascadeUnit(1000.0);
  for (Standard_Integer index = 1; index <= model->NbEntities(); ++index) {
    check_cancelled(options);
    const auto entity = model->Entity(index);
    Handle(StepRepr_GlobalUnitAssignedContext) context =
        Handle(StepRepr_GlobalUnitAssignedContext)::DownCast(entity);
    if (context.IsNull()) {
      auto combined = Handle(StepGeom_GeometricRepresentationContextAndGlobalUnitAssignedContext)::DownCast(entity);
      if (!combined.IsNull()) context = combined->GlobalUnitAssignedContext();
    }
    if (context.IsNull()) {
      auto combined = Handle(StepGeom_GeomRepContextAndGlobUnitAssCtxAndGlobUncertaintyAssCtx)::DownCast(entity);
      if (!combined.IsNull()) context = combined->GlobalUnitAssignedContext();
    }
    if (context.IsNull()) continue;
    if (contexts.size() >= 4096) throw std::runtime_error("STEP source-unit context limit exceeded");
    const auto label = model->IdentLabel(entity);
    if (label <= 0) throw std::runtime_error("OCCT source-unit context has no STEP entity identity");
    STEPConstruct_UnitContext units;
    const auto status = units.ComputeFactors(context, factors);
    const bool length_done = status == 0 && units.LengthDone();
    contexts.push_back(OcctStepContextUnitPayload{
        static_cast<std::uint64_t>(label), length_done,
        length_done ? units.LengthFactor() : 0.0});
  }
  occt_validate_step_source_units(bytes,
      rust::Slice<const OcctStepContextUnitPayload>(contexts.data(), contexts.size()),
      options.require_declared_step_units);
}

CadDocument read_step_shape(const std::string& path,
                            rust::Slice<const std::uint8_t> bytes,
                            std::istream& stream,
                            const OcctImportOptions& options) {
  auto exchange_lock = lock_cad_exchange(options);
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
  // CAD tools write body, face, edge and vertex names on STEP representation items. OCCT reads
  // them into XCAF sub-shape labels only when the model asks for it; set it on this file's model
  // so `select name` resolves authored names without touching process-global settings.
  Handle(StepData_StepModel) step_model =
      Handle(StepData_StepModel)::DownCast(reader.Reader().WS()->Model());
  if (step_model.IsNull()) {
    throw std::runtime_error("OCCT STEP reader produced no STEP model");
  }
  step_model->InternalParameters.ReadSubshapeNames = true;
  verify_step_source_units(reader.Reader(), bytes, options);
  // STEP declares physical units. Exact imports request a stable millimetre
  // kernel representation to preserve tolerance-scale features; preview imports
  // request their display coordinate unit. Rust projects exact contracts to SI.
  // The target is document-local, never process-global.
  if (!std::isfinite(options.step_coordinate_unit_m) ||
      options.step_coordinate_unit_m <= 0.0) {
    throw std::runtime_error("STEP transfer requires a finite positive length unit");
  }
  CadDocument result;
  result.document = new_xcaf_document();
  const Handle(TDocStd_Document)& document = result.document;
  // XCAF transfer takes its target unit from the destination document; setting
  // only the underlying STEP reader is overwritten by STEPCAF transfer.
  XCAFDoc_DocumentTool::SetLengthUnit(document, options.step_coordinate_unit_m);
  RunmatCancelProgressIndicator transfer_progress(options);
  if (!reader.Transfer(document, transfer_progress.Start())) {
    check_cancelled(options);
    throw std::runtime_error("OCCT STEP XCAF transfer failed");
  }
  check_cancelled(options);
  CadDocument finished = finish_xcaf_document(std::move(result));
  label_named_step_subshapes(reader.Reader().WS(), finished);
  return finished;
}

CadDocument read_iges_shape(const std::string& path,
                            std::istream& stream,
                            const std::string& materialized_path,
                            const OcctImportOptions& options) {
  auto exchange_lock = lock_cad_exchange(options);
  check_cancelled(options);
  IGESCAFControl_Reader reader;
  reader.SetNameMode(Standard_True);
  reader.SetColorMode(Standard_True);
  reader.SetLayerMode(Standard_True);
  // OCCT 7.8 exposes the inherited stream API for IGES but does not implement IGES stream
  // loading. Rust provides a securely created, content-identical temporary file until the
  // pinned kernel has native IGES stream support. The path never enters geometry identity.
  IFSelect_ReturnStatus status = materialized_path.empty()
                                     ? reader.ReadStream(path.c_str(), stream)
                                     : reader.ReadFile(materialized_path.c_str());
  check_cancelled(options);
  if (status != IFSelect_RetDone) {
    throw std::runtime_error(
        "OCCT IGES reader did not return IFSelect_RetDone (status " +
        std::to_string(static_cast<int>(status)) + ")");
  }
  // IGES declares physical units like STEP. Exact imports request the stable
  // millimetre kernel representation; preview imports request the asset's
  // coordinate unit so the reported unit matches the coordinates.
  if (!std::isfinite(options.step_coordinate_unit_m) ||
      options.step_coordinate_unit_m <= 0.0) {
    throw std::runtime_error("IGES transfer requires a finite positive length unit");
  }
  CadDocument result;
  result.document = new_xcaf_document();
  const Handle(TDocStd_Document)& document = result.document;
  XCAFDoc_DocumentTool::SetLengthUnit(document, options.step_coordinate_unit_m);
  RunmatCancelProgressIndicator transfer_progress(options);
  if (!reader.Transfer(document, transfer_progress.Start())) {
    check_cancelled(options);
    throw std::runtime_error("OCCT IGES XCAF transfer failed");
  }
  check_cancelled(options);
  return finish_xcaf_document(std::move(result));
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
  const double scale = options.unitless_source_unit_m / options.step_coordinate_unit_m;
  if (!std::isfinite(scale) || scale <= 0.0) {
    throw std::runtime_error("BREP source/output coordinate conversion must be finite and positive");
  }
  if (scale != 1.0) {
    gp_Trsf transform;
    transform.SetScale(gp_Pnt(0.0, 0.0, 0.0), scale);
    shape = BRepBuilderAPI_Transform(shape, transform, Standard_True).Shape();
    check_cancelled(options);
  }
  CadDocument result;
  result.shape = shape;
  return result;
}

CadDocument read_shape(const std::string& path,
                       rust::Slice<const std::uint8_t> bytes,
                       const std::string& materialized_path,
                       OcctCadFormat format,
                       const OcctImportOptions& options) {
  std::string payload = bytes_to_string(bytes);
  std::istringstream stream(payload);
  switch (format) {
    case OcctCadFormat::Step:
      return read_step_shape(path, bytes, stream, options);
    case OcctCadFormat::Iges:
      return read_iges_shape(path, stream, materialized_path, options);
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

bool finite_value(double value) {
  return std::isfinite(value);
}

void append_face_evaluation_sample_at(OcctImportPayload& result,
                                      const TopoDS_Face& face,
                                      std::uint64_t face_id,
                                      Standard_Real u,
                                      Standard_Real v,
                                      const OcctImportOptions& options) {
  check_cancelled(options);
  if (!finite_value(u) || !finite_value(v)) {
    return;
  }

  BRepAdaptor_Surface surface(face);
  gp_Pnt point;
  gp_Vec d_u;
  gp_Vec d_v;
  surface.D1(u, v, point, d_u, d_v);
  gp_Vec normal = d_u.Crossed(d_v);
  const Standard_Real normal_magnitude = normal.Magnitude();
  if (!finite_value(normal_magnitude) || normal_magnitude <= 1.0e-15) {
    return;
  }
  normal.Divide(normal_magnitude);
  if (face.Orientation() == TopAbs_REVERSED) {
    normal.Reverse();
  }

  OcctFaceEvaluationSamplePayload sample;
  sample.face_id = face_id;
  sample.u = u;
  sample.v = v;
  sample.point_x = point.X();
  sample.point_y = point.Y();
  sample.point_z = point.Z();
  sample.normal_x = normal.X();
  sample.normal_y = normal.Y();
  sample.normal_z = normal.Z();
  sample.projection_error = 0.0;
  if (!finite_value(sample.point_x) || !finite_value(sample.point_y) ||
      !finite_value(sample.point_z) || !finite_value(sample.normal_x) ||
      !finite_value(sample.normal_y) || !finite_value(sample.normal_z)) {
    return;
  }
  result.face_evaluation_samples.push_back(sample);
}

void append_face_evaluation_sample(OcctImportPayload& result,
                                   const TopoDS_Face& face,
                                   std::uint64_t face_id,
                                   const OcctImportOptions& options) {
  check_cancelled(options);
  Standard_Real u_min = 0.0;
  Standard_Real u_max = 0.0;
  Standard_Real v_min = 0.0;
  Standard_Real v_max = 0.0;
  BRepTools::UVBounds(face, u_min, u_max, v_min, v_max);
  if (!finite_value(u_min) || !finite_value(u_max) || !finite_value(v_min) ||
      !finite_value(v_max) || u_max <= u_min || v_max <= v_min) {
    return;
  }
  const Standard_Real u_mid = 0.5 * (u_min + u_max);
  const Standard_Real v_mid = 0.5 * (v_min + v_max);
  const Standard_Real u_low = u_min + 0.25 * (u_max - u_min);
  const Standard_Real u_high = u_min + 0.75 * (u_max - u_min);
  const Standard_Real v_low = v_min + 0.25 * (v_max - v_min);
  const Standard_Real v_high = v_min + 0.75 * (v_max - v_min);
  append_face_evaluation_sample_at(result, face, face_id, u_mid, v_mid, options);
  append_face_evaluation_sample_at(result, face, face_id, u_low, v_mid, options);
  append_face_evaluation_sample_at(result, face, face_id, u_high, v_mid, options);
  append_face_evaluation_sample_at(result, face, face_id, u_mid, v_low, options);
  append_face_evaluation_sample_at(result, face, face_id, u_mid, v_high, options);
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
                                   rust::Str materialized_path,
                                   OcctCadFormat format,
                                   OcctImportOptions options) {
  const std::string path_string = str_from_rust(path);
  const std::string materialized_path_string = str_from_rust(materialized_path);
  check_cancelled(options);
  CadDocument document =
      read_shape(path_string, bytes, materialized_path_string, format, options);
  check_cancelled(options);

  IMeshTools_Parameters mesh_parameters = mesh_parameters_from_options(options);
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
    append_face_evaluation_sample(result, face, face_id, options);
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

// Exports often carry construction geometry beside a part's solids: PMI leader curves,
// supplemental wireframe and points that bound no face. It is not part of the analysis model,
// and its curves would be meshed and sized against the real edges, so a part that has faces
// keeps only its solids, shells and faces. A wireframe-only part is left unchanged.
static TopoDS_Shape without_free_wireframe(const TopoDS_Shape& shape, bool& changed) {
  if (shape.ShapeType() != TopAbs_COMPOUND) {
    return shape;
  }
  BRep_Builder builder;
  TopoDS_Compound result;
  builder.MakeCompound(result);
  bool empty = true;
  for (TopoDS_Iterator children(shape); children.More(); children.Next()) {
    const TopoDS_Shape& child = children.Value();
    switch (child.ShapeType()) {
      case TopAbs_EDGE:
      case TopAbs_WIRE:
      case TopAbs_VERTEX:
        changed = true;
        continue;
      case TopAbs_COMPOUND: {
        const TopoDS_Shape kept = without_free_wireframe(child, changed);
        if (!TopExp_Explorer(kept, TopAbs_FACE).More()) {
          changed = true;
          continue;
        }
        builder.Add(result, kept);
        break;
      }
      default:
        builder.Add(result, child);
        break;
    }
    empty = false;
  }
  if (empty) {
    return shape;
  }
  result.Location(TopLoc_Location());
  return result;
}

static void strip_free_wireframe(CadDocument& document) {
  if (!TopExp_Explorer(document.shape, TopAbs_FACE).More()) {
    return;
  }
  if (!document.has_xcaf || document.shape_tool.IsNull()) {
    bool changed = false;
    TopoDS_Shape stripped = without_free_wireframe(document.shape, changed);
    if (changed) {
      document.shape = stripped;
    }
    return;
  }
  TDF_LabelSequence labels;
  document.shape_tool->GetShapes(labels);
  bool any = false;
  for (Standard_Integer index = 1; index <= labels.Length(); ++index) {
    const TDF_Label& label = labels.Value(index);
    if (!XCAFDoc_ShapeTool::IsSimpleShape(label)) {
      continue;
    }
    TopoDS_Shape shape = XCAFDoc_ShapeTool::GetShape(label);
    bool changed = false;
    TopoDS_Shape stripped;
    if (!TopExp_Explorer(shape, TopAbs_FACE).More()) {
      // A part holding only construction geometry contributes nothing to the model.
      TopoDS_Compound empty;
      BRep_Builder().MakeCompound(empty);
      stripped = empty;
      changed = !shape.IsNull();
    } else {
      stripped = without_free_wireframe(shape, changed);
    }
    if (changed) {
      document.shape_tool->SetShape(label, stripped);
      any = true;
    }
  }
  if (any) {
    document.shape_tool->UpdateAssemblies();
    TDF_LabelSequence free_shapes;
    document.shape_tool->GetFreeShapes(free_shapes);
    document.shape = XCAFDoc_ShapeTool::GetOneShape(free_shapes);
  }
}

// Edge tolerances written by CAD exporters bound the gap between an edge's 3D curve and its
// pcurves only at the writer's control points (OCCT checks 23), and exports often understate it.
// RunMat's curve, surface and validation stages rely on the edge tolerance continuously, so each
// understated tolerance is raised to the measured gap plus a 5% margin, as OCCT's own shape
// healing does. The gap is sampled at 101 control points; where that comes within half of the
// written tolerance, the true maximum between the samples is found by optimisation on each face,
// since a narrow peak between samples can exceed the sampled gap by a third (an exported filleted
// skid: 0.25 um sampled, 0.34 um found). A gap is admitted when it is at most ten times the written
// tolerance or at most kAdmissibleEdgeGapM (1 um), whichever is larger. The absolute bound keeps
// the rule meaningful for writers that stamp a nominal tolerance near the kernel's confusion
// distance: an exported EV battery pack writes 0.9 nm on every edge while its B-spline pcurves
// sit 8-13 nm off their 3D curves, ordinary for real CAD. Larger gaps are real defects and are
// left for validation to report.

static void admit_measured_edge_tolerances(const TopoDS_Shape& shape, const OcctImportOptions& options) {
  TopTools_IndexedMapOfShape edges;
  TopExp::MapShapes(shape, TopAbs_EDGE, edges);
  TopTools_IndexedDataMapOfShapeListOfShape edge_faces;
  TopExp::MapShapesAndUniqueAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_faces);
  ShapeAnalysis_Edge analysis;
  BRep_Builder builder;
  for (Standard_Integer index = 1; index <= edges.Extent(); ++index) {
    if ((index & 63) == 0) {
      check_cancelled(options);
    }
    const TopoDS_Edge& edge = TopoDS::Edge(edges(index));
    if (BRep_Tool::Degenerated(edge)) {
      continue;
    }
    Standard_Real deviation = 0.0;
    analysis.CheckSameParameter(edge, deviation, 101);
    const double written = BRep_Tool::Tolerance(edge);
    if (deviation > 0.5 * written) {
      for (const TopoDS_Shape& face : edge_faces.FindFromKey(edge)) {
        BRepLib_CheckCurveOnSurface exact(edge, TopoDS::Face(face));
        exact.Perform();
        if (exact.IsDone() && std::isfinite(exact.MaxDistance()))
          deviation = std::max(deviation, exact.MaxDistance());
      }
    }
    const double required = deviation * 1.05;
    const double admissible =
        std::max(10.0 * written, kAdmissibleEdgeGapM / options.step_coordinate_unit_m);
    if (!std::isfinite(required) || required <= written || deviation > admissible) {
      continue;
    }
    builder.UpdateEdge(edge, required);
    for (TopoDS_Iterator vertices(edge, Standard_False, Standard_False); vertices.More();
         vertices.Next()) {
      if (vertices.Value().ShapeType() == TopAbs_VERTEX &&
          BRep_Tool::Tolerance(TopoDS::Vertex(vertices.Value())) < required) {
        builder.UpdateVertex(TopoDS::Vertex(vertices.Value()), required);
      }
    }
  }
}

OcctExactShapePayload import_exact_cad_bytes(
    rust::Str path,
    rust::Slice<const std::uint8_t> bytes,
    rust::Str materialized_path,
    OcctCadFormat format,
    OcctImportOptions options) {
  const std::string path_string = str_from_rust(path);
  const std::string materialized_path_string = str_from_rust(materialized_path);
  check_cancelled(options);
  CadDocument document =
      read_shape(path_string, bytes, materialized_path_string, format, options);
  check_cancelled(options);

  // Polygonal data is a derived display cache. Removing it before serialization ensures the
  // representation consumed by exact evaluators cannot accidentally carry tessellation authority.
  BRepTools::Clean(document.shape);
  check_cancelled(options);
  strip_free_wireframe(document);
  admit_measured_edge_tolerances(document.shape, options);
  check_cancelled(options);

  std::array<std::uint8_t, 32> original_geometry_digest{};
  bool original_kernel_valid = false;
  std::uint64_t healing_identity_work_bytes = 0;
  bool duplicates_consolidated = false;
  bool post_duplicate_kernel_valid = false;
  bool post_sewing_kernel_valid = false;
  bool post_small_topology_kernel_valid = false;
  bool orientation_repaired = false;
  bool sewn = false;
  bool gaps_repaired = false;
  bool short_edges_simplified = false;
  bool sliver_faces_simplified = false;
  double maximum_healing_displacement = 0.0;
  std::array<double, 3> displacement_original{0.0, 0.0, 0.0};
  std::array<double, 3> displacement_proposed{0.0, 0.0, 0.0};
  std::vector<ExactHealingMutation::Relation> healing_relations;
  if (options.heal_orientation || options.heal_duplicates || options.heal_sew ||
      options.heal_gaps || options.heal_short_edges_and_sliver_faces) {
    const std::string original_representation = serialize_exact_shape(document.shape, options);
    if (static_cast<std::uint64_t>(original_representation.size()) >
        options.max_exact_representation_bytes) {
      throw std::runtime_error("OCCT exact representation exceeded its byte budget");
    }
    original_geometry_digest = geometry_digest(original_representation);
    const BRepCheck_Analyzer original_analyzer(document.shape, Standard_True);
    original_kernel_valid = original_analyzer.IsValid();
    if (document.has_xcaf) {
      ExactXcafHealingResult healing = heal_exact_xcaf_definitions(
          document.shape_tool, options, healing_identity_work_bytes);
      document.shape = healing.shape;
      healing_identity_work_bytes = healing.identity_work_bytes;
      duplicates_consolidated = healing.duplicates_consolidated;
      post_duplicate_kernel_valid = healing.post_duplicate_kernel_valid;
      sewn = healing.sewn;
      gaps_repaired = healing.gaps_repaired;
      post_sewing_kernel_valid = healing.post_sewing_kernel_valid;
      short_edges_simplified = healing.short_edges_simplified;
      sliver_faces_simplified = healing.sliver_faces_simplified;
      post_small_topology_kernel_valid = healing.post_small_topology_kernel_valid;
      orientation_repaired = healing.orientation_repaired;
      maximum_healing_displacement = healing.maximum_displacement;
      displacement_original = healing.displacement_original;
      displacement_proposed = healing.displacement_proposed;
      healing_relations = std::move(healing.relations);
    } else {
      if (options.heal_duplicates) {
        ExactHealingMutation consolidation = consolidate_exact_duplicates(
            document.shape, options, healing_identity_work_bytes);
        document.shape = consolidation.shape;
        healing_identity_work_bytes = consolidation.identity_work_bytes;
        duplicates_consolidated = consolidation.changed;
        const BRepCheck_Analyzer duplicate_analyzer(document.shape, Standard_True);
        post_duplicate_kernel_valid = duplicate_analyzer.IsValid();
      }
      if (options.heal_sew || options.heal_gaps) {
        const std::string before_sewing = serialize_exact_shape(document.shape, options);
        if (static_cast<std::uint64_t>(before_sewing.size()) >
            options.max_exact_representation_bytes) {
          throw std::runtime_error("OCCT exact representation exceeded its byte budget");
        }
        ExactHealingMutation sewing = sew_exact_shape(
            document.shape,
            options,
            options.maximum_healing_displacement,
            healing_identity_work_bytes);
        document.shape = sewing.shape;
        healing_identity_work_bytes = sewing.identity_work_bytes;
        maximum_healing_displacement = sewing.maximum_displacement;
        displacement_original = sewing.displacement_original;
        displacement_proposed = sewing.displacement_proposed;
        healing_relations = sewing.relations;
        const std::string after_sewing = serialize_exact_shape(document.shape, options);
        if (static_cast<std::uint64_t>(after_sewing.size()) >
            options.max_exact_representation_bytes) {
          throw std::runtime_error("OCCT exact representation exceeded its byte budget");
        }
        const bool changed = geometry_digest(before_sewing) != geometry_digest(after_sewing);
        sewn = options.heal_sew && changed;
        gaps_repaired = options.heal_gaps && changed;
        const BRepCheck_Analyzer sewing_analyzer(document.shape, Standard_True);
        post_sewing_kernel_valid = sewing_analyzer.IsValid();
      }
      if (options.heal_short_edges_and_sliver_faces) {
        const std::string before_small_topology =
            serialize_exact_shape(document.shape, options);
        if (static_cast<std::uint64_t>(before_small_topology.size()) >
            options.max_exact_representation_bytes) {
          throw std::runtime_error("OCCT exact representation exceeded its byte budget");
        }
        ExactHealingMutation small_topology = simplify_exact_small_topology(
            document.shape,
            options,
            options.maximum_healing_displacement,
            healing_identity_work_bytes,
            short_edges_simplified,
            sliver_faces_simplified);
        document.shape = small_topology.shape;
        healing_identity_work_bytes = small_topology.identity_work_bytes;
        maximum_healing_displacement = small_topology.maximum_displacement;
        displacement_original = small_topology.displacement_original;
        displacement_proposed = small_topology.displacement_proposed;
        healing_relations = small_topology.relations;
        const std::string after_small_topology =
            serialize_exact_shape(document.shape, options);
        if (static_cast<std::uint64_t>(after_small_topology.size()) >
            options.max_exact_representation_bytes) {
          throw std::runtime_error("OCCT exact representation exceeded its byte budget");
        }
        const bool changed =
            geometry_digest(before_small_topology) != geometry_digest(after_small_topology);
        if (changed != (short_edges_simplified || sliver_faces_simplified)) {
          throw std::runtime_error(
              "OCCT small-topology repair produced unclassified topology mutation");
        }
        const BRepCheck_Analyzer small_topology_analyzer(document.shape, Standard_True);
        post_small_topology_kernel_valid = small_topology_analyzer.IsValid();
      }
      if (options.heal_orientation) {
        const std::array<std::uint8_t, 32> before_orientation =
            geometry_digest(serialize_exact_shape(document.shape, options));
        ExactHealingMutation repair = repair_exact_orientation(
            document.shape, options, healing_identity_work_bytes);
        document.shape = repair.shape;
        healing_identity_work_bytes = repair.identity_work_bytes;
        orientation_repaired = before_orientation !=
                               geometry_digest(serialize_exact_shape(document.shape, options));
      }
    }
    BRepTools::Clean(document.shape);
    check_cancelled(options);
  }

  // Shared interfaces replace the part structure with one compound of placed solids whose
  // coincident faces are shared; names travel with the kernel shapes instead of XCAF labels.
  std::optional<ExactNameTable> shared_names;
  if (options.share_interfaces) {
    ExactSharedInterfaces shared = share_exact_interfaces(
        document.shape, document.shape_tool, document.has_xcaf, options);
    document.shape = shared.shape;
    shared_names.emplace(std::move(shared.names));
    BRepTools::Clean(document.shape);
    check_cancelled(options);
  }

  const std::string representation = serialize_exact_shape(document.shape, options);
  if (static_cast<std::uint64_t>(representation.size()) >
      options.max_exact_representation_bytes) {
    throw std::runtime_error("OCCT exact representation exceeded its byte budget");
  }

  OcctExactShapePayload result;
  result.kernel_version = OCC_VERSION_COMPLETE;
  result.kernel_abi = std::string("occt/") + OCC_VERSION_COMPLETE + "/brep-v3";
  result.original_kernel_valid = original_kernel_valid;
  result.healing_identity_work_bytes = healing_identity_work_bytes;
  result.orientation_repaired = orientation_repaired;
  result.duplicates_consolidated = duplicates_consolidated;
  result.post_duplicate_kernel_valid = post_duplicate_kernel_valid;
  result.post_sewing_kernel_valid = post_sewing_kernel_valid;
  result.post_small_topology_kernel_valid = post_small_topology_kernel_valid;
  result.sewn = sewn;
  result.gaps_repaired = gaps_repaired;
  result.short_edges_simplified = short_edges_simplified;
  result.sliver_faces_simplified = sliver_faces_simplified;
  result.maximum_healing_displacement = maximum_healing_displacement;
  result.displacement_original_x = displacement_original[0];
  result.displacement_original_y = displacement_original[1];
  result.displacement_original_z = displacement_original[2];
  result.displacement_proposed_x = displacement_proposed[0];
  result.displacement_proposed_y = displacement_proposed[1];
  result.displacement_proposed_z = displacement_proposed[2];
  for (const ExactHealingMutation::Relation& relation : healing_relations) {
    OcctHealingRelationPayload payload;
    switch (relation.kind) {
      case 0:
        payload.kind = OcctHealingEntityKind::Vertex;
        break;
      case 1:
        payload.kind = OcctHealingEntityKind::Edge;
        break;
      case 2:
        payload.kind = OcctHealingEntityKind::Face;
        break;
      default:
        throw std::runtime_error("OCCT healing relation has an invalid entity kind");
    }
    for (const std::string& segment : relation.path_segments) {
      payload.path_segments.push_back(segment);
    }
    for (const std::uint8_t byte : relation.source_digest) {
      payload.source_digest.push_back(byte);
    }
    const bool has_target = std::any_of(
        relation.target_digest.begin(), relation.target_digest.end(), [](const auto byte) {
          return byte != 0;
        });
    if (has_target) {
      for (const std::uint8_t byte : relation.target_digest) {
        payload.target_digest.push_back(byte);
      }
    }
    result.healing_relations.push_back(payload);
  }
  if (result.orientation_repaired || result.duplicates_consolidated || result.sewn ||
      result.gaps_repaired || result.short_edges_simplified ||
      result.sliver_faces_simplified) {
    for (const std::uint8_t byte : original_geometry_digest) {
      result.original_geometry_digest.push_back(byte);
    }
  }
  result.representation.reserve(representation.size());
  for (const unsigned char byte : representation) {
    result.representation.push_back(static_cast<std::uint8_t>(byte));
  }

  const std::pair<TopAbs_ShapeEnum, std::uint64_t*> inventories[] = {
      {TopAbs_COMPOUND, &result.compound_count},
      {TopAbs_COMPSOLID, &result.compsolid_count},
      {TopAbs_SOLID, &result.solid_count},
      {TopAbs_SHELL, &result.shell_count},
      {TopAbs_FACE, &result.face_count},
      {TopAbs_WIRE, &result.wire_count},
      {TopAbs_EDGE, &result.edge_count},
      {TopAbs_VERTEX, &result.vertex_count},
  };
  for (const auto& inventory : inventories) {
    check_cancelled(options);
    TopTools_IndexedMapOfShape shapes;
    TopExp::MapShapes(document.shape, inventory.first, shapes);
    *inventory.second = static_cast<std::uint64_t>(shapes.Extent());
    if (*inventory.second > options.max_exact_entities) {
      throw std::runtime_error("OCCT exact topology exceeded its entity budget");
    }
  }

  append_exact_occurrences(result,
                           document.shape,
                           document.shape_tool,
                           document.has_xcaf && !shared_names.has_value(),
                           representation,
                           shared_names.has_value() ? &*shared_names : nullptr,
                           options);
  const std::uint64_t projected_entity_count =
      static_cast<std::uint64_t>(result.occurrences.size()) +
      static_cast<std::uint64_t>(result.vertices.size()) +
      static_cast<std::uint64_t>(result.edges.size()) +
      static_cast<std::uint64_t>(result.faces.size()) +
      static_cast<std::uint64_t>(result.wires.size()) +
      static_cast<std::uint64_t>(result.coedges.size()) +
      static_cast<std::uint64_t>(result.shells.size()) +
      static_cast<std::uint64_t>(result.solids.size()) +
      static_cast<std::uint64_t>(result.lumps.size());
  if (projected_entity_count > options.max_exact_entities) {
    throw std::runtime_error("OCCT exact topology exceeded its occurrence entity budget");
  }
  result.vertex_count = static_cast<std::uint64_t>(result.vertices.size());
  result.edge_count = static_cast<std::uint64_t>(result.edges.size());
  result.face_count = static_cast<std::uint64_t>(result.faces.size());
  result.wire_count = static_cast<std::uint64_t>(result.wires.size());
  result.shell_count = static_cast<std::uint64_t>(result.shells.size());
  result.solid_count = static_cast<std::uint64_t>(result.solids.size());

  BRepCheck_Analyzer analyzer(document.shape, Standard_True);
  result.kernel_valid = analyzer.IsValid();
  check_cancelled(options);

  GProp_GProps surface_properties;
  BRepGProp::SurfaceProperties(document.shape, surface_properties);
  result.surface_area = surface_properties.Mass();

  GProp_GProps volume_properties;
  BRepGProp::VolumeProperties(document.shape, volume_properties);
  result.volume = volume_properties.Mass();
  result.has_volume_properties = result.solid_count > 0 && finite_value(result.volume) &&
                                 result.volume > 0.0;
  if (result.has_volume_properties) {
    const gp_Pnt centroid = volume_properties.CentreOfMass();
    const gp_Mat inertia = volume_properties.MatrixOfInertia();
    result.centroid_x = centroid.X();
    result.centroid_y = centroid.Y();
    result.centroid_z = centroid.Z();
    result.inertia_xx = inertia.Value(1, 1);
    result.inertia_yy = inertia.Value(2, 2);
    result.inertia_zz = inertia.Value(3, 3);
    result.inertia_xy = inertia.Value(1, 2);
    result.inertia_xz = inertia.Value(1, 3);
    result.inertia_yz = inertia.Value(2, 3);
  }
  check_cancelled(options);
  return result;
}

OcctPreviewSessionStartPayload start_cad_preview_session(
    rust::Str path,
    rust::Slice<const std::uint8_t> bytes,
    rust::Str materialized_path,
    OcctCadFormat format,
    OcctImportOptions options) {
  const std::string path_string = str_from_rust(path);
  const std::string materialized_path_string = str_from_rust(materialized_path);
  check_cancelled(options);
  auto session = std::make_unique<CadPreviewSession>();
  session->path = path_string;
  session->format = format;
  session->document =
      read_shape(path_string, bytes, materialized_path_string, format, options);
  session->mesh_parameters = mesh_parameters_from_options(options);
  check_cancelled(options);

  for (TopExp_Explorer explorer(session->document.shape, TopAbs_FACE);
       explorer.More();
       explorer.Next()) {
    check_cancelled(options);
    session->faces.push_back(TopoDS::Face(explorer.Current()));
  }
  if (session->faces.empty()) {
    throw std::runtime_error("OCCT import produced no faces");
  }

  const std::uint64_t session_id =
      next_preview_session_id().fetch_add(1, std::memory_order_relaxed);
  OcctPreviewSessionStartPayload result;
  result.session_id = session_id;
  result.face_count = static_cast<std::uint64_t>(session->faces.size());
  {
    std::lock_guard<std::mutex> lock(preview_sessions_mutex());
    preview_sessions().emplace(session_id, std::move(session));
  }
  return result;
}

OcctPreviewSessionChunkPayload read_cad_preview_session_chunk(
    std::uint64_t session_id,
    OcctPreviewSessionChunkOptions chunk_options) {
  std::lock_guard<std::mutex> lock(preview_sessions_mutex());
  auto found = preview_sessions().find(session_id);
  if (found == preview_sessions().end()) {
    throw std::runtime_error("OCCT CAD preview session does not exist");
  }

  CadPreviewSession& session = *found->second;
  OcctImportOptions options;
  options.linear_deflection = session.mesh_parameters.Deflection;
  options.angular_deflection = session.mesh_parameters.Angle;
  options.relative_deflection = session.mesh_parameters.Relative;
  options.max_triangles = std::numeric_limits<std::uint64_t>::max();
  options.truncate_at_max_triangles = false;
  options.max_exact_representation_bytes = std::numeric_limits<std::uint64_t>::max();
  options.max_exact_entities = std::numeric_limits<std::uint64_t>::max();
  options.cancel_token_id = chunk_options.cancel_token_id;
  check_cancelled(options);

  OcctImportPayload topology;
  topology.backend = "occt-native";
  topology.format_name = format_name(session.format);
  topology.truncated = false;
  topology.triangle_budget = chunk_options.target_triangles;
  if (session.face_cursor == 0) {
    append_assembly_tree(topology, session.document, options);
  }

  const std::uint64_t target_triangles =
      chunk_options.target_triangles == 0 ? 128000 : chunk_options.target_triangles;
  const std::uint64_t max_faces =
      chunk_options.max_faces == 0 ? std::numeric_limits<std::uint64_t>::max()
                                   : chunk_options.max_faces;
  std::uint64_t faces_processed = 0;
  while (session.face_cursor < session.faces.size()) {
    check_cancelled(options);
    if (!topology.triangle_face_ids.empty() &&
        static_cast<std::uint64_t>(topology.triangle_face_ids.size()) >= target_triangles) {
      break;
    }
    if (!topology.triangle_face_ids.empty() && faces_processed >= max_faces) {
      break;
    }

    const std::uint64_t face_id = session.face_cursor;
    const TopoDS_Face face = session.faces[static_cast<std::size_t>(session.face_cursor)];
    OcctFaceSemanticPayload semantic = face_semantics(session.document, face, face_id);
    const std::string resolved_face_name = semantic.label_name.empty()
                                               ? face_name(face_id)
                                               : static_cast<std::string>(semantic.label_name);
    RunmatCancelProgressIndicator face_mesh_progress(options);
    BRepMesh_IncrementalMesh face_mesh(face,
                                       session.mesh_parameters,
                                       face_mesh_progress.Start());
    check_cancelled(options);
    if (!face_mesh.IsDone()) {
      topology.warnings.push_back("OCCT preview tessellation did not complete for face " +
                                  std::to_string(face_id + 1));
    }
    append_face_evaluation_sample(topology, face, face_id, options);
    append_face_mesh(topology, face, face_id, resolved_face_name, options);
    if (has_face_semantics(semantic)) {
      topology.face_semantics.push_back(semantic);
    }
    session.face_cursor += 1;
    faces_processed += 1;
  }

  const bool done = session.face_cursor >= session.faces.size();
  OcctPreviewSessionChunkPayload result;
  result.session_id = session_id;
  result.done = done;
  result.face_cursor = session.face_cursor;
  result.face_count = static_cast<std::uint64_t>(session.faces.size());
  result.topology = std::move(topology);

  if (done) {
    preview_sessions().erase(found);
  }
  return result;
}

void close_cad_preview_session(std::uint64_t session_id) {
  std::lock_guard<std::mutex> lock(preview_sessions_mutex());
  preview_sessions().erase(session_id);
}

} // namespace occt_backend
} // namespace runmat_geometry_io
