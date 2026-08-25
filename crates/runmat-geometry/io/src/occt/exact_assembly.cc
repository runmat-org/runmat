#include "runmat-geometry-io/src/occt/ffi.rs.h"
#include "exact_assembly.hxx"
#include "exact_identity.hxx"
#include "exact_topology.hxx"

#include <BRepTools.hxx>
#include <BRepTools_ShapeSet.hxx>
#include <Message_ProgressIndicator.hxx>
#include <TCollection_AsciiString.hxx>
#include <TDF_Label.hxx>
#include <TDF_LabelSequence.hxx>
#include <TDF_Tool.hxx>
#include <TopLoc_Location.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_Trsf.hxx>

#include <cstdint>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace runmat_geometry_io {
namespace occt_backend {
namespace {

void check_cancelled(const OcctImportOptions& options) {
  if (options.cancel_token_id != 0 && occt_import_cancelled(options.cancel_token_id)) {
    throw std::runtime_error("OCCT CAD import cancelled");
  }
}

class CancelProgress final : public Message_ProgressIndicator {
public:
  explicit CancelProgress(const OcctImportOptions& options) : options_(options) {}

protected:
  Standard_Boolean UserBreak() override {
    return options_.cancel_token_id != 0 && occt_import_cancelled(options_.cancel_token_id);
  }

  void Show(const Message_ProgressScope&, const Standard_Boolean) override {}

private:
  OcctImportOptions options_;
};

std::string label_entry(const TDF_Label& label) {
  if (label.IsNull()) {
    return {};
  }
  TCollection_AsciiString entry;
  TDF_Tool::Entry(label, entry);
  return std::string(entry.ToCString());
}

void append_bytes(rust::Vec<std::uint8_t>& destination, const std::string& bytes) {
  destination.reserve(bytes.size());
  for (const unsigned char byte : bytes) {
    destination.push_back(static_cast<std::uint8_t>(byte));
  }
}

void append_transform(rust::Vec<double>& destination, const gp_Trsf& transform) {
  destination.reserve(16);
  for (Standard_Integer row = 1; row <= 3; ++row) {
    for (Standard_Integer column = 1; column <= 4; ++column) {
      destination.push_back(transform.Value(row, column));
    }
  }
  destination.push_back(0.0);
  destination.push_back(0.0);
  destination.push_back(0.0);
  destination.push_back(1.0);
}

std::uint64_t append_definition(
    OcctExactShapePayload& result,
    const TopoDS_Shape& definition_shape,
    const std::string& definition_key,
    std::unordered_map<std::string, std::uint64_t>& definition_indices,
    std::uint64_t& exact_byte_total,
    const OcctImportOptions& options) {
  const auto existing = definition_indices.find(definition_key);
  if (existing != definition_indices.end()) {
    return existing->second;
  }
  TopoDS_Shape canonical_definition = definition_shape;
  canonical_definition.Location(TopLoc_Location());
  const std::string bytes = serialize_exact_shape(canonical_definition, options);
  if (static_cast<std::uint64_t>(bytes.size()) >
      options.max_exact_representation_bytes - exact_byte_total) {
    throw std::runtime_error(
        "OCCT exact representation exceeded its aggregate byte budget while serializing definitions");
  }
  exact_byte_total += static_cast<std::uint64_t>(bytes.size());

  OcctExactDefinitionPayload payload;
  payload.definition_index = static_cast<std::uint64_t>(result.definitions.size()) + 1;
  append_bytes(payload.representation, bytes);
  result.definitions.push_back(std::move(payload));
  definition_indices.emplace(definition_key, result.definitions.back().definition_index);
  return result.definitions.back().definition_index;
}

void append_occurrence(
    OcctExactShapePayload& result,
    const TDF_Label& occurrence_label,
    std::uint64_t parent_occurrence_index,
    const TopLoc_Location& parent_location,
    std::vector<std::string> path,
    std::set<std::string>& active_definitions,
    std::unordered_map<std::string, std::uint64_t>& definition_indices,
    std::uint64_t& exact_byte_total,
    const BRepTools_ShapeSet& shape_set,
    ExactIdentityContext& identity_context,
    std::uint32_t depth,
    const OcctImportOptions& options) {
  check_cancelled(options);
  if (occurrence_label.IsNull() || depth > 256) {
    throw std::runtime_error("OCCT exact assembly exceeds its occurrence depth budget");
  }

  TDF_Label definition_label = occurrence_label;
  TDF_Label referred;
  if (XCAFDoc_ShapeTool::IsReference(occurrence_label) &&
      XCAFDoc_ShapeTool::GetReferredShape(occurrence_label, referred) && !referred.IsNull()) {
    definition_label = referred;
  }
  const std::string occurrence_segment = label_entry(occurrence_label);
  const std::string definition_key = label_entry(definition_label);
  if (occurrence_segment.empty() || definition_key.empty()) {
    throw std::runtime_error("OCCT exact assembly contains an unlabeled occurrence");
  }
  if (!active_definitions.insert(definition_key).second) {
    throw std::runtime_error("OCCT exact assembly contains a cyclic definition graph");
  }
  path.push_back(occurrence_segment);

  TopoDS_Shape definition_shape = XCAFDoc_ShapeTool::GetShape(definition_label);
  if (definition_shape.IsNull()) {
    throw std::runtime_error("OCCT exact assembly definition has no shape");
  }
  const TopLoc_Location local_location = XCAFDoc_ShapeTool::GetLocation(occurrence_label);
  const TopLoc_Location occurrence_location = parent_location * local_location;
  const std::uint64_t definition_index = append_definition(
      result, definition_shape, definition_key, definition_indices, exact_byte_total, options);

  OcctExactOccurrencePayload occurrence;
  occurrence.occurrence_index = static_cast<std::uint64_t>(result.occurrences.size()) + 1;
  occurrence.parent_occurrence_index = parent_occurrence_index;
  for (const std::string& segment : path) {
    occurrence.path_segments.push_back(segment);
  }
  occurrence.definition_index = definition_index;
  append_transform(occurrence.transform, local_location.Transformation());
  const std::uint64_t occurrence_index = occurrence.occurrence_index;
  result.occurrences.push_back(std::move(occurrence));

  TDF_LabelSequence components;
  if (XCAFDoc_ShapeTool::IsAssembly(definition_label)) {
    if (!XCAFDoc_ShapeTool::GetComponents(definition_label, components, Standard_False) ||
        components.Length() == 0) {
      throw std::runtime_error("OCCT exact assembly definition contains no components");
    }
    for (Standard_Integer index = 1; index <= components.Length(); ++index) {
      append_occurrence(result,
                        components.Value(index),
                        occurrence_index,
                        occurrence_location,
                        path,
                        active_definitions,
                        definition_indices,
                        exact_byte_total,
                        shape_set,
                        identity_context,
                        depth + 1,
                        options);
    }
  } else {
    TopoDS_Shape located_shape = XCAFDoc_ShapeTool::GetShape(occurrence_label);
    if (located_shape.IsNull()) {
      throw std::runtime_error("OCCT exact assembly occurrence has no shape");
    }
    located_shape.Move(parent_location);
    Standard_Integer body_shape_key = shape_set.Index(located_shape);
    if (body_shape_key <= 0) {
      for (Standard_Integer index = 1; index <= shape_set.NbShapes(); ++index) {
        if (shape_set.Shape(index).IsPartner(located_shape)) {
          body_shape_key = index;
          break;
        }
      }
    }
    if (body_shape_key <= 0) {
      throw std::runtime_error(
          "OCCT exact assembly occurrence definition is absent from the canonical B-rep shape table");
    }
    append_exact_topology(result,
                          shape_set.Shape(body_shape_key),
                          shape_set,
                          identity_context,
                          occurrence_index,
                          options);
  }
  active_definitions.erase(definition_key);
}

} // namespace

std::string serialize_exact_shape(const TopoDS_Shape& shape,
                                  const OcctImportOptions& options) {
  std::ostringstream stream;
  CancelProgress progress(options);
  BRepTools::Write(shape, stream, progress.Start());
  check_cancelled(options);
  const std::string bytes = stream.str();
  if (bytes.empty()) {
    throw std::runtime_error("OCCT exact definition serialization produced no bytes");
  }
  return bytes;
}

void append_exact_occurrences(OcctExactShapePayload& result,
                              const TopoDS_Shape& root_shape,
                              const Handle(XCAFDoc_ShapeTool)& shape_tool,
                              bool has_xcaf,
                              const std::string& representation,
                              const OcctImportOptions& options) {
  BRepTools_ShapeSet shape_set(Standard_False);
  shape_set.Add(root_shape);
  ExactIdentityContext identity_context(result.healing_identity_work_bytes);

  OcctExactOccurrencePayload root;
  root.occurrence_index = 1;
  root.parent_occurrence_index = 0;
  root.path_segments.push_back("root");
  root.definition_index = 0;
  append_transform(root.transform, gp_Trsf());
  result.occurrences.push_back(std::move(root));

  if (!has_xcaf || shape_tool.IsNull()) {
    const Standard_Integer body_shape_key = shape_set.Index(root_shape);
    if (body_shape_key <= 0) {
      throw std::runtime_error("OCCT exact root shape is absent from its B-rep shape table");
    }
    append_exact_topology(result, root_shape, shape_set, identity_context, 1, options);
    return;
  }

  TDF_LabelSequence free_shapes;
  shape_tool->GetFreeShapes(free_shapes);
  std::set<std::string> active_definitions;
  std::unordered_map<std::string, std::uint64_t> definition_indices;
  std::uint64_t exact_byte_total = static_cast<std::uint64_t>(representation.size());
  for (Standard_Integer index = 1; index <= free_shapes.Length(); ++index) {
    append_occurrence(result,
                      free_shapes.Value(index),
                      1,
                      TopLoc_Location(),
                      {"root"},
                      active_definitions,
                      definition_indices,
                      exact_byte_total,
                      shape_set,
                      identity_context,
                      0,
                      options);
  }
}

} // namespace occt_backend
} // namespace runmat_geometry_io
