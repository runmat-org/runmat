#include "runmat-geometry-io/src/occt/ffi.rs.h"
#include "runmat-geometry-io/src/occt/exact_assembly.hxx"
#include "runmat-geometry-io/src/occt/exact_xcaf_healing.hxx"

#include <BRepCheck_Analyzer.hxx>
#include <TCollection_AsciiString.hxx>
#include <TDF_Label.hxx>
#include <TDF_LabelSequence.hxx>
#include <TDF_Tool.hxx>
#include <TopLoc_Location.hxx>

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace runmat_geometry_io {
namespace occt_backend {
namespace {

struct DefinitionUses {
  TDF_Label label;
  std::vector<std::vector<std::string>> paths;
};

std::string label_entry(const TDF_Label& label) {
  TCollection_AsciiString entry;
  TDF_Tool::Entry(label, entry);
  return std::string(entry.ToCString());
}

void check_cancelled(const OcctImportOptions& options) {
  if (options.cancel_token_id != 0 && occt_import_cancelled(options.cancel_token_id)) {
    throw std::runtime_error("OCCT CAD import cancelled");
  }
}

void collect_definition_uses(
    const TDF_Label& occurrence_label,
    std::vector<std::string> path,
    std::set<std::string>& active_definitions,
    std::map<std::string, DefinitionUses>& definitions,
    const OcctImportOptions& options) {
  check_cancelled(options);
  TDF_Label definition_label = occurrence_label;
  TDF_Label referred;
  if (XCAFDoc_ShapeTool::IsReference(occurrence_label) &&
      XCAFDoc_ShapeTool::GetReferredShape(occurrence_label, referred) &&
      !referred.IsNull()) {
    definition_label = referred;
  }
  const std::string occurrence_key = label_entry(occurrence_label);
  const std::string definition_key = label_entry(definition_label);
  if (occurrence_key.empty() || definition_key.empty()) {
    throw std::runtime_error("OCCT exact assembly contains an unlabeled occurrence");
  }
  if (!active_definitions.insert(definition_key).second) {
    throw std::runtime_error("OCCT exact assembly contains a cyclic definition graph");
  }
  path.push_back(occurrence_key);
  if (XCAFDoc_ShapeTool::IsAssembly(definition_label)) {
    TDF_LabelSequence components;
    if (!XCAFDoc_ShapeTool::GetComponents(
            definition_label, components, Standard_False) ||
        components.Length() == 0) {
      throw std::runtime_error("OCCT exact assembly definition contains no components");
    }
    for (Standard_Integer index = 1; index <= components.Length(); ++index) {
      collect_definition_uses(
          components.Value(index), path, active_definitions, definitions, options);
    }
  } else if (XCAFDoc_ShapeTool::IsSimpleShape(definition_label)) {
    DefinitionUses& uses = definitions[definition_key];
    uses.label = definition_label;
    uses.paths.push_back(std::move(path));
  } else {
    throw std::runtime_error("OCCT exact assembly leaf is not a simple shape");
  }
  active_definitions.erase(definition_key);
}

void append_occurrence_relations(const ExactHealingMutation& mutation,
                                 const DefinitionUses& uses,
                                 ExactXcafHealingResult& result) {
  for (const std::vector<std::string>& path : uses.paths) {
    for (const ExactHealingMutation::Relation& source : mutation.relations) {
      ExactHealingMutation::Relation relation = source;
      relation.path_segments = path;
      result.relations.push_back(std::move(relation));
    }
  }
}

void retain_maximum_displacement(const ExactHealingMutation& mutation,
                                 ExactXcafHealingResult& result) {
  if (mutation.maximum_displacement > result.maximum_displacement) {
    result.maximum_displacement = mutation.maximum_displacement;
    result.displacement_original = mutation.displacement_original;
    result.displacement_proposed = mutation.displacement_proposed;
  }
}

} // namespace

ExactXcafHealingResult heal_exact_xcaf_definitions(
    const Handle(XCAFDoc_ShapeTool)& shape_tool,
    const OcctImportOptions& options,
    const std::uint64_t initial_identity_work) {
  if (shape_tool.IsNull()) {
    throw std::runtime_error("OCCT XCAF healing requires a shape tool");
  }
  TDF_LabelSequence free_shapes;
  shape_tool->GetFreeShapes(free_shapes);
  std::map<std::string, DefinitionUses> definitions;
  std::set<std::string> active_definitions;
  for (Standard_Integer index = 1; index <= free_shapes.Length(); ++index) {
    collect_definition_uses(free_shapes.Value(index),
                            {"root"},
                            active_definitions,
                            definitions,
                            options);
  }
  if (definitions.empty()) {
    throw std::runtime_error("OCCT XCAF document contains no healable definitions");
  }

  ExactXcafHealingResult result;
  result.identity_work_bytes = initial_identity_work;
  const bool changes_topology = options.heal_duplicates || options.heal_sew ||
                                options.heal_gaps ||
                                options.heal_short_edges_and_sliver_faces;
  for (auto& [definition_key, uses] : definitions) {
    (void)definition_key;
    std::sort(uses.paths.begin(), uses.paths.end());
    if (changes_topology) {
      TDF_LabelSequence subshapes;
      if (XCAFDoc_ShapeTool::GetSubShapes(uses.label, subshapes) &&
          subshapes.Length() > 0) {
        throw std::runtime_error(
            "OCCT topology healing cannot preserve labeled XCAF subshapes");
      }
    }
    TopoDS_Shape definition_shape = XCAFDoc_ShapeTool::GetShape(uses.label);
    if (definition_shape.IsNull()) {
      throw std::runtime_error("OCCT XCAF definition has no shape");
    }
    definition_shape.Location(TopLoc_Location());
    TopoDS_Shape current = definition_shape;
    bool definition_changed = false;

    if (options.heal_duplicates) {
      const std::string before = serialize_exact_shape(current, options);
      ExactHealingMutation mutation = consolidate_exact_duplicates(
          current, options, result.identity_work_bytes);
      result.identity_work_bytes = mutation.identity_work_bytes;
      current = mutation.shape;
      const bool changed = before != serialize_exact_shape(current, options);
      result.duplicates_consolidated |= changed;
      definition_changed |= changed;
      result.post_duplicate_kernel_valid &=
          BRepCheck_Analyzer(current, Standard_True).IsValid();
    }
    if (options.heal_sew || options.heal_gaps) {
      const std::string before = serialize_exact_shape(current, options);
      ExactHealingMutation mutation = sew_exact_shape(
          current,
          options,
          options.maximum_healing_displacement,
          result.identity_work_bytes);
      result.identity_work_bytes = mutation.identity_work_bytes;
      current = mutation.shape;
      const bool changed = before != serialize_exact_shape(current, options);
      if (changed) {
        append_occurrence_relations(mutation, uses, result);
        retain_maximum_displacement(mutation, result);
      }
      result.sewn |= options.heal_sew && changed;
      result.gaps_repaired |= options.heal_gaps && changed;
      definition_changed |= changed;
      result.post_sewing_kernel_valid &=
          BRepCheck_Analyzer(current, Standard_True).IsValid();
    }
    if (options.heal_short_edges_and_sliver_faces) {
      bool short_edges_simplified = false;
      bool sliver_faces_simplified = false;
      ExactHealingMutation mutation = simplify_exact_small_topology(
          current,
          options,
          options.maximum_healing_displacement,
          result.identity_work_bytes,
          short_edges_simplified,
          sliver_faces_simplified);
      result.identity_work_bytes = mutation.identity_work_bytes;
      current = mutation.shape;
      if (mutation.changed) {
        append_occurrence_relations(mutation, uses, result);
        retain_maximum_displacement(mutation, result);
      }
      result.short_edges_simplified |= short_edges_simplified;
      result.sliver_faces_simplified |= sliver_faces_simplified;
      definition_changed |= mutation.changed;
      result.post_small_topology_kernel_valid &=
          BRepCheck_Analyzer(current, Standard_True).IsValid();
    }
    if (options.heal_orientation) {
      const std::string before = serialize_exact_shape(current, options);
      ExactHealingMutation mutation = repair_exact_orientation(
          current, options, result.identity_work_bytes);
      result.identity_work_bytes = mutation.identity_work_bytes;
      current = mutation.shape;
      const bool changed = before != serialize_exact_shape(current, options);
      result.orientation_repaired |= changed;
      definition_changed |= changed;
    }
    if (definition_changed) {
      shape_tool->SetShape(uses.label, current);
    }
    check_cancelled(options);
  }

  shape_tool->UpdateAssemblies();
  result.shape = shape_tool->GetOneShape();
  if (result.shape.IsNull()) {
    throw std::runtime_error("OCCT XCAF healing produced a null assembly shape");
  }
  result.changed = result.duplicates_consolidated || result.sewn ||
                   result.gaps_repaired || result.short_edges_simplified ||
                   result.sliver_faces_simplified || result.orientation_repaired;
  return result;
}

} // namespace occt_backend
} // namespace runmat_geometry_io
