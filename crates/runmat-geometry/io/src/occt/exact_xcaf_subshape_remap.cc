#include "runmat-geometry-io/src/occt/ffi.rs.h"
#include "runmat-geometry-io/src/occt/exact_xcaf_subshape_remap.hxx"

#include <TCollection_AsciiString.hxx>
#include <TDF_LabelSequence.hxx>
#include <TDF_Tool.hxx>
#include <TNaming_Builder.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp.hxx>
#include <TopTools_IndexedMapOfShape.hxx>

#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace runmat_geometry_io {
namespace occt_backend {
namespace {

struct LabeledSubshape {
  TDF_Label label;
  std::string entry;
  std::uint8_t kind = 0;
  ExactHealingDigest source_digest{};
  ExactHealingDigest target_digest{};
};

std::string label_entry(const TDF_Label& label) {
  TCollection_AsciiString entry;
  TDF_Tool::Entry(label, entry);
  return std::string(entry.ToCString());
}

std::uint8_t relation_kind(const TopAbs_ShapeEnum kind,
                           const std::string& entry) {
  switch (kind) {
    case TopAbs_VERTEX:
      return 0;
    case TopAbs_EDGE:
      return 1;
    case TopAbs_FACE:
      return 2;
    default:
      throw std::runtime_error(
          "RUNMAT_XCAF_REMAP|unsupported_kind|" + entry + "||");
  }
}

std::string digest_text(const ExactHealingDigest& digest) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(digest.size() * 2);
  for (const std::uint8_t byte : digest) {
    result.push_back(digits[byte >> 4]);
    result.push_back(digits[byte & 0x0f]);
  }
  return result;
}

bool empty_digest(const ExactHealingDigest& digest) {
  for (const std::uint8_t byte : digest) {
    if (byte != 0) {
      return false;
    }
  }
  return true;
}

std::vector<LabeledSubshape> collect_labeled_subshapes(
    const TDF_Label& definition_label,
    const OcctImportOptions& options,
    std::uint64_t& identity_work_bytes) {
  TDF_LabelSequence labels;
  if (!XCAFDoc_ShapeTool::GetSubShapes(definition_label, labels)) {
    return {};
  }
  std::vector<LabeledSubshape> result;
  result.reserve(static_cast<std::size_t>(labels.Length()));
  for (Standard_Integer index = 1; index <= labels.Length(); ++index) {
    const TDF_Label label = labels.Value(index);
    const std::string entry = label_entry(label);
    const TopoDS_Shape shape = XCAFDoc_ShapeTool::GetShape(label);
    if (entry.empty() || shape.IsNull()) {
      throw std::runtime_error(
          "OCCT healing found an invalid labeled XCAF subshape");
    }
    LabeledSubshape record;
    record.label = label;
    record.entry = entry;
    record.kind = relation_kind(shape.ShapeType(), entry);
    record.source_digest =
        persistent_healing_digest(shape, options, identity_work_bytes);
    result.push_back(std::move(record));
  }
  return result;
}

void resolve_targets(std::vector<LabeledSubshape>& labels,
                     const ExactHealingMutation& mutation) {
  std::map<ExactHealingDigest,
           std::vector<std::pair<std::string, ExactHealingDigest>>> target_owners;
  for (LabeledSubshape& label : labels) {
    std::set<ExactHealingDigest> candidates;
    bool deleted = false;
    for (const ExactHealingMutation::Relation& relation : mutation.relations) {
      if (relation.kind != label.kind ||
          relation.source_digest != label.source_digest) {
        continue;
      }
      if (empty_digest(relation.target_digest)) {
        deleted = true;
      } else {
        candidates.insert(relation.target_digest);
      }
    }
    if (deleted || candidates.empty()) {
      throw std::runtime_error(
          "RUNMAT_XCAF_REMAP|deleted|" + label.entry + "|" +
          digest_text(label.source_digest) + "|");
    }
    if (candidates.size() != 1) {
      std::ostringstream message;
      message << "RUNMAT_XCAF_REMAP|split|" << label.entry << '|'
              << digest_text(label.source_digest) << '|';
      bool first = true;
      for (const ExactHealingDigest& candidate : candidates) {
        if (!first) {
          message << ',';
        }
        first = false;
        message << digest_text(candidate);
      }
      throw std::runtime_error(message.str());
    }
    label.target_digest = *candidates.begin();
    target_owners[label.target_digest].push_back(
        {label.entry, label.source_digest});
  }
  for (const auto& [target, owners] : target_owners) {
    if (owners.size() > 1) {
      std::ostringstream message;
      message << "RUNMAT_XCAF_REMAP|merged|";
      for (std::size_t index = 0; index < owners.size(); ++index) {
        if (index != 0) {
          message << ',';
        }
        message << owners[index].first;
      }
      message << '|';
      for (std::size_t index = 0; index < owners.size(); ++index) {
        if (index != 0) {
          message << ',';
        }
        message << digest_text(owners[index].second);
      }
      message << '|' << digest_text(target);
      throw std::runtime_error(message.str());
    }
  }
}

std::map<ExactHealingDigest, TopoDS_Shape> target_shapes(
    const TopoDS_Shape& healed_definition,
    const std::vector<LabeledSubshape>& labels,
    const OcctImportOptions& options,
    std::uint64_t& identity_work_bytes) {
  std::set<std::uint8_t> kinds;
  for (const LabeledSubshape& label : labels) {
    kinds.insert(label.kind);
  }
  std::map<ExactHealingDigest, TopoDS_Shape> result;
  for (const std::uint8_t kind : kinds) {
    const TopAbs_ShapeEnum topology_kind =
        kind == 0 ? TopAbs_VERTEX : (kind == 1 ? TopAbs_EDGE : TopAbs_FACE);
    TopTools_IndexedMapOfShape shapes;
    TopExp::MapShapes(healed_definition, topology_kind, shapes);
    for (Standard_Integer index = 1; index <= shapes.Extent(); ++index) {
      const TopoDS_Shape& shape = shapes(index);
      const ExactHealingDigest digest =
          persistent_healing_digest(shape, options, identity_work_bytes);
      const auto [existing, inserted] = result.emplace(digest, shape);
      if (!inserted && !existing->second.IsSame(shape)) {
        throw std::runtime_error(
            "RUNMAT_XCAF_REMAP|ambiguous_target|||" + digest_text(digest));
      }
    }
  }
  return result;
}

} // namespace

void remap_exact_xcaf_subshapes(
    const Handle(XCAFDoc_ShapeTool)& shape_tool,
    const TDF_Label& definition_label,
    const TopoDS_Shape& healed_definition,
    const ExactHealingMutation& mutation,
    const OcctImportOptions& options,
    std::uint64_t& identity_work_bytes) {
  std::vector<LabeledSubshape> labels = collect_labeled_subshapes(
      definition_label, options, identity_work_bytes);
  shape_tool->SetShape(definition_label, healed_definition);
  if (labels.empty()) {
    return;
  }
  resolve_targets(labels, mutation);
  const std::map<ExactHealingDigest, TopoDS_Shape> targets = target_shapes(
      healed_definition, labels, options, identity_work_bytes);
  for (const LabeledSubshape& label : labels) {
    const auto target = targets.find(label.target_digest);
    if (target == targets.end()) {
      throw std::runtime_error(
          "RUNMAT_XCAF_REMAP|missing_target|" + label.entry + "||" +
          digest_text(label.target_digest));
    }
    TNaming_Builder(label.label).Generated(target->second);
  }
}

} // namespace occt_backend
} // namespace runmat_geometry_io
