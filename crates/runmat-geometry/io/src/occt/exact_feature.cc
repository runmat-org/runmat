#include "runmat-geometry-io/src/occt/ffi.rs.h"
#include "exact_feature_internal.hxx"
#include "occt_bridge.hxx"

#include <BRepCheck_Analyzer.hxx>
#include <BRepTools.hxx>
#include <BRepTools_ShapeSet.hxx>
#include <Message_ProgressRange.hxx>
#include <TopExp.hxx>
#include <TopTools_IndexedMapOfShape.hxx>

#include <array>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace runmat_geometry_io {
namespace occt_backend {
namespace {

constexpr std::array<std::pair<TopAbs_ShapeEnum, OcctFeatureEntityKind>, 4>
    kSelectableKinds{{
        {TopAbs_VERTEX, OcctFeatureEntityKind::Vertex},
        {TopAbs_EDGE, OcctFeatureEntityKind::Edge},
        {TopAbs_FACE, OcctFeatureEntityKind::Face},
        {TopAbs_SOLID, OcctFeatureEntityKind::Solid},
    }};

OcctFeatureEntityKind entity_kind(const TopoDS_Shape& shape) {
  for (const auto& [shape_kind, feature_kind] : kSelectableKinds) {
    if (shape.ShapeType() == shape_kind) {
      return feature_kind;
    }
  }
  throw std::runtime_error("OCCT feature history contains non-selectable topology");
}

OcctFeatureExecutionPayload encode_execution(FeatureExecution execution,
                                             const OcctImportOptions& options) {
  check_feature_cancelled(options);
  if (execution.shape.IsNull()) {
    throw std::runtime_error("OCCT feature produced a null shape");
  }
  BRepTools::Clean(execution.shape);
  if (!BRepCheck_Analyzer(execution.shape, Standard_True).IsValid()) {
    throw std::runtime_error("OCCT feature produced an invalid BREP shape");
  }

  TopTools_IndexedMapOfShape result_shapes;
  TopExp::MapShapes(execution.shape, result_shapes);
  // BRepTools preserves this result-only canonical shape table through the representation
  // round trip. History uses its private key to join to the imported result topology; public
  // persistent identity remains the standalone digest projected by the importer.
  BRepTools_ShapeSet result_shape_set(Standard_False);
  result_shape_set.Add(execution.shape);

  OcctFeatureExecutionPayload payload;
  for (const FeatureHistorySource& source : execution.history) {
    check_feature_cancelled(options);
    OcctFeatureSourcePayload encoded_source;
    encoded_source.operand_index = source.operand_index;
    encoded_source.kind = source.kind;
    for (const std::uint8_t byte : source.source_digest) {
      encoded_source.identity_digest.push_back(byte);
    }
    encoded_source.deleted = source.deleted;
    payload.sources.push_back(std::move(encoded_source));

    bool has_surviving_shape = result_shapes.Contains(source.source);
    for (const FeatureHistoryRelation& relation : source.relations) {
      const Standard_Integer result_index = result_shapes.FindIndex(relation.target);
      if (result_index <= 0) {
        throw std::runtime_error(
            "OCCT feature history names a target absent from the final result");
      }
      const TopoDS_Shape& final_target = result_shapes(result_index);
      OcctFeatureRelationPayload encoded_relation;
      encoded_relation.operand_index = source.operand_index;
      encoded_relation.source_kind = source.kind;
      for (const std::uint8_t byte : source.source_digest) {
        encoded_relation.source_identity_digest.push_back(byte);
      }
      encoded_relation.result_kind = entity_kind(final_target);
      const Standard_Integer result_shape_key = result_shape_set.Index(final_target);
      if (result_shape_key <= 0) {
        throw std::runtime_error(
            "OCCT feature history target is absent from the result shape table");
      }
      encoded_relation.result_shape_key =
          static_cast<std::uint64_t>(result_shape_key);
      encoded_relation.relation = relation.kind;
      payload.relations.push_back(std::move(encoded_relation));
      has_surviving_shape = true;
    }
    if (!source.deleted && !has_surviving_shape) {
      throw std::runtime_error(
          "OCCT feature history neither preserves, replaces, generates from, nor deletes a source entity");
    }
  }

  std::ostringstream stream;
  BRepTools::Write(execution.shape, stream, Message_ProgressRange());
  const std::string bytes = stream.str();
  if (bytes.empty() ||
      static_cast<std::uint64_t>(bytes.size()) > options.max_exact_representation_bytes) {
    throw std::runtime_error("OCCT feature representation exceeds its admitted byte bound");
  }
  payload.representation.reserve(bytes.size());
  for (unsigned char byte : bytes) {
    payload.representation.push_back(static_cast<std::uint8_t>(byte));
  }
  return payload;
}

}  // namespace

OcctFeatureExecutionPayload execute_exact_feature(
    OcctFeatureKind kind,
    rust::Slice<const double> scalars,
    rust::Slice<const double> matrices,
    rust::Vec<OcctFeatureSelection> selections,
    rust::Vec<OcctFeatureOperand> operands,
    OcctImportOptions options) {
  if (options.max_exact_representation_bytes == 0 ||
      options.max_exact_identity_bytes == 0) {
    throw std::runtime_error("OCCT feature byte bounds must be positive");
  }
  check_feature_cancelled(options);
  return encode_execution(
      execute_feature_operation(kind, scalars, matrices, selections, operands, options), options);
}

}  // namespace occt_backend
}  // namespace runmat_geometry_io
