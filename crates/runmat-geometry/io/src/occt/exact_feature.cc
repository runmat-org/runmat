#include "runmat-geometry-io/src/occt/ffi.rs.h"
#include "exact_identity.hxx"
#include "occt_bridge.hxx"

#include <BRepAlgoAPI_Common.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepBuilderAPI_GTransform.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <BRepTools.hxx>
#include <BRepTools_History.hxx>
#include <BRepTools_ShapeSet.hxx>
#include <BRep_Builder.hxx>
#include <Message_ProgressRange.hxx>
#include <ShapeBuild_ReShape.hxx>
#include <ShapeFix_Shape.hxx>
#include <ShapeUpgrade_UnifySameDomain.hxx>
#include <TopExp.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopTools_ListIteratorOfListOfShape.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_Ax2.hxx>
#include <gp_Dir.hxx>
#include <gp_GTrsf.hxx>
#include <gp_Mat.hxx>
#include <gp_Pnt.hxx>
#include <gp_XYZ.hxx>

#include <array>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace runmat_geometry_io {
namespace occt_backend {
namespace {

struct HistoryRelation {
  OcctFeatureRelationKind kind;
  TopoDS_Shape target;
};

struct HistorySource {
  std::uint64_t operand_index;
  OcctFeatureEntityKind kind;
  TopoDS_Shape source;
  std::array<std::uint8_t, 32> source_digest;
  bool deleted = false;
  std::vector<HistoryRelation> relations;
};

struct FeatureExecution {
  TopoDS_Shape shape;
  std::vector<TopoDS_Shape> operands;
  std::vector<HistorySource> history;
  std::uint64_t identity_work_bytes = 0;
};

constexpr std::array<std::pair<TopAbs_ShapeEnum, OcctFeatureEntityKind>, 4>
    kSelectableKinds{{
        {TopAbs_VERTEX, OcctFeatureEntityKind::Vertex},
        {TopAbs_EDGE, OcctFeatureEntityKind::Edge},
        {TopAbs_FACE, OcctFeatureEntityKind::Face},
        {TopAbs_SOLID, OcctFeatureEntityKind::Solid},
    }};

void check_cancelled(const OcctImportOptions& options) {
  if (options.cancel_token_id != 0 && occt_import_cancelled(options.cancel_token_id)) {
    throw std::runtime_error("OCCT exact feature execution cancelled");
  }
}

void require_count(rust::Slice<const double> values,
                   std::size_t expected,
                   const char* field) {
  if (values.size() != expected) {
    throw std::runtime_error(std::string(field) + " has an invalid value count");
  }
  for (double value : values) {
    if (!std::isfinite(value)) {
      throw std::runtime_error(std::string(field) + " contains a non-finite value");
    }
  }
}

void require_operands(const rust::Vec<OcctFeatureOperand>& operands,
                      std::size_t minimum,
                      const char* operation) {
  if (operands.size() < minimum) {
    throw std::runtime_error(std::string(operation) + " has too few operands");
  }
}

TopoDS_Shape read_operand(const OcctFeatureOperand& operand) {
  if (operand.representation.empty()) {
    throw std::runtime_error("OCCT feature operand is empty");
  }
  std::string bytes(reinterpret_cast<const char*>(operand.representation.data()),
                    operand.representation.size());
  std::istringstream stream(bytes);
  TopoDS_Shape shape;
  BRep_Builder builder;
  BRepTools::Read(shape, stream, builder, Message_ProgressRange());
  if (shape.IsNull()) {
    throw std::runtime_error("OCCT feature operand is not a valid BREP shape");
  }
  return shape;
}

std::vector<TopoDS_Shape> read_operands(const rust::Vec<OcctFeatureOperand>& operands) {
  std::vector<TopoDS_Shape> result;
  result.reserve(operands.size());
  for (const OcctFeatureOperand& operand : operands) {
    result.push_back(read_operand(operand));
  }
  return result;
}

gp_Dir direction(rust::Slice<const double> values, std::size_t offset) {
  const double x = values[offset];
  const double y = values[offset + 1];
  const double z = values[offset + 2];
  if (x * x + y * y + z * z <= 0.0) {
    throw std::runtime_error("OCCT feature direction must be nonzero");
  }
  return gp_Dir(x, y, z);
}

gp_Pnt point(rust::Slice<const double> values, std::size_t offset) {
  return gp_Pnt(values[offset], values[offset + 1], values[offset + 2]);
}

gp_GTrsf general_transform(rust::Slice<const double> matrices, std::size_t offset) {
  const double tolerance = 1.0e-12;
  if (std::abs(matrices[offset + 12]) > tolerance ||
      std::abs(matrices[offset + 13]) > tolerance ||
      std::abs(matrices[offset + 14]) > tolerance ||
      std::abs(matrices[offset + 15] - 1.0) > tolerance) {
    throw std::runtime_error("OCCT feature transform must be affine");
  }
  gp_GTrsf transform;
  transform.SetVectorialPart(gp_Mat(matrices[offset], matrices[offset + 1],
                                    matrices[offset + 2], matrices[offset + 4],
                                    matrices[offset + 5], matrices[offset + 6],
                                    matrices[offset + 8], matrices[offset + 9],
                                    matrices[offset + 10]));
  transform.SetTranslationPart(gp_XYZ(matrices[offset + 3],
                                      matrices[offset + 7],
                                      matrices[offset + 11]));
  return transform;
}

void append_source_inventory(std::vector<HistorySource>& history,
                             std::uint64_t operand_index,
                             const TopoDS_Shape& source,
                             const OcctImportOptions& options,
                             std::uint64_t& identity_work_bytes) {
  for (const auto& [shape_kind, feature_kind] : kSelectableKinds) {
    TopTools_IndexedMapOfShape shapes;
    TopExp::MapShapes(source, shape_kind, shapes);
    for (Standard_Integer index = 1; index <= shapes.Extent(); ++index) {
      const TopoDS_Shape& entity = shapes(index);
      history.push_back(HistorySource{operand_index,
                                      feature_kind,
                                      entity,
                                      exact_identity_digest(
                                          entity, options, identity_work_bytes),
                                      false,
                                      {}});
    }
  }
}

void append_relations(HistorySource& source,
                      const TopTools_ListOfShape& targets,
                      OcctFeatureRelationKind kind) {
  for (TopTools_ListIteratorOfListOfShape iterator(targets); iterator.More(); iterator.Next()) {
    const TopoDS_Shape& target = iterator.Value();
    for (const auto& [shape_kind, ignored] : kSelectableKinds) {
      if (target.ShapeType() == shape_kind) {
        source.relations.push_back(HistoryRelation{kind, target});
        break;
      }
    }
  }
}

std::vector<TopoDS_Shape> repaired_descendants(
    const TopoDS_Shape& source,
    const Handle(ShapeBuild_ReShape)& context,
    const TopTools_IndexedMapOfShape& final_shapes) {
  const TopAbs_ShapeEnum kind = source.ShapeType();
  const TopoDS_Shape replacement = context->Apply(source);
  if (replacement.IsNull()) {
    return {};
  }
  TopTools_IndexedMapOfShape candidates;
  if (replacement.ShapeType() == kind) {
    candidates.Add(replacement);
  }
  TopExp::MapShapes(replacement, kind, candidates);
  std::vector<TopoDS_Shape> result;
  for (Standard_Integer index = 1; index <= candidates.Extent(); ++index) {
    if (final_shapes.Contains(candidates(index))) {
      result.push_back(candidates(index));
    }
  }
  return result;
}

void repair_boolean_result(FeatureExecution& execution) {
  const TopoDS_Shape before = execution.shape;
  TopTools_IndexedMapOfShape before_shapes;
  TopExp::MapShapes(before, before_shapes);
  Handle(ShapeBuild_ReShape) context = new ShapeBuild_ReShape();
  ShapeFix_Shape repair(before);
  repair.SetContext(context);
  repair.Perform();
  const TopoDS_Shape after = repair.Shape();
  if (after.IsNull()) {
    throw std::runtime_error("OCCT Boolean topology repair produced a null shape");
  }
  TopTools_IndexedMapOfShape final_shapes;
  TopExp::MapShapes(after, final_shapes);

  for (HistorySource& source : execution.history) {
    std::vector<HistoryRelation> candidates = source.relations;
    if (before_shapes.Contains(source.source)) {
      candidates.push_back(
          HistoryRelation{OcctFeatureRelationKind::Modified, source.source});
    }
    source.relations.clear();
    for (const HistoryRelation& candidate : candidates) {
      for (const TopoDS_Shape& target :
           repaired_descendants(candidate.target, context, final_shapes)) {
        source.relations.push_back(HistoryRelation{candidate.kind, target});
      }
    }
    source.deleted = !final_shapes.Contains(source.source) && source.relations.empty();
  }
  execution.shape = after;
}

void compose_kernel_history(FeatureExecution& execution,
                            const TopoDS_Shape& after,
                            const Handle(BRepTools_History)& history) {
  if (history.IsNull() || after.IsNull()) {
    throw std::runtime_error("OCCT Boolean post-processing omitted topology history");
  }
  TopTools_IndexedMapOfShape before_shapes;
  TopExp::MapShapes(execution.shape, before_shapes);
  TopTools_IndexedMapOfShape final_shapes;
  TopExp::MapShapes(after, final_shapes);
  for (HistorySource& source : execution.history) {
    std::vector<HistoryRelation> candidates = source.relations;
    if (before_shapes.Contains(source.source)) {
      candidates.push_back(
          HistoryRelation{OcctFeatureRelationKind::Modified, source.source});
    }
    source.relations.clear();
    for (const HistoryRelation& candidate : candidates) {
      const TopTools_ListOfShape& modified = history->Modified(candidate.target);
      const TopTools_ListOfShape& generated = history->Generated(candidate.target);
      if (modified.IsEmpty() && !history->IsRemoved(candidate.target) &&
          final_shapes.Contains(candidate.target)) {
        source.relations.push_back(candidate);
      }
      append_relations(source, modified, candidate.kind);
      append_relations(source, generated, OcctFeatureRelationKind::Generated);
    }
    source.deleted = !final_shapes.Contains(source.source) && source.relations.empty();
  }
  execution.shape = after;
}

void unify_boolean_result(FeatureExecution& execution) {
  ShapeUpgrade_UnifySameDomain unify(
      execution.shape, Standard_True, Standard_True, Standard_True);
  unify.Build();
  compose_kernel_history(execution, unify.Shape(), unify.History());
}

template <typename Algorithm>
FeatureExecution execute_boolean_algorithm(Algorithm& algorithm,
                                           std::vector<TopoDS_Shape> operands,
                                           const OcctImportOptions& options,
                                           const char* operation) {
  FeatureExecution execution{TopoDS_Shape(), std::move(operands), {}, 0};
  for (std::size_t operand_index = 0; operand_index < execution.operands.size();
       ++operand_index) {
    append_source_inventory(execution.history,
                            static_cast<std::uint64_t>(operand_index),
                            execution.operands[operand_index],
                            options,
                            execution.identity_work_bytes);
  }
  TopTools_ListOfShape arguments;
  arguments.Append(execution.operands.front());
  TopTools_ListOfShape tools;
  for (std::size_t index = 1; index < execution.operands.size(); ++index) {
    tools.Append(execution.operands[index]);
  }
  algorithm.SetArguments(arguments);
  algorithm.SetTools(tools);
  algorithm.SetNonDestructive(Standard_True);
  algorithm.SetCheckInverted(Standard_True);
  algorithm.SetToFillHistory(Standard_True);
  algorithm.Build();
  if (!algorithm.IsDone() || algorithm.Shape().IsNull()) {
    throw std::runtime_error(std::string("OCCT ") + operation + " did not complete");
  }
  // OCCT composes this simplification into the Boolean algorithm's own history. Subsequent
  // unification and repair are composed explicitly below before any persistent identity crosses
  // the FFI boundary.
  algorithm.SimplifyResult();
  execution.shape = algorithm.Shape();
  for (HistorySource& source : execution.history) {
    source.deleted = algorithm.IsDeleted(source.source);
    append_relations(source,
                     algorithm.Modified(source.source),
                     OcctFeatureRelationKind::Modified);
    append_relations(source,
                     algorithm.Generated(source.source),
                     OcctFeatureRelationKind::Generated);
  }
  unify_boolean_result(execution);
  repair_boolean_result(execution);
  return execution;
}

FeatureExecution execute_boolean(OcctFeatureKind kind,
                                 const rust::Vec<OcctFeatureOperand>& operands,
                                 const OcctImportOptions& options) {
  require_operands(operands, 2, "boolean feature");
  std::vector<TopoDS_Shape> shapes = read_operands(operands);
  if (kind == OcctFeatureKind::Union) {
    BRepAlgoAPI_Fuse algorithm;
    return execute_boolean_algorithm(algorithm, std::move(shapes), options, "union");
  }
  if (kind == OcctFeatureKind::Subtract) {
    BRepAlgoAPI_Cut algorithm;
    return execute_boolean_algorithm(algorithm, std::move(shapes), options, "subtraction");
  }
  BRepAlgoAPI_Common algorithm;
  return execute_boolean_algorithm(algorithm, std::move(shapes), options, "intersection");
}

void append_transform_history(std::vector<HistorySource>& history,
                              BRepBuilderAPI_GTransform& builder) {
  for (HistorySource& source : history) {
    append_relations(source,
                     builder.Modified(source.source),
                     OcctFeatureRelationKind::Modified);
  }
}

FeatureExecution execute_transform(rust::Slice<const double> matrices,
                                   const rust::Vec<OcctFeatureOperand>& operands,
                                   const OcctImportOptions& options) {
  require_operands(operands, 1, "transform");
  require_count(matrices, 16, "transform matrix");
  std::vector<TopoDS_Shape> shapes = read_operands(operands);
  FeatureExecution execution;
  execution.operands = shapes;
  append_source_inventory(
      execution.history, 0, shapes[0], options, execution.identity_work_bytes);
  BRepBuilderAPI_GTransform builder(shapes[0], general_transform(matrices, 0), Standard_True);
  if (!builder.IsDone()) {
    throw std::runtime_error("OCCT general transform did not complete");
  }
  execution.shape = builder.Shape();
  append_transform_history(execution.history, builder);
  return execution;
}

FeatureExecution execute_instances(rust::Slice<const double> matrices,
                                   const rust::Vec<OcctFeatureOperand>& operands,
                                   const OcctImportOptions& options) {
  require_operands(operands, 1, "instance");
  if (matrices.empty() || matrices.size() % 16 != 0) {
    throw std::runtime_error("OCCT instance transforms must contain 16 values each");
  }
  std::vector<TopoDS_Shape> shapes = read_operands(operands);
  FeatureExecution execution;
  execution.operands = shapes;
  append_source_inventory(
      execution.history, 0, shapes[0], options, execution.identity_work_bytes);
  TopoDS_Compound compound;
  BRep_Builder compound_builder;
  compound_builder.MakeCompound(compound);
  for (std::size_t offset = 0; offset < matrices.size(); offset += 16) {
    BRepBuilderAPI_GTransform builder(
        shapes[0], general_transform(matrices, offset), Standard_True);
    if (!builder.IsDone()) {
      throw std::runtime_error("OCCT instance transform did not complete");
    }
    compound_builder.Add(compound, builder.Shape());
    append_transform_history(execution.history, builder);
  }
  execution.shape = compound;
  return execution;
}

FeatureExecution execute_feature(OcctFeatureKind kind,
                                 rust::Slice<const double> scalars,
                                 rust::Slice<const double> matrices,
                                 const rust::Vec<OcctFeatureOperand>& operands,
                                 const OcctImportOptions& options) {
  switch (kind) {
    case OcctFeatureKind::Box:
      require_count(scalars, 3, "box");
      return FeatureExecution{
          BRepPrimAPI_MakeBox(scalars[0], scalars[1], scalars[2]).Shape(), {}, {}, 0};
    case OcctFeatureKind::Cylinder:
      require_count(scalars, 8, "cylinder");
      return FeatureExecution{
          BRepPrimAPI_MakeCylinder(
              gp_Ax2(point(scalars, 5), direction(scalars, 2)), scalars[0], scalars[1])
              .Shape(),
          {},
          {},
          0};
    case OcctFeatureKind::Sphere:
      require_count(scalars, 4, "sphere");
      return FeatureExecution{
          BRepPrimAPI_MakeSphere(point(scalars, 1), scalars[0]).Shape(), {}, {}, 0};
    case OcctFeatureKind::Transform:
      return execute_transform(matrices, operands, options);
    case OcctFeatureKind::Instance:
      return execute_instances(matrices, operands, options);
    case OcctFeatureKind::Union:
    case OcctFeatureKind::Subtract:
    case OcctFeatureKind::Intersect:
      return execute_boolean(kind, operands, options);
  }
  throw std::runtime_error("unsupported OCCT exact feature kind");
}

OcctFeatureEntityKind entity_kind(const TopoDS_Shape& shape) {
  for (const auto& [shape_kind, feature_kind] : kSelectableKinds) {
    if (shape.ShapeType() == shape_kind) {
      return feature_kind;
    }
  }
  throw std::runtime_error("OCCT feature history contains non-selectable topology");
}

void append_digest(rust::Vec<std::uint8_t>& destination,
                   const TopoDS_Shape& shape,
                   const BRepTools_ShapeSet& shape_set,
                   ExactIdentityContext& identity_context,
                   const OcctImportOptions& options) {
  const Standard_Integer key = shape_set.Index(shape);
  if (key <= 0) {
    throw std::runtime_error("OCCT feature history shape is absent from its canonical table");
  }
  identity_context.append(destination,
                          shape_set.Shape(key),
                          static_cast<std::uint64_t>(key),
                          options);
}

OcctFeatureExecutionPayload encode_execution(FeatureExecution execution,
                                             const OcctImportOptions& options) {
  check_cancelled(options);
  if (execution.shape.IsNull()) {
    throw std::runtime_error("OCCT feature produced a null shape");
  }
  BRepTools::Clean(execution.shape);
  if (!BRepCheck_Analyzer(execution.shape, Standard_True).IsValid()) {
    throw std::runtime_error("OCCT feature produced an invalid BREP shape");
  }

  TopTools_IndexedMapOfShape result_shapes;
  TopExp::MapShapes(execution.shape, result_shapes);
  BRepTools_ShapeSet shape_set(Standard_False);
  for (const TopoDS_Shape& operand : execution.operands) {
    shape_set.Add(operand);
  }
  shape_set.Add(execution.shape);
  ExactIdentityContext identity_context(execution.identity_work_bytes);

  OcctFeatureExecutionPayload payload;
  for (const HistorySource& source : execution.history) {
    check_cancelled(options);
    OcctFeatureSourcePayload encoded_source;
    encoded_source.operand_index = source.operand_index;
    encoded_source.kind = source.kind;
    for (const std::uint8_t byte : source.source_digest) {
      encoded_source.identity_digest.push_back(byte);
    }
    encoded_source.deleted = source.deleted;
    payload.sources.push_back(std::move(encoded_source));

    bool has_surviving_shape = result_shapes.Contains(source.source);
    for (const HistoryRelation& relation : source.relations) {
      if (!result_shapes.Contains(relation.target)) {
        throw std::runtime_error(
            "OCCT feature history names a target absent from the final result");
      }
      OcctFeatureRelationPayload encoded_relation;
      encoded_relation.operand_index = source.operand_index;
      encoded_relation.source_kind = source.kind;
      for (const std::uint8_t byte : source.source_digest) {
        encoded_relation.source_identity_digest.push_back(byte);
      }
      encoded_relation.result_kind = entity_kind(relation.target);
      append_digest(encoded_relation.result_identity_digest,
                    relation.target,
                    shape_set,
                    identity_context,
                    options);
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
    rust::Vec<OcctFeatureOperand> operands,
    OcctImportOptions options) {
  if (options.max_exact_representation_bytes == 0 ||
      options.max_exact_identity_bytes == 0) {
    throw std::runtime_error("OCCT feature byte bounds must be positive");
  }
  check_cancelled(options);
  return encode_execution(execute_feature(kind, scalars, matrices, operands, options), options);
}

}  // namespace occt_backend
}  // namespace runmat_geometry_io
