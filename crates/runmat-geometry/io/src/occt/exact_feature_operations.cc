#include "exact_feature_internal.hxx"
#include "exact_identity.hxx"
#include "occt_bridge.hxx"

#include <BRepAlgoAPI_Common.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepBuilderAPI_GTransform.hxx>
#include <BRepBndLib.hxx>
#include <BRepFilletAPI_MakeChamfer.hxx>
#include <BRepFilletAPI_MakeFillet.hxx>
#include <BRepGProp.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <BRepPrimAPI_MakeRevol.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <BRepTools.hxx>
#include <BRep_Builder.hxx>
#include <Bnd_Box.hxx>
#include <GProp_GProps.hxx>
#include <Message_ProgressRange.hxx>
#include <TopExp.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <gp_Ax1.hxx>
#include <gp_Ax2.hxx>
#include <gp_Dir.hxx>
#include <gp_GTrsf.hxx>
#include <gp_Mat.hxx>
#include <gp_Pnt.hxx>
#include <gp_Vec.hxx>
#include <gp_XYZ.hxx>

#include <algorithm>
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

constexpr std::array<std::pair<TopAbs_ShapeEnum, OcctFeatureEntityKind>, 4>
    kSelectableKinds{{
        {TopAbs_VERTEX, OcctFeatureEntityKind::Vertex},
        {TopAbs_EDGE, OcctFeatureEntityKind::Edge},
        {TopAbs_FACE, OcctFeatureEntityKind::Face},
        {TopAbs_SOLID, OcctFeatureEntityKind::Solid},
    }};

void require_values(rust::Slice<const double> values,
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
                      std::size_t expected,
                      const char* operation) {
  if (operands.size() != expected) {
    throw std::runtime_error(std::string(operation) + " has an invalid operand count");
  }
}

void require_no_selections(const rust::Vec<OcctFeatureSelection>& selections,
                           const char* operation) {
  if (!selections.empty()) {
    throw std::runtime_error(std::string(operation) + " does not accept entity selections");
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

void append_source_inventory(FeatureExecution& execution,
                             std::uint64_t operand_index,
                             const TopoDS_Shape& source,
                             const OcctImportOptions& options) {
  for (const auto& [shape_kind, feature_kind] : kSelectableKinds) {
    TopTools_IndexedMapOfShape shapes;
    TopExp::MapShapes(source, shape_kind, shapes);
    for (Standard_Integer index = 1; index <= shapes.Extent(); ++index) {
      const TopoDS_Shape& entity = shapes(index);
      execution.history.push_back(FeatureHistorySource{
          operand_index,
          feature_kind,
          entity,
          exact_identity_digest(entity, options, execution.identity_work_bytes),
          false,
          {}});
    }
  }
}

FeatureExecution begin_execution(std::vector<TopoDS_Shape> operands,
                                 const OcctImportOptions& options) {
  FeatureExecution execution;
  execution.operands = std::move(operands);
  for (std::size_t index = 0; index < execution.operands.size(); ++index) {
    append_source_inventory(
        execution, static_cast<std::uint64_t>(index), execution.operands[index], options);
  }
  return execution;
}

void append_relations(FeatureHistorySource& source,
                      const TopTools_ListOfShape& targets,
                      OcctFeatureRelationKind kind) {
  for (TopTools_ListOfShape::Iterator iterator(targets); iterator.More(); iterator.Next()) {
    const TopoDS_Shape& target = iterator.Value();
    if (std::any_of(kSelectableKinds.begin(), kSelectableKinds.end(), [&](const auto& entry) {
          return target.ShapeType() == entry.first;
        })) {
      source.relations.push_back(FeatureHistoryRelation{kind, target});
    }
  }
}

template <typename Algorithm, typename Predicate>
void capture_algorithm_history(FeatureExecution& execution,
                               Algorithm& algorithm,
                               const TopoDS_Shape& result,
                               Predicate include) {
  TopTools_IndexedMapOfShape result_shapes;
  TopExp::MapShapes(result, result_shapes);
  for (FeatureHistorySource& source : execution.history) {
    if (!include(source)) {
      continue;
    }
    std::vector<FeatureHistoryRelation> candidates = source.relations;
    if (candidates.empty() && !source.deleted) {
      candidates.push_back(
          FeatureHistoryRelation{OcctFeatureRelationKind::Modified, source.source});
    }
    source.relations.clear();
    for (const FeatureHistoryRelation& candidate : candidates) {
      const TopTools_ListOfShape& modified = algorithm.Modified(candidate.target);
      const bool has_modified = !modified.IsEmpty();
      append_relations(source, modified, candidate.kind);
      const TopTools_ListOfShape& generated = algorithm.Generated(candidate.target);
      const bool has_generated = !generated.IsEmpty();
      append_relations(source, generated, OcctFeatureRelationKind::Generated);
      if (!has_modified && !has_generated && !algorithm.IsDeleted(candidate.target) &&
          result_shapes.Contains(candidate.target)) {
        source.relations.push_back(candidate);
      }
    }
    source.deleted = !result_shapes.Contains(source.source) && source.relations.empty();
  }
}

template <typename Algorithm>
void capture_all_history(FeatureExecution& execution,
                         Algorithm& algorithm,
                         const TopoDS_Shape& result) {
  capture_algorithm_history(
      execution, algorithm, result, [](const FeatureHistorySource&) { return true; });
}

template <typename Algorithm>
FeatureExecution run_boolean(Algorithm& algorithm,
                             FeatureExecution execution,
                             const std::vector<TopoDS_Shape>& boolean_shapes,
                             const char* operation) {
  TopTools_ListOfShape arguments;
  arguments.Append(boolean_shapes.front());
  TopTools_ListOfShape tools;
  for (std::size_t index = 1; index < boolean_shapes.size(); ++index) {
    tools.Append(boolean_shapes[index]);
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
  algorithm.SimplifyResult();
  execution.shape = algorithm.Shape();
  capture_all_history(execution, algorithm, execution.shape);
  return execution;
}

FeatureExecution execute_boolean(OcctFeatureKind kind,
                                 const rust::Vec<OcctFeatureOperand>& operands,
                                 const OcctImportOptions& options) {
  if (operands.size() < 2) {
    throw std::runtime_error("OCCT Boolean feature requires at least two operands");
  }
  std::vector<TopoDS_Shape> shapes = read_operands(operands);
  FeatureExecution execution = begin_execution(shapes, options);
  if (kind == OcctFeatureKind::Union) {
    BRepAlgoAPI_Fuse algorithm;
    return run_boolean(algorithm, std::move(execution), shapes, "union");
  }
  if (kind == OcctFeatureKind::Subtract) {
    BRepAlgoAPI_Cut algorithm;
    return run_boolean(algorithm, std::move(execution), shapes, "subtraction");
  }
  BRepAlgoAPI_Common algorithm;
  return run_boolean(algorithm, std::move(execution), shapes, "intersection");
}

FeatureExecution execute_transform(rust::Slice<const double> matrices,
                                   const rust::Vec<OcctFeatureOperand>& operands,
                                   const OcctImportOptions& options) {
  require_operands(operands, 1, "transform");
  require_values(matrices, 16, "transform matrix");
  std::vector<TopoDS_Shape> shapes = read_operands(operands);
  FeatureExecution execution = begin_execution(shapes, options);
  BRepBuilderAPI_GTransform builder(shapes[0], general_transform(matrices, 0), Standard_True);
  if (!builder.IsDone()) {
    throw std::runtime_error("OCCT general transform did not complete");
  }
  execution.shape = builder.Shape();
  capture_all_history(execution, builder, execution.shape);
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
  FeatureExecution execution = begin_execution(shapes, options);
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
    // Each placed copy is a new entity generated from the template: it keeps lineage to the
    // template and is introduced by this feature, so `boundary_introduced_by` an instance
    // selects the copies (and their images through later Booleans).
    for (FeatureHistorySource& source : execution.history) {
      append_relations(source,
                       builder.Modified(source.source),
                       OcctFeatureRelationKind::Generated);
    }
  }
  execution.shape = compound;
  return execution;
}

FeatureExecution execute_extrude(rust::Slice<const double> scalars,
                                 const rust::Vec<OcctFeatureOperand>& operands,
                                 const OcctImportOptions& options) {
  require_operands(operands, 1, "extrude");
  require_values(scalars, 4, "extrude");
  if (scalars[0] <= 0.0) {
    throw std::runtime_error("OCCT extrusion distance must be positive");
  }
  std::vector<TopoDS_Shape> shapes = read_operands(operands);
  FeatureExecution execution = begin_execution(shapes, options);
  const gp_Dir axis = direction(scalars, 1);
  BRepPrimAPI_MakePrism builder(
      shapes[0], gp_Vec(axis.XYZ()) * scalars[0], Standard_True, Standard_True);
  if (!builder.IsDone()) {
    throw std::runtime_error("OCCT extrusion did not complete");
  }
  execution.shape = builder.Shape();
  capture_all_history(execution, builder, execution.shape);
  return execution;
}

FeatureExecution execute_revolve(rust::Slice<const double> scalars,
                                 const rust::Vec<OcctFeatureOperand>& operands,
                                 const OcctImportOptions& options) {
  require_operands(operands, 1, "revolve");
  require_values(scalars, 7, "revolve");
  constexpr double pi = 3.141592653589793238462643383279502884;
  if (scalars[6] <= 0.0 || scalars[6] > 2.0 * pi) {
    throw std::runtime_error("OCCT revolution angle must be in (0, 2*pi]");
  }
  std::vector<TopoDS_Shape> shapes = read_operands(operands);
  FeatureExecution execution = begin_execution(shapes, options);
  BRepPrimAPI_MakeRevol builder(
      shapes[0], gp_Ax1(point(scalars, 3), direction(scalars, 0)), scalars[6], Standard_True);
  if (!builder.IsDone()) {
    throw std::runtime_error("OCCT revolution did not complete");
  }
  execution.shape = builder.Shape();
  capture_all_history(execution, builder, execution.shape);
  return execution;
}

void require_volume_overlap(const TopoDS_Shape& target,
                            const TopoDS_Shape& tool,
                            const char* operation) {
  BRepAlgoAPI_Common overlap(target, tool);
  overlap.Build();
  GProp_GProps overlap_properties;
  if (overlap.IsDone()) {
    BRepGProp::VolumeProperties(overlap.Shape(), overlap_properties);
  }
  if (overlap.IsDone() && overlap_properties.Mass() > 0.0) {
    return;
  }
  Bnd_Box target_bounds;
  Bnd_Box tool_bounds;
  BRepBndLib::Add(target, target_bounds);
  BRepBndLib::Add(tool, tool_bounds);
  Standard_Real tx_min, ty_min, tz_min, tx_max, ty_max, tz_max;
  Standard_Real px_min, py_min, pz_min, px_max, py_max, pz_max;
  target_bounds.Get(tx_min, ty_min, tz_min, tx_max, ty_max, tz_max);
  tool_bounds.Get(px_min, py_min, pz_min, px_max, py_max, pz_max);
  std::ostringstream message;
  message << "OCCT " << operation << " cutting tool does not overlap the target volume; target "
          << "bounds [" << tx_min << ", " << ty_min << ", " << tz_min << "]..[" << tx_max
          << ", " << ty_max << ", " << tz_max << "], tool bounds [" << px_min << ", "
          << py_min << ", " << pz_min << "]..[" << px_max << ", " << py_max << ", "
          << pz_max << "]";
  throw std::runtime_error(message.str());
}

FeatureExecution execute_hole(rust::Slice<const double> scalars,
                              const rust::Vec<OcctFeatureOperand>& operands,
                              const OcctImportOptions& options) {
  require_operands(operands, 1, "hole");
  require_values(scalars, 8, "hole");
  if (scalars[0] <= 0.0 || scalars[1] <= 0.0) {
    throw std::runtime_error("OCCT hole dimensions must be positive");
  }
  std::vector<TopoDS_Shape> sources = read_operands(operands);
  FeatureExecution execution = begin_execution(sources, options);
  const TopoDS_Shape tool = BRepPrimAPI_MakeCylinder(
                                gp_Ax2(point(scalars, 5), direction(scalars, 2)),
                                scalars[0], scalars[1])
                                .Shape();
  require_volume_overlap(sources[0], tool, "hole");
  const std::vector<TopoDS_Shape> boolean_shapes{sources[0], tool};
  BRepAlgoAPI_Cut algorithm;
  return run_boolean(algorithm, std::move(execution), boolean_shapes, "hole");
}

FeatureExecution execute_pocket(rust::Slice<const double> scalars,
                                const rust::Vec<OcctFeatureOperand>& operands,
                                const OcctImportOptions& options) {
  require_operands(operands, 2, "pocket");
  require_values(scalars, 4, "pocket");
  if (scalars[0] <= 0.0) {
    throw std::runtime_error("OCCT pocket depth must be positive");
  }
  std::vector<TopoDS_Shape> sources = read_operands(operands);
  FeatureExecution execution = begin_execution(sources, options);
  const gp_Dir axis = direction(scalars, 1);
  BRepPrimAPI_MakePrism sweep(
      sources[1], gp_Vec(axis.XYZ()) * scalars[0], Standard_True, Standard_True);
  if (!sweep.IsDone()) {
    throw std::runtime_error("OCCT pocket profile extrusion did not complete");
  }
  TopTools_IndexedMapOfShape swept_solids;
  TopExp::MapShapes(sweep.Shape(), TopAbs_SOLID, swept_solids);
  if (swept_solids.IsEmpty()) {
    throw std::runtime_error("OCCT pocket profile extrusion produced no solid cutting tool");
  }
  require_volume_overlap(sources[0], sweep.Shape(), "pocket");
  capture_algorithm_history(
      execution,
      sweep,
      sweep.Shape(),
      [](const FeatureHistorySource& source) { return source.operand_index == 1; });
  const std::vector<TopoDS_Shape> boolean_shapes{sources[0], sweep.Shape()};
  BRepAlgoAPI_Cut algorithm;
  return run_boolean(algorithm, std::move(execution), boolean_shapes, "pocket");
}

std::vector<TopoDS_Edge> selected_edges(
    FeatureExecution& execution,
    const rust::Vec<OcctFeatureSelection>& selections) {
  if (selections.empty()) {
    throw std::runtime_error("OCCT local edge operation requires a nonempty selection");
  }
  std::vector<TopoDS_Edge> result;
  result.reserve(selections.size());
  for (const OcctFeatureSelection& selection : selections) {
    if (selection.kind != OcctFeatureEntityKind::Edge || selection.identity_digest.size() != 32) {
      throw std::runtime_error("OCCT local operation received an invalid edge identity");
    }
    const auto match = std::find_if(
        execution.history.begin(), execution.history.end(), [&](const FeatureHistorySource& source) {
          return source.kind == OcctFeatureEntityKind::Edge &&
                 std::equal(source.source_digest.begin(),
                            source.source_digest.end(),
                            selection.identity_digest.begin());
        });
    if (match == execution.history.end()) {
      throw std::runtime_error("OCCT local operation edge is absent from the target topology");
    }
    if (std::any_of(result.begin(), result.end(), [&](const TopoDS_Edge& edge) {
          return edge.IsSame(match->source);
        })) {
      throw std::runtime_error("OCCT local operation repeats a selected edge");
    }
    result.push_back(TopoDS::Edge(match->source));
  }
  return result;
}

FeatureExecution execute_fillet_or_chamfer(
    OcctFeatureKind kind,
    rust::Slice<const double> scalars,
    const rust::Vec<OcctFeatureSelection>& selections,
    const rust::Vec<OcctFeatureOperand>& operands,
    const OcctImportOptions& options) {
  require_operands(operands, 1, kind == OcctFeatureKind::Fillet ? "fillet" : "chamfer");
  require_values(scalars, 1, kind == OcctFeatureKind::Fillet ? "fillet" : "chamfer");
  if (scalars[0] <= 0.0) {
    throw std::runtime_error("OCCT local edge-operation size must be positive");
  }
  std::vector<TopoDS_Shape> sources = read_operands(operands);
  FeatureExecution execution = begin_execution(sources, options);
  const std::vector<TopoDS_Edge> edges = selected_edges(execution, selections);
  if (kind == OcctFeatureKind::Fillet) {
    BRepFilletAPI_MakeFillet builder(sources[0]);
    for (const TopoDS_Edge& edge : edges) {
      builder.Add(scalars[0], edge);
    }
    builder.Build();
    if (!builder.IsDone()) {
      throw std::runtime_error("OCCT fillet did not complete");
    }
    execution.shape = builder.Shape();
    capture_all_history(execution, builder, execution.shape);
  } else {
    BRepFilletAPI_MakeChamfer builder(sources[0]);
    for (const TopoDS_Edge& edge : edges) {
      builder.Add(scalars[0], edge);
    }
    builder.Build();
    if (!builder.IsDone()) {
      throw std::runtime_error("OCCT chamfer did not complete");
    }
    execution.shape = builder.Shape();
    capture_all_history(execution, builder, execution.shape);
  }
  return execution;
}

}  // namespace

void check_feature_cancelled(const OcctImportOptions& options) {
  if (options.cancel_token_id != 0 && occt_import_cancelled(options.cancel_token_id)) {
    throw std::runtime_error("OCCT exact feature execution cancelled");
  }
}

FeatureExecution execute_feature_operation(
    OcctFeatureKind kind,
    rust::Slice<const double> scalars,
    rust::Slice<const double> matrices,
    const rust::Vec<OcctFeatureSelection>& selections,
    const rust::Vec<OcctFeatureOperand>& operands,
    const OcctImportOptions& options) {
  check_feature_cancelled(options);
  switch (kind) {
    case OcctFeatureKind::Box:
      require_no_selections(selections, "box");
      require_operands(operands, 0, "box");
      require_values(scalars, 3, "box");
      return FeatureExecution{
          BRepPrimAPI_MakeBox(scalars[0], scalars[1], scalars[2]).Shape(), {}, {}, 0};
    case OcctFeatureKind::Cylinder:
      require_no_selections(selections, "cylinder");
      require_operands(operands, 0, "cylinder");
      require_values(scalars, 8, "cylinder");
      return FeatureExecution{
          BRepPrimAPI_MakeCylinder(
              gp_Ax2(point(scalars, 5), direction(scalars, 2)), scalars[0], scalars[1])
              .Shape(),
          {},
          {},
          0};
    case OcctFeatureKind::Sphere:
      require_no_selections(selections, "sphere");
      require_operands(operands, 0, "sphere");
      require_values(scalars, 4, "sphere");
      return FeatureExecution{
          BRepPrimAPI_MakeSphere(point(scalars, 1), scalars[0]).Shape(), {}, {}, 0};
    case OcctFeatureKind::Transform:
      require_no_selections(selections, "transform");
      return execute_transform(matrices, operands, options);
    case OcctFeatureKind::Instance:
      require_no_selections(selections, "instance");
      return execute_instances(matrices, operands, options);
    case OcctFeatureKind::Extrude:
      require_no_selections(selections, "extrude");
      return execute_extrude(scalars, operands, options);
    case OcctFeatureKind::Revolve:
      require_no_selections(selections, "revolve");
      return execute_revolve(scalars, operands, options);
    case OcctFeatureKind::Union:
    case OcctFeatureKind::Subtract:
    case OcctFeatureKind::Intersect:
      require_no_selections(selections, "boolean feature");
      return execute_boolean(kind, operands, options);
    case OcctFeatureKind::Hole:
      require_no_selections(selections, "hole");
      return execute_hole(scalars, operands, options);
    case OcctFeatureKind::Pocket:
      require_no_selections(selections, "pocket");
      return execute_pocket(scalars, operands, options);
    case OcctFeatureKind::Fillet:
    case OcctFeatureKind::Chamfer:
      return execute_fillet_or_chamfer(kind, scalars, selections, operands, options);
  }
  throw std::runtime_error("unsupported OCCT exact feature kind");
}

}  // namespace occt_backend
}  // namespace runmat_geometry_io
