#include "runmat-geometry-io/src/occt/ffi.rs.h"
#include "runmat-geometry-io/src/occt/exact_healing.hxx"

#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <BRepBuilderAPI_Sewing.hxx>
#include <BRepTools.hxx>
#include <Message_ProgressIndicator.hxx>
#include <ShapeBuild_ReShape.hxx>
#include <ShapeFix_Shape.hxx>
#include <ShapeFix_Shell.hxx>
#include <ShapeFix_Solid.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Iterator.hxx>
#include <TopoDS.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <gp_Pnt.hxx>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace runmat_geometry_io {
namespace occt_backend {
namespace {

class HealingProgress final : public Message_ProgressIndicator {
public:
  explicit HealingProgress(const OcctImportOptions& options) : options_(options) {}

protected:
  Standard_Boolean UserBreak() override {
    return options_.cancel_token_id != 0 && occt_import_cancelled(options_.cancel_token_id);
  }

  void Show(const Message_ProgressScope&, const Standard_Boolean) override {}

private:
  OcctImportOptions options_;
};

using ShapeDigest = std::array<std::uint8_t, 32>;

ShapeDigest persistent_digest(const TopoDS_Shape& shape,
                              const OcctImportOptions& options,
                              std::uint64_t& byte_work) {
  const TopoDS_Shape canonical = shape.Oriented(TopAbs_FORWARD);
  std::ostringstream stream;
  HealingProgress progress(options);
  BRepTools::Write(canonical, stream, progress.Start());
  if (options.cancel_token_id != 0 && occt_import_cancelled(options.cancel_token_id)) {
    throw std::runtime_error("OCCT CAD import cancelled");
  }
  const std::string bytes = stream.str();
  if (bytes.empty() || byte_work > options.max_exact_identity_bytes ||
      static_cast<std::uint64_t>(bytes.size()) >
          options.max_exact_identity_bytes - byte_work) {
    throw std::runtime_error("OCCT exact persistent identity exceeded its byte-work budget");
  }
  byte_work += static_cast<std::uint64_t>(bytes.size());
  rust::Vec<std::uint8_t> digest = occt_exact_identity_digest(
      rust::Slice<const std::uint8_t>(
          reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
  if (digest.size() != 32) {
    throw std::runtime_error("OCCT healing identity digest has an invalid length");
  }
  ShapeDigest result;
  std::copy(digest.begin(), digest.end(), result.begin());
  return result;
}

std::vector<ShapeDigest> persistent_inventory(const TopoDS_Shape& root,
                                              const TopAbs_ShapeEnum kind,
                                              const OcctImportOptions& options,
                                              std::uint64_t& byte_work) {
  TopTools_IndexedMapOfShape shapes;
  TopExp::MapShapes(root, kind, shapes);
  std::vector<ShapeDigest> result;
  result.reserve(static_cast<std::size_t>(shapes.Extent()));
  for (Standard_Integer index = 1; index <= shapes.Extent(); ++index) {
    result.push_back(persistent_digest(shapes(index), options, byte_work));
  }
  std::sort(result.begin(), result.end());
  return result;
}

bool preserves_persistent_inventory(const TopoDS_Shape& before,
                                    const TopoDS_Shape& after,
                                    const OcctImportOptions& options,
                                    std::uint64_t& byte_work) {
  const TopAbs_ShapeEnum kinds[] = {
      TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX};
  for (const TopAbs_ShapeEnum kind : kinds) {
    if (persistent_inventory(before, kind, options, byte_work) !=
        persistent_inventory(after, kind, options, byte_work)) {
      return false;
    }
  }
  return true;
}

void append_relations(const TopoDS_Shape& before,
                      const TopoDS_Shape& after,
                      const TopAbs_ShapeEnum kind,
                      const std::uint8_t relation_kind,
                      const BRepBuilderAPI_Sewing& sewing,
                      const OcctImportOptions& options,
                      ExactHealingMutation& mutation) {
  TopTools_IndexedMapOfShape target_shapes;
  TopExp::MapShapes(after, kind, target_shapes);
  std::map<ShapeDigest, TopoDS_Shape> targets;
  for (Standard_Integer index = 1; index <= target_shapes.Extent(); ++index) {
    const TopoDS_Shape& target = target_shapes(index);
    targets.emplace(
        persistent_digest(target, options, mutation.identity_work_bytes), target);
  }

  TopTools_IndexedMapOfShape source_shapes;
  TopExp::MapShapes(before, kind, source_shapes);
  for (Standard_Integer index = 1; index <= source_shapes.Extent(); ++index) {
    const TopoDS_Shape& source = source_shapes(index);
    ExactHealingMutation::Relation relation;
    relation.kind = relation_kind;
    relation.path_segments = {"root"};
    relation.source_digest =
        persistent_digest(source, options, mutation.identity_work_bytes);
    TopoDS_Shape target;
    if (sewing.IsModifiedSubShape(source)) {
      target = sewing.ModifiedSubShape(source);
    } else {
      const auto existing = targets.find(relation.source_digest);
      if (existing != targets.end()) {
        target = existing->second;
      }
    }
    if (!target.IsNull()) {
      const ShapeDigest digest =
          persistent_digest(target, options, mutation.identity_work_bytes);
      if (targets.find(digest) != targets.end()) {
        relation.target_digest = digest;
      }
    }
    mutation.relations.push_back(relation);
  }
}

void append_context_relations(const TopoDS_Shape& before,
                              const TopoDS_Shape& after,
                              const TopAbs_ShapeEnum kind,
                              const std::uint8_t relation_kind,
                              const Handle(ShapeBuild_ReShape)& context,
                              const OcctImportOptions& options,
                              ExactHealingMutation& mutation) {
  TopTools_IndexedMapOfShape target_shapes;
  TopExp::MapShapes(after, kind, target_shapes);
  std::map<ShapeDigest, TopoDS_Shape> targets;
  for (Standard_Integer index = 1; index <= target_shapes.Extent(); ++index) {
    const TopoDS_Shape& target = target_shapes(index);
    targets.emplace(
        persistent_digest(target, options, mutation.identity_work_bytes), target);
  }

  TopTools_IndexedMapOfShape source_shapes;
  TopExp::MapShapes(before, kind, source_shapes);
  for (Standard_Integer index = 1; index <= source_shapes.Extent(); ++index) {
    const TopoDS_Shape& source = source_shapes(index);
    ExactHealingMutation::Relation relation;
    relation.kind = relation_kind;
    relation.path_segments = {"root"};
    relation.source_digest =
        persistent_digest(source, options, mutation.identity_work_bytes);
    TopoDS_Shape replacement;
    const Standard_Integer status = context->Status(source, replacement, Standard_True);
    if (status > 0 && !replacement.IsNull()) {
      TopTools_IndexedMapOfShape mapped;
      TopExp::MapShapes(replacement, kind, mapped);
      if (replacement.ShapeType() == kind) {
        mapped.Add(replacement);
      }
      TopoDS_Shape admitted;
      for (Standard_Integer mapped_index = 1; mapped_index <= mapped.Extent(); ++mapped_index) {
        const ShapeDigest digest = persistent_digest(
            mapped(mapped_index), options, mutation.identity_work_bytes);
        if (targets.find(digest) != targets.end()) {
          if (!admitted.IsNull()) {
            throw std::runtime_error(
                "OCCT small-topology repair produced unsupported split lineage");
          }
          admitted = mapped(mapped_index);
        }
      }
      if (!admitted.IsNull()) {
        relation.target_digest =
            persistent_digest(admitted, options, mutation.identity_work_bytes);
      }
    } else if (status == 0) {
      const auto existing = targets.find(relation.source_digest);
      if (existing != targets.end()) {
        relation.target_digest = relation.source_digest;
      }
    }
    mutation.relations.push_back(relation);
  }
}

std::vector<gp_Pnt> vertex_points(const TopoDS_Shape& shape) {
  TopTools_IndexedMapOfShape vertices;
  TopExp::MapShapes(shape, TopAbs_VERTEX, vertices);
  std::vector<gp_Pnt> result;
  result.reserve(static_cast<std::size_t>(vertices.Extent()));
  for (Standard_Integer index = 1; index <= vertices.Extent(); ++index) {
    result.push_back(BRep_Tool::Pnt(TopoDS::Vertex(vertices(index))));
  }
  return result;
}

void measure_vertex_displacement(const std::vector<gp_Pnt>& sources,
                                 const std::vector<gp_Pnt>& targets,
                                 ExactHealingMutation& mutation) {
  // These healing operations retain the underlying curves and surfaces while
  // changing boundary topology and tolerances. The symmetric vertex Hausdorff
  // distance therefore measures the only admitted movement and supplies its witness.
  if (sources.empty() || targets.empty()) {
    throw std::runtime_error("OCCT sewing produced an empty vertex inventory");
  }
  const auto measure = [&mutation](const std::vector<gp_Pnt>& originals,
                                   const std::vector<gp_Pnt>& proposed,
                                   const bool reverse_witness) {
    for (const gp_Pnt& original : originals) {
      double nearest = std::numeric_limits<double>::infinity();
      const gp_Pnt* nearest_point = nullptr;
      for (const gp_Pnt& candidate : proposed) {
        const double distance = original.Distance(candidate);
        if (distance < nearest) {
          nearest = distance;
          nearest_point = &candidate;
        }
      }
      if (nearest > mutation.maximum_displacement && nearest_point != nullptr) {
        mutation.maximum_displacement = nearest;
        const gp_Pnt& source = reverse_witness ? *nearest_point : original;
        const gp_Pnt& target = reverse_witness ? original : *nearest_point;
        mutation.displacement_original = {source.X(), source.Y(), source.Z()};
        mutation.displacement_proposed = {target.X(), target.Y(), target.Z()};
      }
    }
  };
  measure(sources, targets, false);
  measure(targets, sources, true);
}

} // namespace

ExactHealingMutation consolidate_exact_duplicates(const TopoDS_Shape& shape,
                                                  const OcctImportOptions& options,
                                                  std::uint64_t initial_identity_work) {
  ExactHealingMutation mutation;
  mutation.shape = shape;
  mutation.identity_work_bytes = initial_identity_work;
  if (shape.ShapeType() != TopAbs_COMPOUND) {
    return mutation;
  }

  std::vector<std::pair<ShapeDigest, TopoDS_Shape>> children;
  for (TopoDS_Iterator iterator(shape, Standard_False, Standard_True); iterator.More();
       iterator.Next()) {
    const TopoDS_Shape& child = iterator.Value();
    children.emplace_back(
        persistent_digest(child, options, mutation.identity_work_bytes), child);
  }
  std::sort(children.begin(), children.end(), [](const auto& left, const auto& right) {
    return left.first < right.first;
  });
  const auto unique_end = std::unique(
      children.begin(), children.end(), [](const auto& left, const auto& right) {
        return left.first == right.first;
      });
  if (unique_end == children.end()) {
    return mutation;
  }
  children.erase(unique_end, children.end());
  BRep_Builder builder;
  TopoDS_Compound consolidated;
  builder.MakeCompound(consolidated);
  for (const auto& child : children) {
    builder.Add(consolidated, child.second.Oriented(TopAbs_FORWARD));
  }
  mutation.shape = consolidated;
  mutation.changed = true;
  return mutation;
}

ExactHealingMutation repair_exact_orientation(const TopoDS_Shape& shape,
                                              const OcctImportOptions& options,
                                              std::uint64_t initial_identity_work) {
  ShapeFix_Shape fixer(shape);
  fixer.FixSolidMode() = 1;
  fixer.FixFreeShellMode() = 1;
  fixer.FixFreeFaceMode() = 0;
  fixer.FixFreeWireMode() = 0;
  fixer.FixSameParameterMode() = 0;
  fixer.FixVertexPositionMode() = 0;
  fixer.FixVertexTolMode() = 0;
  fixer.FixSolidTool()->FixShellMode() = 1;
  fixer.FixSolidTool()->FixShellOrientationMode() = 1;
  fixer.FixSolidTool()->FixShellTool()->FixFaceMode() = 0;
  fixer.FixSolidTool()->FixShellTool()->FixOrientationMode() = 1;

  HealingProgress progress(options);
  fixer.Perform(progress.Start());
  if (options.cancel_token_id != 0 && occt_import_cancelled(options.cancel_token_id)) {
    throw std::runtime_error("OCCT CAD import cancelled");
  }
  TopoDS_Shape result = fixer.Shape();
  if (result.IsNull()) {
    throw std::runtime_error("OCCT orientation repair produced a null shape");
  }
  ExactHealingMutation repair;
  repair.identity_work_bytes = initial_identity_work;
  if (!preserves_persistent_inventory(
          shape, result, options, repair.identity_work_bytes)) {
    throw std::runtime_error(
        "OCCT orientation repair attempted a non-orientation topology mutation");
  }
  repair.shape = result;
  repair.changed = true;
  return repair;
}

ExactHealingMutation sew_exact_shape(const TopoDS_Shape& shape,
                                    const OcctImportOptions& options,
                                    const double tolerance,
                                    const std::uint64_t initial_identity_work) {
  if (!std::isfinite(tolerance) || tolerance <= 0.0) {
    throw std::runtime_error("OCCT sewing requires a positive finite tolerance");
  }
  BRepBuilderAPI_Sewing sewing(tolerance, Standard_True, Standard_True, Standard_True,
                               Standard_False);
  sewing.SetMaxTolerance(tolerance);
  sewing.SetNonManifoldMode(Standard_False);
  sewing.Add(shape);
  HealingProgress progress(options);
  sewing.Perform(progress.Start());
  if (options.cancel_token_id != 0 && occt_import_cancelled(options.cancel_token_id)) {
    throw std::runtime_error("OCCT CAD import cancelled");
  }
  const TopoDS_Shape result = sewing.SewedShape();
  if (result.IsNull()) {
    throw std::runtime_error("OCCT sewing produced a null shape");
  }

  ExactHealingMutation mutation;
  mutation.shape = result;
  mutation.identity_work_bytes = initial_identity_work;
  append_relations(shape, result, TopAbs_VERTEX, 0, sewing, options, mutation);
  append_relations(shape, result, TopAbs_EDGE, 1, sewing, options, mutation);
  append_relations(shape, result, TopAbs_FACE, 2, sewing, options, mutation);
  measure_vertex_displacement(vertex_points(shape), vertex_points(result), mutation);
  mutation.changed = true;
  return mutation;
}

void append_small_topology_relations(
    const TopoDS_Shape& before,
    const TopoDS_Shape& after,
    const Handle(ShapeBuild_ReShape)& context,
    const OcctImportOptions& options,
    ExactHealingMutation& mutation) {
  append_context_relations(
      before, after, TopAbs_VERTEX, 0, context, options, mutation);
  append_context_relations(before, after, TopAbs_EDGE, 1, context, options, mutation);
  append_context_relations(before, after, TopAbs_FACE, 2, context, options, mutation);
}

void measure_healing_vertex_displacement(const TopoDS_Shape& before,
                                         const TopoDS_Shape& after,
                                         ExactHealingMutation& mutation) {
  measure_vertex_displacement(vertex_points(before), vertex_points(after), mutation);
}

} // namespace occt_backend
} // namespace runmat_geometry_io
