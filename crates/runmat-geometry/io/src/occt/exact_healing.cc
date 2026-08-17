#include "runmat-geometry-io/src/occt/ffi.rs.h"
#include "runmat-geometry-io/src/occt/exact_healing.hxx"

#include <BRep_Builder.hxx>
#include <BRepTools.hxx>
#include <Message_ProgressIndicator.hxx>
#include <ShapeFix_Shape.hxx>
#include <ShapeFix_Shell.hxx>
#include <ShapeFix_Solid.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Iterator.hxx>
#include <TopTools_IndexedMapOfShape.hxx>

#include <algorithm>
#include <array>
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

} // namespace occt_backend
} // namespace runmat_geometry_io
