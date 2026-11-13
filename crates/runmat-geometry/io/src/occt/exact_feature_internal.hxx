#pragma once

#include "runmat-geometry-io/src/occt/ffi.rs.h"

#include <TopoDS_Shape.hxx>

#include <array>
#include <cstdint>
#include <vector>

namespace runmat_geometry_io {
namespace occt_backend {

struct FeatureHistoryRelation {
  OcctFeatureRelationKind kind;
  TopoDS_Shape target;
};

struct FeatureHistorySource {
  std::uint64_t operand_index;
  OcctFeatureEntityKind kind;
  TopoDS_Shape source;
  std::array<std::uint8_t, 32> source_digest;
  bool deleted = false;
  std::vector<FeatureHistoryRelation> relations;
};

struct FeatureExecution {
  TopoDS_Shape shape;
  std::vector<TopoDS_Shape> operands;
  std::vector<FeatureHistorySource> history;
  std::uint64_t identity_work_bytes = 0;
};

void check_feature_cancelled(const OcctImportOptions& options);

FeatureExecution execute_feature_operation(
    OcctFeatureKind kind,
    rust::Slice<const double> scalars,
    rust::Slice<const double> matrices,
    const rust::Vec<OcctFeatureSelection>& selections,
    const rust::Vec<OcctFeatureOperand>& operands,
    const OcctImportOptions& options);

}  // namespace occt_backend
}  // namespace runmat_geometry_io
