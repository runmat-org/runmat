#pragma once

#include <TopoDS_Shape.hxx>

#include <cstdint>

namespace runmat_geometry_io {
namespace occt_backend {

struct OcctImportOptions;

struct ExactHealingMutation {
  TopoDS_Shape shape;
  std::uint64_t identity_work_bytes = 0;
  bool changed = false;
};

ExactHealingMutation consolidate_exact_duplicates(const TopoDS_Shape& shape,
                                                  const OcctImportOptions& options,
                                                  std::uint64_t initial_identity_work);

ExactHealingMutation repair_exact_orientation(const TopoDS_Shape& shape,
                                              const OcctImportOptions& options,
                                              std::uint64_t initial_identity_work);

} // namespace occt_backend
} // namespace runmat_geometry_io
