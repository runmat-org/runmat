#pragma once

#include <TopoDS_Shape.hxx>

#include <array>
#include <cstdint>
#include <vector>

namespace runmat_geometry_io {
namespace occt_backend {

struct OcctImportOptions;

struct ExactHealingMutation {
  TopoDS_Shape shape;
  std::uint64_t identity_work_bytes = 0;
  bool changed = false;
  double maximum_displacement = 0.0;
  std::array<double, 3> displacement_original{0.0, 0.0, 0.0};
  std::array<double, 3> displacement_proposed{0.0, 0.0, 0.0};
  struct Relation {
    std::uint8_t kind = 0;
    std::array<std::uint8_t, 32> source_digest{};
    std::array<std::uint8_t, 32> target_digest{};
  };
  std::vector<Relation> relations;
};

ExactHealingMutation consolidate_exact_duplicates(const TopoDS_Shape& shape,
                                                  const OcctImportOptions& options,
                                                  std::uint64_t initial_identity_work);

ExactHealingMutation repair_exact_orientation(const TopoDS_Shape& shape,
                                              const OcctImportOptions& options,
                                              std::uint64_t initial_identity_work);

ExactHealingMutation sew_exact_shape(const TopoDS_Shape& shape,
                                    const OcctImportOptions& options,
                                    double tolerance,
                                    std::uint64_t initial_identity_work);

} // namespace occt_backend
} // namespace runmat_geometry_io
