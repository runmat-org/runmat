#pragma once

#include <TopoDS_Shape.hxx>

#include <cstdint>

namespace runmat_geometry_io {
namespace occt_backend {

struct OcctImportOptions;

struct ExactOrientationRepair {
  TopoDS_Shape shape;
  std::uint64_t identity_work_bytes = 0;
};

ExactOrientationRepair repair_exact_orientation(const TopoDS_Shape& shape,
                                                const OcctImportOptions& options);

} // namespace occt_backend
} // namespace runmat_geometry_io
