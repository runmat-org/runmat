#pragma once

#include "exact_healing.hxx"

#include <XCAFDoc_ShapeTool.hxx>

namespace runmat_geometry_io {
namespace occt_backend {

struct ExactXcafHealingResult : ExactHealingMutation {
  bool duplicates_consolidated = false;
  bool post_duplicate_kernel_valid = true;
  bool sewn = false;
  bool gaps_repaired = false;
  bool post_sewing_kernel_valid = true;
  bool short_edges_simplified = false;
  bool sliver_faces_simplified = false;
  bool post_small_topology_kernel_valid = true;
  bool orientation_repaired = false;
};

ExactXcafHealingResult heal_exact_xcaf_definitions(
    const Handle(XCAFDoc_ShapeTool)& shape_tool,
    const OcctImportOptions& options,
    std::uint64_t initial_identity_work);

} // namespace occt_backend
} // namespace runmat_geometry_io
