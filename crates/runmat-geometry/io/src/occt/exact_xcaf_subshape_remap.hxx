#pragma once

#include "exact_healing.hxx"

#include <TDF_Label.hxx>
#include <XCAFDoc_ShapeTool.hxx>

namespace runmat_geometry_io {
namespace occt_backend {

// Rebinds existing XCAF subshape labels to their unique healed descendants.
// Labels whose topology is deleted, merged, split, or otherwise ambiguous are
// rejected rather than silently losing their semantic attributes.
void remap_exact_xcaf_subshapes(
    const Handle(XCAFDoc_ShapeTool)& shape_tool,
    const TDF_Label& definition_label,
    const TopoDS_Shape& healed_definition,
    const ExactHealingMutation& mutation,
    const OcctImportOptions& options,
    std::uint64_t& identity_work_bytes);

} // namespace occt_backend
} // namespace runmat_geometry_io
