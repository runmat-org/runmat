#pragma once

class TopoDS_Shell;
class TopoDS_Solid;

namespace runmat_geometry_io {
namespace occt_backend {

struct OcctImportOptions;

void validate_exact_solid_nesting(const TopoDS_Solid& solid,
                                  const TopoDS_Shell& outer_shell,
                                  const OcctImportOptions& options);

} // namespace occt_backend
} // namespace runmat_geometry_io
