#include "runmat-geometry-io/src/occt/ffi.rs.h"
#include "exact_nesting.hxx"

#include <BRepClass3d_SolidClassifier.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <TopAbs_State.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Shell.hxx>
#include <TopoDS_Solid.hxx>
#include <TopoDS_Vertex.hxx>
#include <gp_Pnt.hxx>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace runmat_geometry_io {
namespace occt_backend {
namespace {

void check_cancelled(const OcctImportOptions& options) {
  if (options.cancel_token_id != 0 && occt_import_cancelled(options.cancel_token_id)) {
    throw std::runtime_error("OCCT CAD import cancelled");
  }
}

TopoDS_Solid solid_from_shell(const TopoDS_Shell& shell) {
  BRep_Builder builder;
  TopoDS_Solid solid;
  builder.MakeSolid(solid);
  builder.Add(solid, shell);
  return solid;
}

TopAbs_State classify_infinite_point(const TopoDS_Shell& shell, double tolerance) {
  BRepClass3d_SolidClassifier classifier(solid_from_shell(shell));
  classifier.PerformInfinitePoint(tolerance);
  return classifier.State();
}

gp_Pnt shell_witness(const TopoDS_Shell& shell) {
  TopExp_Explorer vertices(shell, TopAbs_VERTEX);
  if (!vertices.More()) {
    throw std::runtime_error("OCCT exact solid shell has no nesting witness vertex");
  }
  return BRep_Tool::Pnt(TopoDS::Vertex(vertices.Current()));
}

double shell_tolerance(const TopoDS_Shell& shell) {
  double tolerance = 1.0e-12;
  for (TopExp_Explorer vertices(shell, TopAbs_VERTEX); vertices.More(); vertices.Next()) {
    tolerance = std::max(tolerance, BRep_Tool::Tolerance(TopoDS::Vertex(vertices.Current())));
  }
  return tolerance;
}

TopAbs_State classify_point(const TopoDS_Shell& shell,
                            const gp_Pnt& point,
                            double tolerance) {
  BRepClass3d_SolidClassifier classifier(solid_from_shell(shell));
  classifier.Perform(point, tolerance);
  return classifier.State();
}

} // namespace

void validate_exact_solid_nesting(const TopoDS_Solid& solid,
                                  const TopoDS_Shell& outer_shell,
                                  const OcctImportOptions& options) {
  check_cancelled(options);
  // Reclassify shell-only solids independently of the BRepClass3d::OuterShell selection used by
  // topology extraction. The infinite-point states prove shell role orientation; finite witness
  // classification proves containment. Vertex tolerances are the imported source tolerances and
  // are used only for geometric classification, never for a combinatorial topology decision.
  const double outer_tolerance = shell_tolerance(outer_shell);
  if (classify_infinite_point(outer_shell, outer_tolerance) != TopAbs_OUT) {
    throw std::runtime_error("OCCT exact outer shell does not bound finite material");
  }

  std::vector<TopoDS_Shell> void_shells;
  for (TopoDS_Iterator shells(solid, Standard_False, Standard_True); shells.More(); shells.Next()) {
    if (shells.Value().ShapeType() == TopAbs_SHELL && !shells.Value().IsSame(outer_shell)) {
      void_shells.push_back(TopoDS::Shell(shells.Value()));
    }
  }
  for (std::size_t index = 0; index < void_shells.size(); ++index) {
    check_cancelled(options);
    const TopoDS_Shell& cavity = void_shells[index];
    const double tolerance = std::max(outer_tolerance, shell_tolerance(cavity));
    if (classify_infinite_point(cavity, tolerance) != TopAbs_IN) {
      throw std::runtime_error("OCCT exact void shell does not bound a cavity");
    }
    const gp_Pnt witness = shell_witness(cavity);
    if (classify_point(outer_shell, witness, tolerance) != TopAbs_IN) {
      throw std::runtime_error("OCCT exact void shell is not nested inside its outer shell");
    }
    for (std::size_t other = 0; other < void_shells.size(); ++other) {
      if (other == index) {
        continue;
      }
      if (classify_point(void_shells[other], witness, tolerance) != TopAbs_OUT) {
        throw std::runtime_error("OCCT exact void shells overlap or nest ambiguously");
      }
    }
  }
}

} // namespace occt_backend
} // namespace runmat_geometry_io
