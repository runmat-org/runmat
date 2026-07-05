use runmat_meshing_core::contracts::ProtectedBoundaryComplex;
use runmat_meshing_plc::validate::{
    classify_boundary_components, classify_shell_nesting, validate_protected_boundary_complex,
};

use super::TetrahedronGenerationError;

pub(super) fn validate_tetrahedron_generation_plc(
    plc: &ProtectedBoundaryComplex,
) -> Result<(), TetrahedronGenerationError> {
    validate_protected_boundary_complex(plc)
        .map(|_| ())
        .map_err(|error| TetrahedronGenerationError::InvalidProtectedBoundaryComplex { error })?;

    let component_report = classify_boundary_components(plc);
    let shell_classification = classify_shell_nesting(plc, &component_report);
    if shell_classification.outer_shell_count != 1
        || shell_classification.nested_shell_count != 0
        || shell_classification.max_nesting_depth != 0
    {
        return Err(TetrahedronGenerationError::UnsupportedNestedShellPlc {
            outer_shell_count: shell_classification.outer_shell_count,
            nested_shell_count: shell_classification.nested_shell_count,
            max_nesting_depth: shell_classification.max_nesting_depth,
        });
    }

    Ok(())
}
