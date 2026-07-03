use crate::artifact::AnalysisMeshArtifact;

use super::AnalysisMeshValidationError;

pub(super) fn validate_no_fan_fallback(
    mesh: &AnalysisMeshArtifact,
    require_no_fan_fallback: bool,
) -> Result<(), AnalysisMeshValidationError> {
    if require_no_fan_fallback && mesh.backend.tetrahedron_fan_fallback_component_count > 0 {
        return Err(AnalysisMeshValidationError::FanFallbackRecoveryPresent {
            component_count: mesh.backend.tetrahedron_fan_fallback_component_count,
        });
    }
    Ok(())
}

pub(super) fn validate_no_unrepaired_exact_quality(
    mesh: &AnalysisMeshArtifact,
    require_no_unrepaired_exact_quality: bool,
) -> Result<(), AnalysisMeshValidationError> {
    if !require_no_unrepaired_exact_quality {
        return Ok(());
    }
    let boundary_adjacent_count = mesh
        .backend
        .tetrahedron_exact_quality_unrepaired_boundary_adjacent_count;
    let general_cavity_count = mesh
        .backend
        .tetrahedron_exact_quality_unrepaired_general_cavity_count;
    let interior_seed_count = mesh
        .backend
        .tetrahedron_exact_quality_unrepaired_interior_seed_count;
    let node_adjacent_count = mesh
        .backend
        .tetrahedron_exact_quality_unrepaired_node_adjacent_count;
    let edge_star_count = mesh
        .backend
        .tetrahedron_exact_quality_unrepaired_edge_star_count;
    let categorized_lower_bound = [
        boundary_adjacent_count,
        node_adjacent_count,
        interior_seed_count,
        edge_star_count,
        general_cavity_count,
    ]
    .into_iter()
    .max()
    .unwrap_or_default();
    let total_count = mesh
        .backend
        .tetrahedron_exact_quality_unrepaired_total_count
        .max(categorized_lower_bound);
    if total_count > 0 {
        return Err(AnalysisMeshValidationError::UnrepairedExactQualityPresent {
            total_count,
            general_cavity_count,
            boundary_adjacent_count,
            node_adjacent_count,
            interior_seed_count,
            edge_star_count,
        });
    }
    Ok(())
}
