use std::collections::BTreeSet;

use crate::contracts::AnalysisMeshArtifact;

use super::{connectivity::boundary_face_edges, AnalysisMeshValidationError};

pub(super) fn validate_no_unrecovered_tetrahedron_components(
    mesh: &AnalysisMeshArtifact,
    require_no_unrecovered_tetrahedron_components: bool,
) -> Result<(), AnalysisMeshValidationError> {
    if require_no_unrecovered_tetrahedron_components
        && mesh.backend.tetrahedron_unrecovered_component_count > 0
    {
        return Err(
            AnalysisMeshValidationError::UnrecoveredTetrahedronComponentsPresent {
                component_count: mesh.backend.tetrahedron_unrecovered_component_count,
            },
        );
    }
    Ok(())
}

pub(super) fn validate_no_rolled_back_material_interface_partitions(
    mesh: &AnalysisMeshArtifact,
) -> Result<(), AnalysisMeshValidationError> {
    let recovery_item_count = mesh
        .backend
        .tetrahedron_rolled_back_absent_material_partition_recovery_item_count;
    let element_count = mesh
        .backend
        .tetrahedron_rolled_back_absent_material_partition_element_count;
    let boundary_face_count = mesh
        .backend
        .tetrahedron_rolled_back_absent_material_partition_boundary_face_count;
    let post_insertion_audit_rejection_count = mesh
        .backend
        .tetrahedron_rejected_absent_material_partition_post_insertion_audit_count;

    if recovery_item_count > 0
        || element_count > 0
        || boundary_face_count > 0
        || post_insertion_audit_rejection_count > 0
    {
        return Err(
            AnalysisMeshValidationError::RolledBackMaterialInterfacePartitionRecoveryPresent {
                recovery_item_count,
                element_count,
                boundary_face_count,
                post_insertion_audit_rejection_count,
            },
        );
    }
    Ok(())
}

pub(super) fn validate_tetrahedron_recovery_complete(
    mesh: &AnalysisMeshArtifact,
) -> Result<(), AnalysisMeshValidationError> {
    let source_face_item_count = mesh
        .backend
        .tetrahedron_missing_source_face_recovery_item_count;
    let source_edge_item_count = mesh
        .backend
        .tetrahedron_missing_source_edge_recovery_item_count;
    let material_interface_item_count = mesh
        .backend
        .tetrahedron_missing_material_interface_recovery_item_count;
    let total_item_count = mesh
        .backend
        .tetrahedron_missing_recovery_item_count
        .max(source_face_item_count + source_edge_item_count + material_interface_item_count);

    if total_item_count > 0 {
        return Err(
            AnalysisMeshValidationError::IncompleteTetrahedronRecoveryPresent {
                missing_item_count: total_item_count,
                missing_source_face_item_count: source_face_item_count,
                missing_source_edge_item_count: source_edge_item_count,
                missing_material_interface_item_count: material_interface_item_count,
            },
        );
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

pub(super) fn validate_boundary_face_recovery(
    mesh: &AnalysisMeshArtifact,
    min_boundary_face_recovery_ratio: f64,
) -> Result<(), AnalysisMeshValidationError> {
    if mesh.boundary_faces.is_empty()
        || !min_boundary_face_recovery_ratio.is_finite()
        || min_boundary_face_recovery_ratio <= 0.0
    {
        return Ok(());
    }
    let recovered_count = mesh
        .boundary_faces
        .iter()
        .filter(|face| !face.adjacent_volume_element_ids.is_empty())
        .count();
    let recovery_ratio = recovered_count as f64 / mesh.boundary_faces.len() as f64;
    if recovery_ratio + 1.0e-9 < min_boundary_face_recovery_ratio {
        return Err(AnalysisMeshValidationError::BoundaryFaceRecoveryFailed {
            recovery_ratio: format!("{recovery_ratio:.6}"),
            required_ratio: format!("{min_boundary_face_recovery_ratio:.6}"),
        });
    }
    Ok(())
}

pub(super) fn validate_boundary_edge_recovery(
    mesh: &AnalysisMeshArtifact,
    recovered_boundary_edges: &BTreeSet<[u32; 2]>,
    min_boundary_edge_recovery_ratio: f64,
) -> Result<(), AnalysisMeshValidationError> {
    if mesh.boundary_faces.is_empty()
        || !min_boundary_edge_recovery_ratio.is_finite()
        || min_boundary_edge_recovery_ratio <= 0.0
    {
        return Ok(());
    }
    let expected_edges = boundary_face_edges(mesh);
    if expected_edges.is_empty() {
        return Ok(());
    }
    let recovered_count = expected_edges
        .iter()
        .filter(|edge| recovered_boundary_edges.contains(*edge))
        .count();
    let recovery_ratio = recovered_count as f64 / expected_edges.len() as f64;
    if recovery_ratio + 1.0e-9 < min_boundary_edge_recovery_ratio {
        return Err(AnalysisMeshValidationError::BoundaryEdgeRecoveryFailed {
            recovery_ratio: format!("{recovery_ratio:.6}"),
            required_ratio: format!("{min_boundary_edge_recovery_ratio:.6}"),
        });
    }
    Ok(())
}
