use std::collections::BTreeSet;

use crate::{
    artifact::{AnalysisMeshArtifact, ANALYSIS_MESH_SCHEMA_VERSION},
    quality::QualityThresholds,
    topology::VolumeElementKind,
};

mod connectivity;
use connectivity::{boundary_face_edges, sorted_edge};
pub use connectivity::{volume_component_count, volume_component_element_counts};

mod geometry;
pub use geometry::mesh_contains_point;
use geometry::{
    element_tetrahedron_points, mesh_boundary_area_m2, mesh_bounds_m, mesh_volume_m3,
    tetrahedron_volume_m3,
};

mod types;
pub use types::{
    analysis_mesh_validation_error_code, AnalysisMeshValidationError, AnalysisMeshValidationOptions,
};

pub fn validate_analysis_mesh(
    mesh: &AnalysisMeshArtifact,
    thresholds: QualityThresholds,
) -> Result<(), AnalysisMeshValidationError> {
    validate_analysis_mesh_with_options(
        mesh,
        AnalysisMeshValidationOptions {
            quality: thresholds,
            ..AnalysisMeshValidationOptions::default()
        },
    )
}

pub fn validate_analysis_mesh_with_options(
    mesh: &AnalysisMeshArtifact,
    options: AnalysisMeshValidationOptions,
) -> Result<(), AnalysisMeshValidationError> {
    if mesh.schema_version != ANALYSIS_MESH_SCHEMA_VERSION {
        return Err(AnalysisMeshValidationError::UnsupportedSchema {
            schema_version: mesh.schema_version.clone(),
        });
    }
    if mesh.nodes.is_empty() {
        return Err(AnalysisMeshValidationError::EmptyNodes);
    }
    if mesh.volume_elements.is_empty() {
        return Err(AnalysisMeshValidationError::EmptyVolumeElements);
    }
    if let Some(max_element_count) = options.max_volume_element_count {
        if mesh.volume_elements.len() > max_element_count {
            return Err(AnalysisMeshValidationError::ElementBudgetExceeded {
                element_count: mesh.volume_elements.len(),
                max_element_count,
            });
        }
    }

    let mut node_ids = BTreeSet::<u32>::new();
    for node in &mesh.nodes {
        if !node_ids.insert(node.node_id) {
            return Err(AnalysisMeshValidationError::DuplicateNodeId {
                node_id: node.node_id,
            });
        }
        if node
            .coordinates_m
            .iter()
            .any(|coordinate| !coordinate.is_finite())
        {
            return Err(AnalysisMeshValidationError::NonFiniteNodeCoordinate {
                node_id: node.node_id,
            });
        }
    }

    let mut element_ids = BTreeSet::<String>::new();
    for element in &mesh.volume_elements {
        if !element_ids.insert(element.element_id.clone()) {
            return Err(AnalysisMeshValidationError::DuplicateElementId {
                element_id: element.element_id.clone(),
            });
        }
        if !element.kind.is_supported_for_solid_solve() {
            return Err(AnalysisMeshValidationError::UnsupportedVolumeElementKind {
                element_id: element.element_id.clone(),
            });
        }
        let expected = element.kind.node_count();
        if element.node_ids.len() != expected {
            return Err(AnalysisMeshValidationError::WrongVolumeElementNodeCount {
                element_id: element.element_id.clone(),
                expected,
                actual: element.node_ids.len(),
            });
        }
        let mut local_nodes = BTreeSet::<u32>::new();
        for node_id in &element.node_ids {
            if !node_ids.contains(node_id) {
                return Err(AnalysisMeshValidationError::UnknownVolumeElementNode {
                    element_id: element.element_id.clone(),
                    node_id: *node_id,
                });
            }
            if !local_nodes.insert(*node_id) {
                return Err(AnalysisMeshValidationError::RepeatedVolumeElementNode {
                    element_id: element.element_id.clone(),
                });
            }
        }
        if element.material_region_id.trim().is_empty() {
            return Err(AnalysisMeshValidationError::MissingMaterialRegion {
                element_id: element.element_id.clone(),
            });
        }
    }

    let mut face_ids = BTreeSet::<String>::new();
    for face in &mesh.boundary_faces {
        if !face_ids.insert(face.face_id.clone()) {
            return Err(AnalysisMeshValidationError::DuplicateBoundaryFaceId {
                face_id: face.face_id.clone(),
            });
        }
        if !face.kind.is_supported_for_boundary_mapping() {
            return Err(
                AnalysisMeshValidationError::UnsupportedBoundaryElementKind {
                    face_id: face.face_id.clone(),
                },
            );
        }
        let expected = face.kind.node_count();
        if face.node_ids.len() != expected {
            return Err(AnalysisMeshValidationError::WrongBoundaryFaceNodeCount {
                face_id: face.face_id.clone(),
                expected,
                actual: face.node_ids.len(),
            });
        }
        let mut local_nodes = BTreeSet::<u32>::new();
        for node_id in &face.node_ids {
            if !node_ids.contains(node_id) {
                return Err(AnalysisMeshValidationError::UnknownBoundaryFaceNode {
                    face_id: face.face_id.clone(),
                    node_id: *node_id,
                });
            }
            if !local_nodes.insert(*node_id) {
                return Err(AnalysisMeshValidationError::RepeatedBoundaryFaceNode {
                    face_id: face.face_id.clone(),
                });
            }
        }
        for element_id in &face.adjacent_volume_element_ids {
            if !element_ids.contains(element_id) {
                return Err(
                    AnalysisMeshValidationError::UnknownBoundaryAdjacentElement {
                        face_id: face.face_id.clone(),
                        element_id: element_id.clone(),
                    },
                );
            }
        }
    }

    let mut boundary_edge_ids = BTreeSet::<String>::new();
    let mut recovered_boundary_edges = BTreeSet::<[u32; 2]>::new();
    for edge in &mesh.boundary_edges {
        if !boundary_edge_ids.insert(edge.edge_id.clone()) {
            return Err(AnalysisMeshValidationError::DuplicateBoundaryEdgeId {
                edge_id: edge.edge_id.clone(),
            });
        }
        if edge.node_ids.len() != 2 {
            return Err(AnalysisMeshValidationError::WrongBoundaryEdgeNodeCount {
                edge_id: edge.edge_id.clone(),
                expected: 2,
                actual: edge.node_ids.len(),
            });
        }
        let mut local_nodes = BTreeSet::<u32>::new();
        for node_id in &edge.node_ids {
            if !node_ids.contains(node_id) {
                return Err(AnalysisMeshValidationError::UnknownBoundaryEdgeNode {
                    edge_id: edge.edge_id.clone(),
                    node_id: *node_id,
                });
            }
            if !local_nodes.insert(*node_id) {
                return Err(AnalysisMeshValidationError::RepeatedBoundaryEdgeNode {
                    edge_id: edge.edge_id.clone(),
                });
            }
        }
        for face_id in &edge.adjacent_boundary_face_ids {
            if !face_ids.contains(face_id) {
                return Err(
                    AnalysisMeshValidationError::UnknownBoundaryEdgeAdjacentFace {
                        edge_id: edge.edge_id.clone(),
                        face_id: face_id.clone(),
                    },
                );
            }
        }
        if !edge.adjacent_boundary_face_ids.is_empty() {
            recovered_boundary_edges.insert(sorted_edge(edge.node_ids[0], edge.node_ids[1]));
        }
    }

    validate_required_boundary_regions(mesh, &options.required_boundary_region_ids)?;
    validate_required_material_regions(mesh, &options.required_material_region_ids)?;
    validate_no_fan_fallback(mesh, options.require_no_fan_fallback)?;
    validate_no_unrepaired_exact_quality(mesh, options.require_no_unrepaired_exact_quality)?;
    validate_volume_component_count(mesh, options.max_volume_component_count)?;
    validate_coverage_samples(
        mesh,
        &options.coverage_sample_points_m,
        options.min_coverage_sample_ratio,
    )?;
    validate_bounds_coverage(
        mesh,
        options.expected_bounds_m,
        options.min_bounds_coverage_ratio,
    )?;
    validate_volume_coverage(
        mesh,
        options.expected_volume_m3,
        options.min_volume_coverage_ratio,
    )?;
    validate_boundary_area_coverage(
        mesh,
        options.expected_boundary_area_m2,
        options.min_boundary_area_ratio,
    )?;
    validate_boundary_face_recovery(mesh, options.min_boundary_face_recovery_ratio)?;
    validate_boundary_edge_recovery(
        mesh,
        &recovered_boundary_edges,
        options.min_boundary_edge_recovery_ratio,
    )?;
    validate_quality(mesh, options.quality)
}

fn validate_no_fan_fallback(
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

fn validate_no_unrepaired_exact_quality(
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

fn validate_volume_component_count(
    mesh: &AnalysisMeshArtifact,
    max_component_count: Option<usize>,
) -> Result<(), AnalysisMeshValidationError> {
    let Some(max_component_count) = max_component_count else {
        return Ok(());
    };
    let component_count = volume_component_count(mesh);
    if component_count > max_component_count {
        return Err(AnalysisMeshValidationError::VolumeComponentCountExceeded {
            component_count,
            max_component_count,
        });
    }
    Ok(())
}

fn validate_coverage_samples(
    mesh: &AnalysisMeshArtifact,
    coverage_sample_points_m: &[[f64; 3]],
    min_coverage_sample_ratio: f64,
) -> Result<(), AnalysisMeshValidationError> {
    if coverage_sample_points_m.is_empty()
        || !min_coverage_sample_ratio.is_finite()
        || min_coverage_sample_ratio <= 0.0
    {
        return Ok(());
    }
    let finite_samples = coverage_sample_points_m
        .iter()
        .copied()
        .filter(|point| point.iter().all(|value| value.is_finite()))
        .collect::<Vec<_>>();
    if finite_samples.is_empty() {
        return Ok(());
    }
    let covered_count = finite_samples
        .iter()
        .filter(|point| mesh_contains_point(mesh, **point))
        .count();
    let coverage_ratio = covered_count as f64 / finite_samples.len() as f64;
    if coverage_ratio + 1.0e-9 < min_coverage_sample_ratio {
        return Err(AnalysisMeshValidationError::CoverageSampleFailed {
            coverage_ratio: format!("{coverage_ratio:.6}"),
            required_ratio: format!("{min_coverage_sample_ratio:.6}"),
        });
    }
    Ok(())
}

fn validate_quality(
    mesh: &AnalysisMeshArtifact,
    thresholds: QualityThresholds,
) -> Result<(), AnalysisMeshValidationError> {
    if !mesh.quality.min_scaled_jacobian.is_finite()
        || mesh.quality.min_scaled_jacobian < thresholds.min_scaled_jacobian
    {
        return Err(AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "min_scaled_jacobian".to_string(),
        });
    }
    if !mesh.quality.min_exact_scaled_jacobian.is_finite()
        || mesh.quality.min_exact_scaled_jacobian < thresholds.min_scaled_jacobian
    {
        return Err(AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "min_exact_scaled_jacobian".to_string(),
        });
    }
    if mesh.quality.elements.iter().any(|element| {
        !element.exact_scaled_jacobian.is_finite()
            || element.exact_scaled_jacobian < thresholds.min_scaled_jacobian
    }) {
        return Err(AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "element_exact_scaled_jacobian".to_string(),
        });
    }
    if !mesh.quality.max_aspect_ratio.is_finite()
        || mesh.quality.max_aspect_ratio > thresholds.max_aspect_ratio
    {
        return Err(AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "max_aspect_ratio".to_string(),
        });
    }
    if !mesh.quality.max_boundary_projection_error_m.is_finite()
        || mesh.quality.max_boundary_projection_error_m > thresholds.max_boundary_projection_error_m
    {
        return Err(AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "max_boundary_projection_error_m".to_string(),
        });
    }
    if !thresholds.allow_inverted_elements && mesh.quality.inverted_element_count > 0 {
        return Err(AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "inverted_element_count".to_string(),
        });
    }
    Ok(())
}

fn validate_required_boundary_regions(
    mesh: &AnalysisMeshArtifact,
    required_region_ids: &[String],
) -> Result<(), AnalysisMeshValidationError> {
    if required_region_ids.is_empty() {
        return Ok(());
    }
    let present = mesh
        .boundary_faces
        .iter()
        .flat_map(|face| face.region_ids.iter().map(String::as_str))
        .collect::<BTreeSet<_>>();
    let recovered = mesh
        .boundary_faces
        .iter()
        .filter(|face| !face.adjacent_volume_element_ids.is_empty())
        .flat_map(|face| face.region_ids.iter().map(String::as_str))
        .collect::<BTreeSet<_>>();
    for region_id in required_region_ids {
        if !present.contains(region_id.as_str()) {
            return Err(AnalysisMeshValidationError::MissingRequiredBoundaryRegion {
                region_id: region_id.clone(),
            });
        }
        if !recovered.contains(region_id.as_str()) {
            return Err(
                AnalysisMeshValidationError::MissingRequiredBoundaryRegionRecovery {
                    region_id: region_id.clone(),
                },
            );
        }
    }
    Ok(())
}

fn validate_required_material_regions(
    mesh: &AnalysisMeshArtifact,
    required_region_ids: &[String],
) -> Result<(), AnalysisMeshValidationError> {
    if required_region_ids.is_empty() {
        return Ok(());
    }
    let present = mesh
        .volume_elements
        .iter()
        .map(|element| element.material_region_id.as_str())
        .collect::<BTreeSet<_>>();
    let positive_volume = mesh
        .volume_elements
        .iter()
        .filter(|element| {
            element.kind == VolumeElementKind::Tetrahedron4 && element.node_ids.len() == 4
        })
        .filter(|element| {
            let Some(points) = element_tetrahedron_points(mesh, element.node_ids.as_slice()) else {
                return false;
            };
            let volume_m3 = tetrahedron_volume_m3(points);
            volume_m3.is_finite() && volume_m3 > f64::EPSILON
        })
        .map(|element| element.material_region_id.as_str())
        .collect::<BTreeSet<_>>();
    for region_id in required_region_ids {
        if !present.contains(region_id.as_str()) {
            return Err(AnalysisMeshValidationError::MissingRequiredMaterialRegion {
                region_id: region_id.clone(),
            });
        }
        if !positive_volume.contains(region_id.as_str()) {
            return Err(
                AnalysisMeshValidationError::MissingRequiredMaterialRegionCoverage {
                    region_id: region_id.clone(),
                },
            );
        }
    }
    Ok(())
}

fn validate_bounds_coverage(
    mesh: &AnalysisMeshArtifact,
    expected_bounds_m: Option<[[f64; 3]; 2]>,
    min_bounds_coverage_ratio: f64,
) -> Result<(), AnalysisMeshValidationError> {
    let Some(expected) = expected_bounds_m else {
        return Ok(());
    };
    if !min_bounds_coverage_ratio.is_finite() || min_bounds_coverage_ratio <= 0.0 {
        return Ok(());
    }
    let Some(actual) = mesh_bounds_m(mesh) else {
        return Ok(());
    };
    for axis in 0..3 {
        let expected_min = expected[0][axis].min(expected[1][axis]);
        let expected_max = expected[0][axis].max(expected[1][axis]);
        if !expected_min.is_finite() || !expected_max.is_finite() {
            continue;
        }
        let expected_span = expected_max - expected_min;
        if expected_span <= f64::EPSILON {
            continue;
        }
        let actual_min = actual[0][axis].min(actual[1][axis]);
        let actual_max = actual[0][axis].max(actual[1][axis]);
        let overlap = (actual_max.min(expected_max) - actual_min.max(expected_min)).max(0.0);
        let coverage = overlap / expected_span;
        if coverage + 1.0e-9 < min_bounds_coverage_ratio {
            return Err(AnalysisMeshValidationError::BoundsCoverageFailed {
                axis,
                coverage_ratio: format!("{coverage:.6}"),
                required_ratio: format!("{min_bounds_coverage_ratio:.6}"),
            });
        }
    }
    Ok(())
}

fn validate_volume_coverage(
    mesh: &AnalysisMeshArtifact,
    expected_volume_m3: Option<f64>,
    min_volume_coverage_ratio: f64,
) -> Result<(), AnalysisMeshValidationError> {
    let Some(expected_volume_m3) = expected_volume_m3 else {
        return Ok(());
    };
    if !expected_volume_m3.is_finite()
        || expected_volume_m3 <= f64::EPSILON
        || !min_volume_coverage_ratio.is_finite()
        || min_volume_coverage_ratio <= 0.0
    {
        return Ok(());
    }
    let actual_volume_m3 = mesh_volume_m3(mesh);
    let coverage_ratio = actual_volume_m3 / expected_volume_m3;
    if coverage_ratio + 1.0e-9 < min_volume_coverage_ratio
        || coverage_ratio - 1.0e-9 > 1.0 / min_volume_coverage_ratio
    {
        return Err(AnalysisMeshValidationError::VolumeCoverageFailed {
            coverage_ratio: format!("{coverage_ratio:.6}"),
            required_ratio: format!("{min_volume_coverage_ratio:.6}"),
        });
    }
    Ok(())
}

fn validate_boundary_area_coverage(
    mesh: &AnalysisMeshArtifact,
    expected_boundary_area_m2: Option<f64>,
    min_boundary_area_ratio: f64,
) -> Result<(), AnalysisMeshValidationError> {
    let Some(expected_boundary_area_m2) = expected_boundary_area_m2 else {
        return Ok(());
    };
    if !expected_boundary_area_m2.is_finite()
        || expected_boundary_area_m2 <= f64::EPSILON
        || !min_boundary_area_ratio.is_finite()
        || min_boundary_area_ratio <= 0.0
    {
        return Ok(());
    }
    let actual_boundary_area_m2 = mesh_boundary_area_m2(mesh);
    let area_ratio = actual_boundary_area_m2 / expected_boundary_area_m2;
    if area_ratio + 1.0e-9 < min_boundary_area_ratio
        || area_ratio - 1.0e-9 > 1.0 / min_boundary_area_ratio
    {
        return Err(AnalysisMeshValidationError::BoundaryAreaCoverageFailed {
            area_ratio: format!("{area_ratio:.6}"),
            required_ratio: format!("{min_boundary_area_ratio:.6}"),
        });
    }
    Ok(())
}

fn validate_boundary_face_recovery(
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

fn validate_boundary_edge_recovery(
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

#[cfg(test)]
mod tests;
