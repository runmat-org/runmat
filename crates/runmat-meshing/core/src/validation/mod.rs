use std::collections::BTreeSet;

use crate::{
    artifact::{AnalysisMeshArtifact, ANALYSIS_MESH_SCHEMA_VERSION},
    quality::QualityThresholds,
};

mod connectivity;
use connectivity::sorted_edge;
pub use connectivity::{volume_component_count, volume_component_element_counts};

mod coverage;
use coverage::{
    validate_boundary_area_coverage, validate_bounds_coverage, validate_coverage_samples,
    validate_volume_coverage,
};

mod geometry;
pub use geometry::mesh_contains_point;

mod quality;
use quality::validate_quality;

mod recovery;
use recovery::{
    validate_boundary_edge_recovery, validate_boundary_face_recovery, validate_no_fan_fallback,
    validate_no_unrepaired_exact_quality,
};

mod regions;
use regions::{validate_required_boundary_regions, validate_required_material_regions};

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

#[cfg(test)]
mod tests;
