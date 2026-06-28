use std::collections::{BTreeMap, BTreeSet};

use crate::{
    artifact::{AnalysisMeshArtifact, ANALYSIS_MESH_SCHEMA_VERSION},
    quality::QualityThresholds,
    topology::{BoundaryElementKind, VolumeElementKind},
};

#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisMeshValidationOptions {
    pub quality: QualityThresholds,
    pub max_volume_element_count: Option<usize>,
    pub max_volume_component_count: Option<usize>,
    pub coverage_sample_points_m: Vec<[f64; 3]>,
    pub min_coverage_sample_ratio: f64,
    pub expected_bounds_m: Option<[[f64; 3]; 2]>,
    pub min_bounds_coverage_ratio: f64,
    pub expected_volume_m3: Option<f64>,
    pub min_volume_coverage_ratio: f64,
    pub expected_boundary_area_m2: Option<f64>,
    pub min_boundary_area_ratio: f64,
    pub min_boundary_face_recovery_ratio: f64,
    pub min_boundary_edge_recovery_ratio: f64,
    pub require_no_fan_fallback: bool,
    pub require_no_unrepaired_exact_quality: bool,
    pub required_boundary_region_ids: Vec<String>,
    pub required_material_region_ids: Vec<String>,
}

impl Default for AnalysisMeshValidationOptions {
    fn default() -> Self {
        Self {
            quality: QualityThresholds::default(),
            max_volume_element_count: None,
            max_volume_component_count: None,
            coverage_sample_points_m: Vec::new(),
            min_coverage_sample_ratio: 1.0,
            expected_bounds_m: None,
            min_bounds_coverage_ratio: 0.90,
            expected_volume_m3: None,
            min_volume_coverage_ratio: 0.90,
            expected_boundary_area_m2: None,
            min_boundary_area_ratio: 0.90,
            min_boundary_face_recovery_ratio: 0.0,
            min_boundary_edge_recovery_ratio: 0.0,
            require_no_fan_fallback: false,
            require_no_unrepaired_exact_quality: false,
            required_boundary_region_ids: Vec::new(),
            required_material_region_ids: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnalysisMeshValidationError {
    UnsupportedSchema {
        schema_version: String,
    },
    EmptyNodes,
    EmptyVolumeElements,
    DuplicateNodeId {
        node_id: u32,
    },
    NonFiniteNodeCoordinate {
        node_id: u32,
    },
    DuplicateElementId {
        element_id: String,
    },
    UnsupportedVolumeElementKind {
        element_id: String,
    },
    WrongVolumeElementNodeCount {
        element_id: String,
        expected: usize,
        actual: usize,
    },
    UnknownVolumeElementNode {
        element_id: String,
        node_id: u32,
    },
    RepeatedVolumeElementNode {
        element_id: String,
    },
    MissingMaterialRegion {
        element_id: String,
    },
    DuplicateBoundaryFaceId {
        face_id: String,
    },
    UnsupportedBoundaryElementKind {
        face_id: String,
    },
    WrongBoundaryFaceNodeCount {
        face_id: String,
        expected: usize,
        actual: usize,
    },
    UnknownBoundaryFaceNode {
        face_id: String,
        node_id: u32,
    },
    RepeatedBoundaryFaceNode {
        face_id: String,
    },
    UnknownBoundaryAdjacentElement {
        face_id: String,
        element_id: String,
    },
    DuplicateBoundaryEdgeId {
        edge_id: String,
    },
    WrongBoundaryEdgeNodeCount {
        edge_id: String,
        expected: usize,
        actual: usize,
    },
    UnknownBoundaryEdgeNode {
        edge_id: String,
        node_id: u32,
    },
    RepeatedBoundaryEdgeNode {
        edge_id: String,
    },
    UnknownBoundaryEdgeAdjacentFace {
        edge_id: String,
        face_id: String,
    },
    QualityThresholdFailed {
        reason: String,
    },
    ElementBudgetExceeded {
        element_count: usize,
        max_element_count: usize,
    },
    VolumeComponentCountExceeded {
        component_count: usize,
        max_component_count: usize,
    },
    CoverageSampleFailed {
        coverage_ratio: String,
        required_ratio: String,
    },
    BoundsCoverageFailed {
        axis: usize,
        coverage_ratio: String,
        required_ratio: String,
    },
    VolumeCoverageFailed {
        coverage_ratio: String,
        required_ratio: String,
    },
    BoundaryAreaCoverageFailed {
        area_ratio: String,
        required_ratio: String,
    },
    BoundaryFaceRecoveryFailed {
        recovery_ratio: String,
        required_ratio: String,
    },
    BoundaryEdgeRecoveryFailed {
        recovery_ratio: String,
        required_ratio: String,
    },
    FanFallbackRecoveryPresent {
        component_count: usize,
    },
    UnrepairedExactQualityPresent {
        total_count: usize,
        general_cavity_count: usize,
        boundary_adjacent_count: usize,
        interior_seed_count: usize,
        edge_star_count: usize,
    },
    MissingRequiredBoundaryRegion {
        region_id: String,
    },
    MissingRequiredBoundaryRegionRecovery {
        region_id: String,
    },
    MissingRequiredMaterialRegion {
        region_id: String,
    },
    MissingRequiredMaterialRegionCoverage {
        region_id: String,
    },
}

pub fn analysis_mesh_validation_error_code(error: &AnalysisMeshValidationError) -> &'static str {
    match error {
        AnalysisMeshValidationError::UnsupportedSchema { .. } => "unsupported_schema",
        AnalysisMeshValidationError::EmptyNodes => "empty_nodes",
        AnalysisMeshValidationError::EmptyVolumeElements => "empty_volume_elements",
        AnalysisMeshValidationError::DuplicateNodeId { .. } => "duplicate_node_id",
        AnalysisMeshValidationError::NonFiniteNodeCoordinate { .. } => "non_finite_node_coordinate",
        AnalysisMeshValidationError::DuplicateElementId { .. } => "duplicate_element_id",
        AnalysisMeshValidationError::UnsupportedVolumeElementKind { .. } => {
            "unsupported_volume_element_kind"
        }
        AnalysisMeshValidationError::WrongVolumeElementNodeCount { .. } => {
            "wrong_volume_element_node_count"
        }
        AnalysisMeshValidationError::UnknownVolumeElementNode { .. } => {
            "unknown_volume_element_node"
        }
        AnalysisMeshValidationError::RepeatedVolumeElementNode { .. } => {
            "repeated_volume_element_node"
        }
        AnalysisMeshValidationError::MissingMaterialRegion { .. } => "missing_material_region",
        AnalysisMeshValidationError::DuplicateBoundaryFaceId { .. } => "duplicate_boundary_face_id",
        AnalysisMeshValidationError::UnsupportedBoundaryElementKind { .. } => {
            "unsupported_boundary_element_kind"
        }
        AnalysisMeshValidationError::WrongBoundaryFaceNodeCount { .. } => {
            "wrong_boundary_face_node_count"
        }
        AnalysisMeshValidationError::UnknownBoundaryFaceNode { .. } => "unknown_boundary_face_node",
        AnalysisMeshValidationError::RepeatedBoundaryFaceNode { .. } => {
            "repeated_boundary_face_node"
        }
        AnalysisMeshValidationError::UnknownBoundaryAdjacentElement { .. } => {
            "unknown_boundary_adjacent_element"
        }
        AnalysisMeshValidationError::DuplicateBoundaryEdgeId { .. } => "duplicate_boundary_edge_id",
        AnalysisMeshValidationError::WrongBoundaryEdgeNodeCount { .. } => {
            "wrong_boundary_edge_node_count"
        }
        AnalysisMeshValidationError::UnknownBoundaryEdgeNode { .. } => "unknown_boundary_edge_node",
        AnalysisMeshValidationError::RepeatedBoundaryEdgeNode { .. } => {
            "repeated_boundary_edge_node"
        }
        AnalysisMeshValidationError::UnknownBoundaryEdgeAdjacentFace { .. } => {
            "unknown_boundary_edge_adjacent_face"
        }
        AnalysisMeshValidationError::QualityThresholdFailed { .. } => "quality_threshold_failed",
        AnalysisMeshValidationError::ElementBudgetExceeded { .. } => "element_budget_exceeded",
        AnalysisMeshValidationError::VolumeComponentCountExceeded { .. } => {
            "volume_component_count_exceeded"
        }
        AnalysisMeshValidationError::CoverageSampleFailed { .. } => "coverage_sample_failed",
        AnalysisMeshValidationError::BoundsCoverageFailed { .. } => "bounds_coverage_failed",
        AnalysisMeshValidationError::VolumeCoverageFailed { .. } => "volume_coverage_failed",
        AnalysisMeshValidationError::BoundaryAreaCoverageFailed { .. } => {
            "boundary_area_coverage_failed"
        }
        AnalysisMeshValidationError::BoundaryFaceRecoveryFailed { .. } => {
            "boundary_face_recovery_failed"
        }
        AnalysisMeshValidationError::BoundaryEdgeRecoveryFailed { .. } => {
            "boundary_edge_recovery_failed"
        }
        AnalysisMeshValidationError::FanFallbackRecoveryPresent { .. } => {
            "fan_fallback_recovery_present"
        }
        AnalysisMeshValidationError::UnrepairedExactQualityPresent { .. } => {
            "unrepaired_exact_quality_present"
        }
        AnalysisMeshValidationError::MissingRequiredBoundaryRegion { .. } => {
            "missing_required_boundary_region"
        }
        AnalysisMeshValidationError::MissingRequiredBoundaryRegionRecovery { .. } => {
            "missing_required_boundary_region_recovery"
        }
        AnalysisMeshValidationError::MissingRequiredMaterialRegion { .. } => {
            "missing_required_material_region"
        }
        AnalysisMeshValidationError::MissingRequiredMaterialRegionCoverage { .. } => {
            "missing_required_material_region_coverage"
        }
    }
}

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
    if require_no_fan_fallback && mesh.backend.tet_fan_fallback_component_count > 0 {
        return Err(AnalysisMeshValidationError::FanFallbackRecoveryPresent {
            component_count: mesh.backend.tet_fan_fallback_component_count,
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
        .tet_exact_quality_unrepaired_boundary_adjacent_count;
    let general_cavity_count = mesh
        .backend
        .tet_exact_quality_unrepaired_general_cavity_count;
    let interior_seed_count = mesh
        .backend
        .tet_exact_quality_unrepaired_interior_seed_count;
    let edge_star_count = mesh.backend.tet_exact_quality_unrepaired_edge_star_count;
    let categorized_count = boundary_adjacent_count
        .saturating_add(interior_seed_count)
        .saturating_add(edge_star_count)
        .saturating_add(general_cavity_count);
    let total_count = mesh
        .backend
        .tet_exact_quality_unrepaired_total_count
        .max(categorized_count);
    if total_count > 0 {
        return Err(AnalysisMeshValidationError::UnrepairedExactQualityPresent {
            total_count,
            general_cavity_count,
            boundary_adjacent_count,
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
        .filter(|element| element.kind == VolumeElementKind::Tet4 && element.node_ids.len() == 4)
        .filter(|element| {
            let Some(points) = element_tet_points(mesh, element.node_ids.as_slice()) else {
                return false;
            };
            let volume_m3 = tet_volume_m3(points);
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

fn mesh_bounds_m(mesh: &AnalysisMeshArtifact) -> Option<[[f64; 3]; 2]> {
    let mut nodes = mesh.nodes.iter();
    let first = nodes.next()?.coordinates_m;
    let mut min = first;
    let mut max = first;
    for node in nodes {
        for axis in 0..3 {
            min[axis] = min[axis].min(node.coordinates_m[axis]);
            max[axis] = max[axis].max(node.coordinates_m[axis]);
        }
    }
    Some([min, max])
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

fn boundary_face_edges(mesh: &AnalysisMeshArtifact) -> BTreeSet<[u32; 2]> {
    let mut edges = BTreeSet::<[u32; 2]>::new();
    for face in &mesh.boundary_faces {
        if face.kind != BoundaryElementKind::Tri3 || face.node_ids.len() != 3 {
            continue;
        }
        edges.insert(sorted_edge(face.node_ids[0], face.node_ids[1]));
        edges.insert(sorted_edge(face.node_ids[1], face.node_ids[2]));
        edges.insert(sorted_edge(face.node_ids[2], face.node_ids[0]));
    }
    edges
}

pub fn volume_component_count(mesh: &AnalysisMeshArtifact) -> usize {
    volume_component_element_counts(mesh).len()
}

pub fn volume_component_element_counts(mesh: &AnalysisMeshArtifact) -> Vec<usize> {
    if mesh.volume_elements.is_empty() {
        return Vec::new();
    }
    let mut face_to_elements = BTreeMap::<[u32; 3], Vec<usize>>::new();
    for (element_index, element) in mesh.volume_elements.iter().enumerate() {
        if element.kind != VolumeElementKind::Tet4 || element.node_ids.len() != 4 {
            continue;
        }
        for face in tet_element_faces(element.node_ids.as_slice()) {
            face_to_elements
                .entry(face)
                .or_default()
                .push(element_index);
        }
    }
    let mut adjacency = vec![Vec::<usize>::new(); mesh.volume_elements.len()];
    for element_indices in face_to_elements.values() {
        for left_position in 0..element_indices.len() {
            for right_position in (left_position + 1)..element_indices.len() {
                let left = element_indices[left_position];
                let right = element_indices[right_position];
                adjacency[left].push(right);
                adjacency[right].push(left);
            }
        }
    }

    let mut visited = vec![false; mesh.volume_elements.len()];
    let mut component_element_counts = Vec::<usize>::new();
    for start in 0..mesh.volume_elements.len() {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut component_element_count = 0_usize;
        let mut stack = vec![start];
        while let Some(current) = stack.pop() {
            component_element_count += 1;
            for neighbor in &adjacency[current] {
                if visited[*neighbor] {
                    continue;
                }
                visited[*neighbor] = true;
                stack.push(*neighbor);
            }
        }
        component_element_counts.push(component_element_count);
    }
    component_element_counts
}

fn tet_element_faces(node_ids: &[u32]) -> [[u32; 3]; 4] {
    [
        sorted_node_face([node_ids[0], node_ids[1], node_ids[2]]),
        sorted_node_face([node_ids[0], node_ids[1], node_ids[3]]),
        sorted_node_face([node_ids[0], node_ids[2], node_ids[3]]),
        sorted_node_face([node_ids[1], node_ids[2], node_ids[3]]),
    ]
}

fn mesh_volume_m3(mesh: &AnalysisMeshArtifact) -> f64 {
    mesh.volume_elements
        .iter()
        .filter(|element| element.kind == VolumeElementKind::Tet4 && element.node_ids.len() == 4)
        .filter_map(|element| {
            Some(tet_volume_m3(element_tet_points(
                mesh,
                element.node_ids.as_slice(),
            )?))
        })
        .sum()
}

fn element_tet_points(mesh: &AnalysisMeshArtifact, node_ids: &[u32]) -> Option<[[f64; 3]; 4]> {
    Some([
        mesh_node(mesh, node_ids[0])?,
        mesh_node(mesh, node_ids[1])?,
        mesh_node(mesh, node_ids[2])?,
        mesh_node(mesh, node_ids[3])?,
    ])
}

fn mesh_boundary_area_m2(mesh: &AnalysisMeshArtifact) -> f64 {
    mesh.boundary_faces
        .iter()
        .filter(|face| face.kind == BoundaryElementKind::Tri3 && face.node_ids.len() == 3)
        .filter_map(|face| {
            Some(triangle_area_m2([
                mesh_node(mesh, face.node_ids[0])?,
                mesh_node(mesh, face.node_ids[1])?,
                mesh_node(mesh, face.node_ids[2])?,
            ]))
        })
        .sum()
}

fn mesh_node(mesh: &AnalysisMeshArtifact, node_id: u32) -> Option<[f64; 3]> {
    mesh.nodes
        .iter()
        .find(|node| node.node_id == node_id)
        .map(|node| node.coordinates_m)
}

pub fn mesh_contains_point(mesh: &AnalysisMeshArtifact, point: [f64; 3]) -> bool {
    mesh.volume_elements
        .iter()
        .filter(|element| element.kind == VolumeElementKind::Tet4 && element.node_ids.len() == 4)
        .filter_map(|element| {
            Some([
                mesh_node(mesh, element.node_ids[0])?,
                mesh_node(mesh, element.node_ids[1])?,
                mesh_node(mesh, element.node_ids[2])?,
                mesh_node(mesh, element.node_ids[3])?,
            ])
        })
        .any(|tet| point_in_tet(point, tet))
}

fn point_in_tet(point: [f64; 3], tet: [[f64; 3]; 4]) -> bool {
    let total = tet_volume_m3(tet);
    if !total.is_finite() || total <= f64::EPSILON {
        return false;
    }
    let subvolume_sum = tet_volume_m3([point, tet[1], tet[2], tet[3]])
        + tet_volume_m3([tet[0], point, tet[2], tet[3]])
        + tet_volume_m3([tet[0], tet[1], point, tet[3]])
        + tet_volume_m3([tet[0], tet[1], tet[2], point]);
    let tolerance = total * 1.0e-8 + f64::EPSILON;
    (subvolume_sum - total).abs() <= tolerance
}

fn tet_volume_m3(points: [[f64; 3]; 4]) -> f64 {
    dot(
        sub(points[1], points[0]),
        cross(sub(points[2], points[0]), sub(points[3], points[0])),
    )
    .abs()
        / 6.0
}

fn triangle_area_m2(points: [[f64; 3]; 3]) -> f64 {
    0.5 * norm(cross(sub(points[1], points[0]), sub(points[2], points[0])))
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn norm(value: [f64; 3]) -> f64 {
    dot(value, value).sqrt()
}

fn sorted_edge(left: u32, right: u32) -> [u32; 2] {
    if left <= right {
        [left, right]
    } else {
        [right, left]
    }
}

fn sorted_node_face(mut node_ids: [u32; 3]) -> [u32; 3] {
    node_ids.sort_unstable();
    node_ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artifact::{
            AnalysisBoundaryEdge, AnalysisBoundaryFace, AnalysisMeshArtifact, AnalysisMeshNode,
            AnalysisVolumeElement, ANALYSIS_MESH_SCHEMA_VERSION,
        },
        provenance::AnalysisMeshProvenance,
        quality::{AnalysisMeshQualityReport, ElementQuality},
        sizing::MeshSizingField,
        topology::{BoundaryElementKind, VolumeElementKind},
    };

    fn valid_tet_mesh() -> AnalysisMeshArtifact {
        AnalysisMeshArtifact {
            schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
            mesh_id: "mesh_valid".to_string(),
            nodes: vec![
                AnalysisMeshNode {
                    node_id: 1,
                    coordinates_m: [0.0, 0.0, 0.0],
                    provenance: Vec::new(),
                },
                AnalysisMeshNode {
                    node_id: 2,
                    coordinates_m: [1.0, 0.0, 0.0],
                    provenance: Vec::new(),
                },
                AnalysisMeshNode {
                    node_id: 3,
                    coordinates_m: [0.0, 1.0, 0.0],
                    provenance: Vec::new(),
                },
                AnalysisMeshNode {
                    node_id: 4,
                    coordinates_m: [0.0, 0.0, 1.0],
                    provenance: Vec::new(),
                },
            ],
            volume_elements: vec![AnalysisVolumeElement {
                element_id: "e1".to_string(),
                kind: VolumeElementKind::Tet4,
                node_ids: vec![1, 2, 3, 4],
                material_region_id: "mat_region".to_string(),
                provenance: Vec::new(),
            }],
            boundary_faces: vec![AnalysisBoundaryFace {
                face_id: "f1".to_string(),
                kind: BoundaryElementKind::Tri3,
                node_ids: vec![1, 2, 3],
                adjacent_volume_element_ids: vec!["e1".to_string()],
                region_ids: vec!["fixed".to_string()],
                provenance: Vec::new(),
            }],
            boundary_edges: Vec::new(),
            quality: AnalysisMeshQualityReport::default(),
            sizing: MeshSizingField::default(),
            backend: Default::default(),
            adaptive_iterations: Vec::new(),
            provenance: AnalysisMeshProvenance {
                algorithm: "test".to_string(),
                source_geometry_id: "geo".to_string(),
                source_geometry_revision: 1,
                source_geometry_sha256: None,
            },
        }
    }

    #[test]
    fn accepts_minimal_valid_tet4_mesh() {
        let mesh = valid_tet_mesh();
        validate_analysis_mesh(&mesh, QualityThresholds::default()).expect("mesh should validate");
    }

    #[test]
    fn rejects_empty_volume_elements() {
        let mut mesh = valid_tet_mesh();
        mesh.volume_elements.clear();
        let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
            .expect_err("empty volume elements should fail");
        assert_eq!(err, AnalysisMeshValidationError::EmptyVolumeElements);
    }

    #[test]
    fn rejects_mesh_that_exceeds_element_budget() {
        let mesh = valid_tet_mesh();
        let err = validate_analysis_mesh_with_options(
            &mesh,
            AnalysisMeshValidationOptions {
                max_volume_element_count: Some(0),
                ..AnalysisMeshValidationOptions::default()
            },
        )
        .expect_err("element budget overrun should fail");
        assert_eq!(
            err,
            AnalysisMeshValidationError::ElementBudgetExceeded {
                element_count: 1,
                max_element_count: 0,
            }
        );
    }

    #[test]
    fn rejects_fan_fallback_recovery_when_policy_requires_strict_recovery() {
        let mut mesh = valid_tet_mesh();
        mesh.backend.tet_fan_fallback_component_count = 1;

        let err = validate_analysis_mesh_with_options(
            &mesh,
            AnalysisMeshValidationOptions {
                require_no_fan_fallback: true,
                ..AnalysisMeshValidationOptions::default()
            },
        )
        .expect_err("strict recovery policy should reject fan fallback evidence");

        assert_eq!(
            err,
            AnalysisMeshValidationError::FanFallbackRecoveryPresent { component_count: 1 }
        );
        assert_eq!(
            analysis_mesh_validation_error_code(&err),
            "fan_fallback_recovery_present"
        );
    }

    #[test]
    fn rejects_unrepaired_exact_quality_when_policy_requires_strict_recovery() {
        let mut mesh = valid_tet_mesh();
        mesh.backend
            .tet_exact_quality_unrepaired_boundary_adjacent_count = 2;
        mesh.backend
            .tet_exact_quality_unrepaired_interior_seed_count = 3;
        mesh.backend.tet_exact_quality_unrepaired_edge_star_count = 5;

        let err = validate_analysis_mesh_with_options(
            &mesh,
            AnalysisMeshValidationOptions {
                require_no_unrepaired_exact_quality: true,
                ..AnalysisMeshValidationOptions::default()
            },
        )
        .expect_err("strict recovery policy should reject unrepaired exact-quality evidence");

        assert_eq!(
            err,
            AnalysisMeshValidationError::UnrepairedExactQualityPresent {
                total_count: 10,
                general_cavity_count: 0,
                boundary_adjacent_count: 2,
                interior_seed_count: 3,
                edge_star_count: 5,
            }
        );
        assert_eq!(
            analysis_mesh_validation_error_code(&err),
            "unrepaired_exact_quality_present"
        );
    }

    #[test]
    fn rejects_unrepaired_general_cavity_exact_quality_when_policy_requires_strict_recovery() {
        let mut mesh = valid_tet_mesh();
        mesh.backend.tet_exact_quality_unrepaired_total_count = 1;
        mesh.backend
            .tet_exact_quality_unrepaired_general_cavity_count = 1;

        let err = validate_analysis_mesh_with_options(
            &mesh,
            AnalysisMeshValidationOptions {
                require_no_unrepaired_exact_quality: true,
                ..AnalysisMeshValidationOptions::default()
            },
        )
        .expect_err("strict recovery policy should reject unclassified cavity evidence");

        assert_eq!(
            err,
            AnalysisMeshValidationError::UnrepairedExactQualityPresent {
                total_count: 1,
                general_cavity_count: 1,
                boundary_adjacent_count: 0,
                interior_seed_count: 0,
                edge_star_count: 0,
            }
        );
    }

    #[test]
    fn accepts_face_connected_volume_components_within_budget() {
        let mut mesh = valid_tet_mesh();
        mesh.nodes.push(AnalysisMeshNode {
            node_id: 5,
            coordinates_m: [0.0, 0.0, -1.0],
            provenance: Vec::new(),
        });
        mesh.volume_elements.push(AnalysisVolumeElement {
            element_id: "e2".to_string(),
            kind: VolumeElementKind::Tet4,
            node_ids: vec![1, 3, 2, 5],
            material_region_id: "mat_region".to_string(),
            provenance: Vec::new(),
        });

        assert_eq!(volume_component_count(&mesh), 1);
        validate_analysis_mesh_with_options(
            &mesh,
            AnalysisMeshValidationOptions {
                max_volume_component_count: Some(1),
                ..AnalysisMeshValidationOptions::default()
            },
        )
        .expect("face-connected tets should remain one volume component");
    }

    #[test]
    fn rejects_unintended_isolated_volume_components() {
        let mut mesh = valid_tet_mesh();
        mesh.nodes.extend([
            AnalysisMeshNode {
                node_id: 5,
                coordinates_m: [10.0, 0.0, 0.0],
                provenance: Vec::new(),
            },
            AnalysisMeshNode {
                node_id: 6,
                coordinates_m: [11.0, 0.0, 0.0],
                provenance: Vec::new(),
            },
            AnalysisMeshNode {
                node_id: 7,
                coordinates_m: [10.0, 1.0, 0.0],
                provenance: Vec::new(),
            },
            AnalysisMeshNode {
                node_id: 8,
                coordinates_m: [10.0, 0.0, 1.0],
                provenance: Vec::new(),
            },
        ]);
        mesh.volume_elements.push(AnalysisVolumeElement {
            element_id: "e2".to_string(),
            kind: VolumeElementKind::Tet4,
            node_ids: vec![5, 6, 7, 8],
            material_region_id: "mat_region".to_string(),
            provenance: Vec::new(),
        });

        let err = validate_analysis_mesh_with_options(
            &mesh,
            AnalysisMeshValidationOptions {
                max_volume_component_count: Some(1),
                ..AnalysisMeshValidationOptions::default()
            },
        )
        .expect_err("isolated volume component should fail");

        assert_eq!(
            err,
            AnalysisMeshValidationError::VolumeComponentCountExceeded {
                component_count: 2,
                max_component_count: 1,
            }
        );
    }

    #[test]
    fn rejects_unsupported_element_kind_until_assembly_exists() {
        let mut mesh = valid_tet_mesh();
        mesh.volume_elements[0].kind = VolumeElementKind::Hex8;
        mesh.volume_elements[0].node_ids = vec![1, 2, 3, 4, 1, 2, 3, 4];
        let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
            .expect_err("unsupported element kind should fail");
        assert_eq!(
            err,
            AnalysisMeshValidationError::UnsupportedVolumeElementKind {
                element_id: "e1".to_string()
            }
        );
    }

    #[test]
    fn rejects_missing_material_coverage() {
        let mut mesh = valid_tet_mesh();
        mesh.volume_elements[0].material_region_id.clear();
        let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
            .expect_err("missing material region should fail");
        assert_eq!(
            err,
            AnalysisMeshValidationError::MissingMaterialRegion {
                element_id: "e1".to_string()
            }
        );
    }

    #[test]
    fn rejects_unmapped_boundary_nodes() {
        let mut mesh = valid_tet_mesh();
        mesh.boundary_faces[0].node_ids = vec![1, 2, 99];
        let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
            .expect_err("unknown boundary node should fail");
        assert_eq!(
            err,
            AnalysisMeshValidationError::UnknownBoundaryFaceNode {
                face_id: "f1".to_string(),
                node_id: 99
            }
        );
    }

    #[test]
    fn rejects_unmapped_boundary_edge_nodes() {
        let mut mesh = valid_tet_mesh();
        mesh.boundary_edges = vec![AnalysisBoundaryEdge {
            edge_id: "edge1".to_string(),
            node_ids: [1, 99],
            adjacent_boundary_face_ids: vec!["f1".to_string()],
            region_ids: Vec::new(),
            provenance: Vec::new(),
        }];
        let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
            .expect_err("unknown boundary edge node should fail");
        assert_eq!(
            err,
            AnalysisMeshValidationError::UnknownBoundaryEdgeNode {
                edge_id: "edge1".to_string(),
                node_id: 99
            }
        );
    }

    #[test]
    fn rejects_missing_boundary_edge_recovery_when_required() {
        let mesh = valid_tet_mesh();
        let err = validate_analysis_mesh_with_options(
            &mesh,
            AnalysisMeshValidationOptions {
                min_boundary_edge_recovery_ratio: 1.0,
                ..AnalysisMeshValidationOptions::default()
            },
        )
        .expect_err("missing boundary edge recovery should fail");
        assert_eq!(
            err,
            AnalysisMeshValidationError::BoundaryEdgeRecoveryFailed {
                recovery_ratio: "0.000000".to_string(),
                required_ratio: "1.000000".to_string(),
            }
        );
    }

    #[test]
    fn rejects_boundary_edge_adjacent_to_unknown_face() {
        let mut mesh = valid_tet_mesh();
        mesh.boundary_edges = vec![AnalysisBoundaryEdge {
            edge_id: "edge1".to_string(),
            node_ids: [1, 2],
            adjacent_boundary_face_ids: vec!["missing_face".to_string()],
            region_ids: Vec::new(),
            provenance: Vec::new(),
        }];
        let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
            .expect_err("unknown boundary edge adjacent face should fail");
        assert_eq!(
            err,
            AnalysisMeshValidationError::UnknownBoundaryEdgeAdjacentFace {
                edge_id: "edge1".to_string(),
                face_id: "missing_face".to_string()
            }
        );
    }

    #[test]
    fn rejects_quality_threshold_failures() {
        let mut mesh = valid_tet_mesh();
        mesh.quality.min_scaled_jacobian = 0.01;
        let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
            .expect_err("low jacobian should fail");
        assert_eq!(
            analysis_mesh_validation_error_code(&err),
            "quality_threshold_failed"
        );
        assert_eq!(
            err,
            AnalysisMeshValidationError::QualityThresholdFailed {
                reason: "min_scaled_jacobian".to_string()
            }
        );
    }

    #[test]
    fn rejects_exact_quality_threshold_failures() {
        let mut mesh = valid_tet_mesh();
        mesh.quality.min_exact_scaled_jacobian = 0.01;
        let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
            .expect_err("low exact jacobian should fail");
        assert_eq!(
            err,
            AnalysisMeshValidationError::QualityThresholdFailed {
                reason: "min_exact_scaled_jacobian".to_string()
            }
        );
    }

    #[test]
    fn rejects_element_exact_quality_threshold_failures() {
        let mut mesh = valid_tet_mesh();
        mesh.quality.elements.push(ElementQuality {
            element_id: "e1".to_string(),
            scaled_jacobian: 0.8,
            exact_scaled_jacobian: 0.01,
            aspect_ratio: 1.0,
            volume_m3: 1.0 / 6.0,
        });
        let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
            .expect_err("low element exact jacobian should fail");
        assert_eq!(
            err,
            AnalysisMeshValidationError::QualityThresholdFailed {
                reason: "element_exact_scaled_jacobian".to_string()
            }
        );
    }

    #[test]
    fn rejects_boundary_projection_quality_threshold_failures() {
        let mut mesh = valid_tet_mesh();
        mesh.quality.max_boundary_projection_error_m = 2.0e-6;

        let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
            .expect_err("boundary projection error should fail");

        assert_eq!(
            err,
            AnalysisMeshValidationError::QualityThresholdFailed {
                reason: "max_boundary_projection_error_m".to_string()
            }
        );
    }

    #[test]
    fn rejects_mesh_that_underfills_expected_bounds() {
        let mesh = valid_tet_mesh();
        let err = validate_analysis_mesh_with_options(
            &mesh,
            AnalysisMeshValidationOptions {
                expected_bounds_m: Some([[0.0, 0.0, 0.0], [4.0, 1.0, 1.0]]),
                ..AnalysisMeshValidationOptions::default()
            },
        )
        .expect_err("mesh should fail bounds coverage");
        assert_eq!(
            err,
            AnalysisMeshValidationError::BoundsCoverageFailed {
                axis: 0,
                coverage_ratio: "0.250000".to_string(),
                required_ratio: "0.900000".to_string(),
            }
        );
    }

    #[test]
    fn rejects_mesh_that_underfills_expected_volume() {
        let mesh = valid_tet_mesh();
        let err = validate_analysis_mesh_with_options(
            &mesh,
            AnalysisMeshValidationOptions {
                expected_volume_m3: Some(1.0),
                ..AnalysisMeshValidationOptions::default()
            },
        )
        .expect_err("mesh should fail volume coverage");
        assert_eq!(
            err,
            AnalysisMeshValidationError::VolumeCoverageFailed {
                coverage_ratio: "0.166667".to_string(),
                required_ratio: "0.900000".to_string(),
            }
        );
    }

    #[test]
    fn rejects_uncovered_interior_coverage_samples() {
        let mesh = valid_tet_mesh();
        let err = validate_analysis_mesh_with_options(
            &mesh,
            AnalysisMeshValidationOptions {
                coverage_sample_points_m: vec![[0.1, 0.1, 0.1], [2.0, 2.0, 2.0]],
                min_coverage_sample_ratio: 1.0,
                ..AnalysisMeshValidationOptions::default()
            },
        )
        .expect_err("uncovered interior coverage sample should fail");
        assert_eq!(
            err,
            AnalysisMeshValidationError::CoverageSampleFailed {
                coverage_ratio: "0.500000".to_string(),
                required_ratio: "1.000000".to_string(),
            }
        );
    }

    #[test]
    fn accepts_covered_interior_coverage_samples() {
        let mesh = valid_tet_mesh();
        validate_analysis_mesh_with_options(
            &mesh,
            AnalysisMeshValidationOptions {
                coverage_sample_points_m: vec![[0.1, 0.1, 0.1]],
                min_coverage_sample_ratio: 1.0,
                ..AnalysisMeshValidationOptions::default()
            },
        )
        .expect("covered interior coverage sample should pass");
    }

    #[test]
    fn rejects_nearby_uncovered_samples_for_small_tets() {
        let mut mesh = valid_tet_mesh();
        for node in &mut mesh.nodes {
            for coordinate in &mut node.coordinates_m {
                *coordinate *= 1.0e-3;
            }
        }
        let err = validate_analysis_mesh_with_options(
            &mesh,
            AnalysisMeshValidationOptions {
                coverage_sample_points_m: vec![[1.01e-3, 1.0e-6, 1.0e-6]],
                min_coverage_sample_ratio: 1.0,
                ..AnalysisMeshValidationOptions::default()
            },
        )
        .expect_err("sample outside a small tet should fail");
        assert_eq!(
            err,
            AnalysisMeshValidationError::CoverageSampleFailed {
                coverage_ratio: "0.000000".to_string(),
                required_ratio: "1.000000".to_string(),
            }
        );
    }

    #[test]
    fn rejects_mesh_that_underfills_expected_boundary_area() {
        let mesh = valid_tet_mesh();
        let err = validate_analysis_mesh_with_options(
            &mesh,
            AnalysisMeshValidationOptions {
                expected_boundary_area_m2: Some(2.0),
                ..AnalysisMeshValidationOptions::default()
            },
        )
        .expect_err("mesh should fail boundary area coverage");
        assert_eq!(
            err,
            AnalysisMeshValidationError::BoundaryAreaCoverageFailed {
                area_ratio: "0.250000".to_string(),
                required_ratio: "0.900000".to_string(),
            }
        );
    }

    #[test]
    fn rejects_unrecovered_boundary_faces_when_required() {
        let mut mesh = valid_tet_mesh();
        mesh.boundary_faces[0].adjacent_volume_element_ids.clear();
        let err = validate_analysis_mesh_with_options(
            &mesh,
            AnalysisMeshValidationOptions {
                min_boundary_face_recovery_ratio: 1.0,
                ..AnalysisMeshValidationOptions::default()
            },
        )
        .expect_err("missing boundary recovery should fail");
        assert_eq!(
            err,
            AnalysisMeshValidationError::BoundaryFaceRecoveryFailed {
                recovery_ratio: "0.000000".to_string(),
                required_ratio: "1.000000".to_string(),
            }
        );
    }

    #[test]
    fn rejects_missing_required_boundary_region() {
        let mesh = valid_tet_mesh();
        let err = validate_analysis_mesh_with_options(
            &mesh,
            AnalysisMeshValidationOptions {
                required_boundary_region_ids: vec!["loaded".to_string()],
                ..AnalysisMeshValidationOptions::default()
            },
        )
        .expect_err("missing boundary region should fail");
        assert_eq!(
            err,
            AnalysisMeshValidationError::MissingRequiredBoundaryRegion {
                region_id: "loaded".to_string()
            }
        );
    }

    #[test]
    fn rejects_required_boundary_region_without_recovered_face() {
        let mut mesh = valid_tet_mesh();
        mesh.boundary_faces[0].adjacent_volume_element_ids.clear();
        let err = validate_analysis_mesh_with_options(
            &mesh,
            AnalysisMeshValidationOptions {
                required_boundary_region_ids: vec!["fixed".to_string()],
                ..AnalysisMeshValidationOptions::default()
            },
        )
        .expect_err("unrecovered boundary region should fail");
        assert_eq!(
            err,
            AnalysisMeshValidationError::MissingRequiredBoundaryRegionRecovery {
                region_id: "fixed".to_string()
            }
        );
    }

    #[test]
    fn rejects_missing_required_material_region() {
        let mesh = valid_tet_mesh();
        let err = validate_analysis_mesh_with_options(
            &mesh,
            AnalysisMeshValidationOptions {
                required_material_region_ids: vec!["rib".to_string()],
                ..AnalysisMeshValidationOptions::default()
            },
        )
        .expect_err("missing material region should fail");
        assert_eq!(
            analysis_mesh_validation_error_code(&err),
            "missing_required_material_region"
        );
        assert_eq!(
            err,
            AnalysisMeshValidationError::MissingRequiredMaterialRegion {
                region_id: "rib".to_string()
            }
        );
    }

    #[test]
    fn rejects_required_material_region_without_positive_volume() {
        let mut mesh = valid_tet_mesh();
        mesh.nodes[3].coordinates_m = mesh.nodes[0].coordinates_m;
        let err = validate_analysis_mesh_with_options(
            &mesh,
            AnalysisMeshValidationOptions {
                required_material_region_ids: vec!["mat_region".to_string()],
                ..AnalysisMeshValidationOptions::default()
            },
        )
        .expect_err("zero-volume material region should fail");
        assert_eq!(
            analysis_mesh_validation_error_code(&err),
            "missing_required_material_region_coverage"
        );
        assert_eq!(
            err,
            AnalysisMeshValidationError::MissingRequiredMaterialRegionCoverage {
                region_id: "mat_region".to_string()
            }
        );
    }
}
