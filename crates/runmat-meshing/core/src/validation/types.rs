use serde::{Deserialize, Serialize};

use crate::quality::QualityThresholds;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
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
    pub require_boundary_source_edge_provenance: bool,
    pub require_no_unrecovered_tetrahedron_components: bool,
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
            require_boundary_source_edge_provenance: false,
            require_no_unrecovered_tetrahedron_components: false,
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
    MissingPlcInputEvidence {
        reason: String,
    },
    MissingBoundarySourceFaceProvenance {
        face_id: String,
    },
    MissingBoundarySourceEdgeProvenance {
        recovered_edge_count: usize,
        required_edge_count: usize,
    },
    UnrecoveredTetrahedronComponentsPresent {
        component_count: usize,
    },
    UnrepairedExactQualityPresent {
        total_count: usize,
        general_cavity_count: usize,
        boundary_adjacent_count: usize,
        node_adjacent_count: usize,
        interior_seed_count: usize,
        edge_star_count: usize,
    },
    RolledBackMaterialInterfacePartitionRecoveryPresent {
        recovery_item_count: usize,
        element_count: usize,
        boundary_face_count: usize,
        post_insertion_audit_rejection_count: usize,
    },
    IncompleteTetrahedronRecoveryPresent {
        missing_item_count: usize,
        missing_source_face_item_count: usize,
        missing_source_edge_item_count: usize,
        missing_material_interface_item_count: usize,
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
        AnalysisMeshValidationError::MissingPlcInputEvidence { .. } => "missing_plc_input_evidence",
        AnalysisMeshValidationError::MissingBoundarySourceFaceProvenance { .. } => {
            "missing_boundary_source_face_provenance"
        }
        AnalysisMeshValidationError::MissingBoundarySourceEdgeProvenance { .. } => {
            "missing_boundary_source_edge_provenance"
        }
        AnalysisMeshValidationError::UnrecoveredTetrahedronComponentsPresent { .. } => {
            "unrecovered_tetrahedron_components_present"
        }
        AnalysisMeshValidationError::UnrepairedExactQualityPresent { .. } => {
            "unrepaired_exact_quality_present"
        }
        AnalysisMeshValidationError::RolledBackMaterialInterfacePartitionRecoveryPresent {
            ..
        } => "rolled_back_material_interface_partition_recovery_present",
        AnalysisMeshValidationError::IncompleteTetrahedronRecoveryPresent { .. } => {
            "incomplete_tetrahedron_recovery_present"
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
