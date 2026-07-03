use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::predicate::Point3;

pub(super) const MAX_MULTI_INTERIOR_REFILL_NODES: usize = 6;
pub(super) const MAX_MULTI_INTERIOR_REFILL_CANDIDATES: usize = 512;
pub(super) const MAX_CONSTRAINED_CAVITY_EXPANSION_STEPS: usize = 64;
pub(super) const MAX_CAP_SIDE_CONNECTOR_CHAIN_DEPTH: usize = 2;
pub(super) const MAX_CAP_SIDE_CONNECTOR_CHAIN_FACES_PER_DEPTH: usize = 128;
pub(super) const MAX_CAP_SIDE_CONNECTORS_PER_CHAIN_FACE: usize = 2;
pub(super) const MAX_CAP_SIDE_CONNECTOR_CHAIN_CANDIDATES: usize = 512;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavity {
    pub removed_tet_ids: Vec<u32>,
    pub boundary_faces: Vec<ConstrainedCavityBoundaryFace>,
    #[serde(default)]
    pub protected_node_ids: Vec<u32>,
    pub target_volume_m3: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavityBoundaryFace {
    pub node_ids: [u32; 3],
    #[serde(default)]
    pub outside_tet_ids: Vec<u32>,
    pub source_face_id: Option<u32>,
    #[serde(default)]
    pub source_edge_ids: [Option<u32>; 3],
    #[serde(default)]
    pub region_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavityValidationReport {
    pub boundary_face_count: usize,
    pub boundary_edge_count: usize,
    pub boundary_node_count: usize,
    pub protected_node_count: usize,
    pub target_volume_m3: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavityRefillOptions {
    pub min_volume_m3: f64,
    pub max_aspect_ratio: f64,
    pub min_scaled_jacobian: f64,
    pub volume_relative_tolerance: f64,
    pub min_protected_node_distance_m: f64,
}

impl Default for ConstrainedCavityRefillOptions {
    fn default() -> Self {
        Self {
            min_volume_m3: 1.0e-18,
            max_aspect_ratio: 1.0e6,
            min_scaled_jacobian: 0.15,
            volume_relative_tolerance: 1.0e-9,
            min_protected_node_distance_m: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavityNode {
    pub node_id: u32,
    pub coordinates_m: Point3,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavityRefillTet {
    pub node_ids: [u32; 4],
    pub volume_m3: f64,
    pub aspect_ratio: f64,
    pub exact_scaled_jacobian: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CavityTet {
    pub tet_id: u32,
    pub component_id: u32,
    pub node_ids: [u32; 4],
    pub source_surface_element_id: u32,
    pub region_ids: Vec<String>,
    pub volume_m3: f64,
    pub aspect_ratio: f64,
    #[serde(default)]
    pub exact_scaled_jacobian: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ConnectivityPoint {
    pub(super) node_id: u32,
    pub(super) coordinates_m: Point3,
    pub(super) is_super: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ConnectivityTet {
    pub(super) vertices: [usize; 4],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavityRefill {
    pub tets: Vec<ConstrainedCavityRefillTet>,
    pub boundary_faces: Vec<ConstrainedCavityBoundaryFace>,
    #[serde(default)]
    pub inserted_nodes: Vec<ConstrainedCavityNode>,
    pub total_volume_m3: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryNodeCompletionDiagnostic {
    pub reason: &'static str,
    pub missing_face_count: usize,
    pub cap_candidate_count: usize,
    pub outside_candidate_count: usize,
    pub duplicate_candidate_count: usize,
    pub max_rejected_scaled_jacobian: f64,
    pub rejected_scaled_jacobian_bins: BTreeMap<String, usize>,
    pub max_rejected_cap_height_ratio: f64,
    pub rejected_cap_height_ratio_bins: BTreeMap<String, usize>,
    pub rejected_scaled_jacobian_worst_corner_bins: BTreeMap<&'static str, usize>,
    pub rejected_cap_node_ids: BTreeMap<u32, usize>,
    pub split_cap_candidate_count: usize,
    pub split_cap_pass_count: usize,
    pub max_split_cap_scaled_jacobian: f64,
    pub split_cap_scaled_jacobian_bins: BTreeMap<String, usize>,
    pub split_cap_scaled_jacobian_worst_corner_bins: BTreeMap<&'static str, usize>,
    pub split_cap_apex_limited_node_ids: BTreeMap<u32, usize>,
    pub edge_split_cap_candidate_count: usize,
    pub edge_split_cap_pass_count: usize,
    pub max_edge_split_cap_scaled_jacobian: f64,
    pub edge_split_cap_scaled_jacobian_bins: BTreeMap<String, usize>,
    pub edge_split_cap_scaled_jacobian_worst_corner_bins: BTreeMap<&'static str, usize>,
    pub edge_split_cap_apex_limited_node_ids: BTreeMap<u32, usize>,
    pub three_edge_split_cap_candidate_count: usize,
    pub three_edge_split_cap_pass_count: usize,
    pub max_three_edge_split_cap_scaled_jacobian: f64,
    pub three_edge_split_cap_scaled_jacobian_bins: BTreeMap<String, usize>,
    pub three_edge_split_cap_scaled_jacobian_worst_corner_bins: BTreeMap<&'static str, usize>,
    pub three_edge_split_cap_apex_limited_node_ids: BTreeMap<u32, usize>,
    pub rejected_by_reason: BTreeMap<&'static str, usize>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryExactCoverDiagnostic {
    pub boundary_node_count: usize,
    pub boundary_face_count: usize,
    pub candidate_count: usize,
    pub solid_candidate_count: usize,
    pub zero_candidate_boundary_face_count: usize,
    pub zero_candidate_boundary_faces: Vec<[u32; 3]>,
    pub min_boundary_face_candidate_count: usize,
    pub min_candidate_boundary_faces: Vec<[u32; 3]>,
    pub max_boundary_face_candidate_count: usize,
    pub zero_solid_candidate_boundary_face_count: usize,
    pub zero_solid_candidate_boundary_faces: Vec<[u32; 3]>,
    pub min_solid_boundary_face_candidate_count: usize,
    pub min_solid_candidate_boundary_faces: Vec<[u32; 3]>,
    pub max_solid_boundary_face_candidate_count: usize,
    pub zero_addable_boundary_face_count: usize,
    pub zero_addable_boundary_faces: Vec<[u32; 3]>,
    pub min_addable_boundary_face_candidate_count: usize,
    pub min_addable_candidate_boundary_faces: Vec<[u32; 3]>,
    pub selected_tet_count: usize,
    pub search_attempt_count: usize,
    pub found_cover: bool,
    pub reason: &'static str,
    pub dead_end_reason: &'static str,
    pub dead_end_face: Option<[u32; 3]>,
    pub dead_end_depth: usize,
    pub dead_end_selected_tets: Vec<[u32; 4]>,
    pub dead_end_current_volume_m3: f64,
    pub dead_end_candidate_volume_m3: f64,
    pub dead_end_target_volume_m3: f64,
    pub dead_end_reason_histogram: BTreeMap<&'static str, usize>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SupportNodeExactCoverDiagnostic {
    pub candidate_node_count: usize,
    pub candidate_count: usize,
    pub root_zero_raw_boundary_face_count: usize,
    pub root_zero_raw_boundary_faces: Vec<[u32; 3]>,
    pub root_min_raw_boundary_face_candidate_count: usize,
    pub root_min_raw_candidate_boundary_faces: Vec<[u32; 3]>,
    pub root_max_raw_boundary_face_candidate_count: usize,
    pub root_zero_addable_boundary_face_count: usize,
    pub root_zero_addable_boundary_faces: Vec<[u32; 3]>,
    pub root_min_addable_boundary_face_candidate_count: usize,
    pub root_min_addable_candidate_boundary_faces: Vec<[u32; 3]>,
    pub root_max_addable_boundary_face_candidate_count: usize,
    pub selected_tet_count: usize,
    pub search_attempt_count: usize,
    pub found_cover: bool,
    pub reason: &'static str,
    pub dead_end_reason: &'static str,
    pub dead_end_face: Option<[u32; 3]>,
    pub dead_end_depth: usize,
    pub dead_end_reason_histogram: BTreeMap<&'static str, usize>,
    pub dead_end_faces_by_reason: BTreeMap<&'static str, Vec<[u32; 3]>>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryExactCoverMateDiagnostic {
    pub target_face: [u32; 3],
    pub candidate_count: usize,
    pub addable_count: usize,
    pub candidates: Vec<BoundaryExactCoverMateCandidateDiagnostic>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryExactCoverMateCandidateDiagnostic {
    pub node_ids: [u32; 4],
    pub exact_scaled_jacobian: f64,
    pub addable: bool,
    pub conflicting_faces: Vec<[u32; 3]>,
    pub missing_future_mate_faces: Vec<[u32; 3]>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryExactCoverInteriorMateClosureDiagnostic {
    pub initial_candidate_count: usize,
    pub candidate_count: usize,
    pub injected_candidate_count: usize,
    pub found_cover: bool,
    pub selected_tet_count: usize,
    pub search_attempt_count: usize,
    pub reason: &'static str,
    pub dead_end_reason: &'static str,
    pub dead_end_face: Option<[u32; 3]>,
    pub dead_end_depth: usize,
    pub dead_end_selected_tets: Vec<[u32; 4]>,
    pub dead_end_current_volume_m3: f64,
    pub dead_end_candidate_volume_m3: f64,
    pub dead_end_target_volume_m3: f64,
    pub dead_end_reason_histogram: BTreeMap<&'static str, usize>,
    pub dead_end_faces_by_reason: BTreeMap<&'static str, Vec<[u32; 3]>>,
    pub dead_end_selected_tets_by_reason: BTreeMap<&'static str, Vec<[u32; 4]>>,
    pub dead_end_selected_roles_by_reason: BTreeMap<&'static str, Vec<&'static str>>,
    pub unforced_found_cover: bool,
    pub unforced_selected_tet_count: usize,
    pub unforced_search_attempt_count: usize,
    pub unforced_dead_end_reason_histogram: BTreeMap<&'static str, usize>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryExactCoverFaceCandidateSourceDiagnostic {
    pub target_face: [u32; 3],
    pub fourth_node_count: usize,
    pub centroid_inside_count: usize,
    pub solid_pass_count: usize,
    pub relaxed_pass_count: usize,
    pub outside_surface_count: usize,
    pub solid_rejected_by_reason: BTreeMap<&'static str, usize>,
    pub relaxed_rejected_by_reason: BTreeMap<&'static str, usize>,
    pub relaxed_candidate_node_ids: Vec<[u32; 4]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryExactCoverFaceCountBlockers {
    pub target_face: [u32; 3],
    pub selected_tet_count: usize,
    pub candidate_count: usize,
    pub blocker_count: usize,
    pub blockers: Vec<BoundaryExactCoverFaceCountBlocker>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryExactCoverFaceCountBlocker {
    pub node_ids: [u32; 4],
    pub exact_scaled_jacobian: f64,
    pub conflicting_faces: Vec<[u32; 3]>,
    pub blocking_selected_tets: Vec<[u32; 4]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryExactCoverSaturatedComponent {
    pub seed_face: [u32; 3],
    pub saturated_face_count: usize,
    pub component_face_count: usize,
    pub component_tet_count: usize,
    pub component_faces: Vec<[u32; 3]>,
    pub component_tets: Vec<[u32; 4]>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundarySteinerExactCoverDiagnostic {
    pub boundary_node_count: usize,
    pub boundary_face_count: usize,
    pub candidate_count: usize,
    pub zero_candidate_boundary_face_count: usize,
    pub min_boundary_face_candidate_count: usize,
    pub max_boundary_face_candidate_count: usize,
    pub selected_tet_count: usize,
    pub search_attempt_count: usize,
    pub found_cover: bool,
    pub reason: &'static str,
    pub max_min_scaled_jacobian: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryPatchSteinerExactCoverDiagnostic {
    pub boundary_node_count: usize,
    pub boundary_face_count: usize,
    pub missing_face_count: usize,
    pub patch_count: usize,
    pub steiner_node_count: usize,
    pub candidate_count: usize,
    pub zero_candidate_boundary_face_count: usize,
    pub min_boundary_face_candidate_count: usize,
    pub max_boundary_face_candidate_count: usize,
    pub selected_tet_count: usize,
    pub search_attempt_count: usize,
    pub found_cover: bool,
    pub reason: &'static str,
    pub max_min_scaled_jacobian: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MissingFaceLocalCapQualityDiagnostic {
    pub missing_face_count: usize,
    pub pass_face_count: usize,
    pub failed_face_count: usize,
    pub candidate_count: usize,
    pub candidate_source_bins: BTreeMap<&'static str, usize>,
    pub max_scaled_jacobian: f64,
    pub max_failed_face_scaled_jacobian: f64,
    pub failed_face_scaled_jacobian_bins: BTreeMap<String, usize>,
    pub failed_face_source_bins: BTreeMap<&'static str, usize>,
    pub rejected_by_reason: BTreeMap<&'static str, usize>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MissingFaceLocalCapStitchDiagnostic {
    pub missing_face_count: usize,
    pub missing_faces: Vec<[u32; 3]>,
    pub patch_count: usize,
    pub patch_size_histogram: BTreeMap<usize, usize>,
    pub patch_capped_face_count_histogram: BTreeMap<usize, usize>,
    pub incomplete_patch_size_histogram: BTreeMap<usize, usize>,
    pub uncapped_faces: Vec<[u32; 3]>,
    pub capped_face_count: usize,
    pub inserted_node_count: usize,
    pub side_connector_candidate_count: usize,
    pub candidate_tet_count: usize,
    pub cap_side_face_count: usize,
    pub zero_mate_cap_side_face_count: usize,
    pub min_cap_side_face_mate_count: usize,
    pub max_cap_side_face_mate_count: usize,
    pub open_interior_face_count: usize,
    pub open_interior_component_count: usize,
    pub open_interior_component_size_histogram: BTreeMap<usize, usize>,
    pub candidate_with_orphan_interior_face_count: usize,
    pub candidate_without_orphan_interior_face_count: usize,
    pub root_boundary_zero_raw_candidate_face_count: usize,
    pub root_boundary_zero_addable_candidate_face_count: usize,
    pub root_boundary_min_raw_candidate_count: usize,
    pub root_boundary_min_addable_candidate_count: usize,
    pub root_boundary_max_addable_candidate_count: usize,
    pub cover_dead_end_reason: &'static str,
    pub cover_dead_end_depth: usize,
    pub cover_dead_end_reason_histogram: BTreeMap<&'static str, usize>,
    pub selected_tet_count: usize,
    pub search_attempt_count: usize,
    pub found_cover: bool,
    pub reason: &'static str,
    pub max_min_scaled_jacobian: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryMissingFaceClusterDiagnostic {
    pub missing_face_count: usize,
    pub edge_component_count: usize,
    pub edge_component_size_histogram: BTreeMap<usize, usize>,
    pub node_component_count: usize,
    pub node_component_size_histogram: BTreeMap<usize, usize>,
    pub node_component_common_node_count_histogram: BTreeMap<usize, usize>,
    pub node_component_common_node_ids: BTreeMap<u32, usize>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InteriorStarQualityDiagnostic {
    pub candidate_count: usize,
    pub pass_count: usize,
    pub scaled_worst_face_candidate_count: usize,
    pub scaled_worst_face_pass_count: usize,
    pub max_min_scaled_jacobian: f64,
    pub max_scaled_worst_face_min_scaled_jacobian: f64,
    pub min_scaled_jacobian_bins: BTreeMap<String, usize>,
    pub min_scaled_jacobian_worst_corner_bins: BTreeMap<&'static str, usize>,
    pub rejected_by_reason: BTreeMap<&'static str, usize>,
}

pub(super) const MAX_ANCHOR_TRIM_STATES: usize = 128;
pub(super) const MAX_BOUNDARY_EXACT_COVER_NODES: usize = 20;
pub(super) const MAX_BOUNDARY_EXACT_COVER_FACES: usize = 40;
pub(super) const MAX_BOUNDARY_EXACT_COVER_CANDIDATES: usize = 512;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavityRefillEvaluation {
    pub refill: Option<ConstrainedCavityRefill>,
    #[serde(default)]
    pub rejected_by_reason: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityValidationError {
    EmptyRemovedTetSet,
    InvalidTargetVolume {
        target_volume_m3: f64,
    },
    TooFewBoundaryFaces {
        boundary_face_count: usize,
    },
    DegenerateBoundaryFace {
        face_index: usize,
        node_ids: [u32; 3],
    },
    DuplicateBoundaryFace {
        node_ids: [u32; 3],
    },
    NonManifoldBoundaryEdge {
        node_ids: [u32; 2],
        face_count: usize,
    },
    ProtectedNodeOutsideBoundary {
        node_id: u32,
    },
    InvalidRefillVolume {
        target_volume_m3: f64,
        candidate_volume_m3: f64,
        tolerance_m3: f64,
    },
    BoundaryFaceCountMismatch {
        expected_count: usize,
        candidate_count: usize,
    },
    MissingBoundaryFace {
        node_ids: [u32; 3],
    },
    UnexpectedBoundaryFace {
        node_ids: [u32; 3],
    },
    BoundarySourceFaceMismatch {
        node_ids: [u32; 3],
        expected_source_face_id: Option<u32>,
        candidate_source_face_id: Option<u32>,
    },
    BoundarySourceEdgeMismatch {
        node_ids: [u32; 2],
        expected_source_edge_id: Option<u32>,
        candidate_source_edge_id: Option<u32>,
    },
    BoundaryRegionMismatch {
        node_ids: [u32; 3],
        expected_region_ids: Vec<String>,
        candidate_region_ids: Vec<String>,
    },
    BoundaryOutsideTetMismatch {
        node_ids: [u32; 3],
        expected_outside_tet_ids: Vec<u32>,
        candidate_outside_tet_ids: Vec<u32>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityBoundarySplitError {
    SplitNodeReusesFaceNode { node_id: u32 },
    MissingBoundaryFace { node_ids: [u32; 3] },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityBoundaryEdgeSplitError {
    MissingBoundaryEdge {
        node_ids: [u32; 2],
    },
    MissingBoundaryNode {
        node_id: u32,
    },
    #[cfg(test)]
    InvalidPatchWeights {
        weights: [f64; 4],
    },
    Split(ConstrainedCavityBoundarySplitError),
    Validation(ConstrainedCavityValidationError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityBoundaryFaceSplitError {
    MissingBoundaryFace { node_ids: [u32; 3] },
    MissingBoundaryNode { node_id: u32 },
    DuplicateBoundaryFace { node_ids: [u32; 3] },
    InvalidBarycentricCoordinates { barycentric: [f64; 3] },
    Split(ConstrainedCavityBoundarySplitError),
    Validation(ConstrainedCavityValidationError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavitySourceEdgeSplit {
    pub cavity: ConstrainedCavity,
    pub split_node: ConstrainedCavityNode,
    pub source_tets: Vec<CavityTet>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavityBoundaryEdgeRecovery {
    pub cavity: ConstrainedCavity,
    pub attempted_boundary_faces: Vec<[u32; 3]>,
    pub recovered_edge: Option<ConstrainedCavityBoundaryEdgeRecoveryStep>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstrainedCavityBoundaryEdgeRecoveryStep {
    pub node_ids: [u32; 2],
    pub added_tet_ids: Vec<u32>,
    pub removed_tet_count_before: usize,
    pub removed_tet_count_after: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavityBoundaryEdgeRecoveryQueue {
    pub cavity: ConstrainedCavity,
    pub steps: Vec<ConstrainedCavityBoundaryEdgeRecoveryStep>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavityBoundaryPatchSplit {
    pub cavity: ConstrainedCavity,
    pub split_nodes: Vec<ConstrainedCavityNode>,
    pub steps: Vec<ConstrainedCavityBoundaryPatchSplitStep>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityBoundaryPatchSplitStep {
    EdgePatch {
        node_ids: [u32; 2],
        split_node_id: u32,
    },
    Face {
        node_ids: [u32; 3],
        split_node_id: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityBoundaryPatchSplitError {
    Edge(ConstrainedCavityBoundaryEdgeSplitError),
    Face(ConstrainedCavityBoundaryFaceSplitError),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstrainedCavitySolidEmptyBoundaryFaces {
    pub faces: Vec<[u32; 3]>,
    pub true_exterior_faces: Vec<[u32; 3]>,
    pub expandable_faces: Vec<[u32; 3]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavitySolidEmptyBoundaryRecovery {
    pub cavity: ConstrainedCavity,
    pub split_nodes: Vec<ConstrainedCavityNode>,
    pub classification: ConstrainedCavitySolidEmptyBoundaryFaces,
    pub split_steps: Vec<ConstrainedCavityBoundaryPatchSplitStep>,
    pub rejected_splits: Vec<ConstrainedCavitySolidEmptyBoundaryRejectedSplit>,
    pub expanded_removed_tet_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstrainedCavitySolidEmptyBoundaryRejectedSplit {
    pub input_faces: Vec<[u32; 3]>,
    pub output_faces: Vec<[u32; 3]>,
    pub split_node_count: usize,
    pub split_step_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavitySolidEmptyBoundaryRecoveryError {
    Refill(ConstrainedCavityRefillError),
    Split(ConstrainedCavityBoundaryPatchSplitError),
    Expansion(ConstrainedCavityExpansionError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavitySourceEdgeSplitError {
    MissingBoundaryEdge { node_ids: [u32; 2] },
    MissingBoundaryNode { node_id: u32 },
    MissingSourceNode { node_id: u32 },
    MissingRemovedSourceTet { tet_id: u32 },
    NoIncidentSourceTet { node_ids: [u32; 2] },
    DegenerateSplitTet { tet_id: u32 },
    Validation(ConstrainedCavityValidationError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityExpansionError {
    BoundaryFaceNotFound { node_ids: [u32; 3] },
    BoundaryFaceHasNoOutsideTet { node_ids: [u32; 3] },
    BoundaryEdgeHasNoOutsideTet { node_ids: [u32; 2] },
    SourceTetIdNotFound { tet_id: u32 },
    ExpansionDidNotConverge { step_count: usize },
    Extraction(ConstrainedCavityExtractionError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityExtractionError {
    EmptySelection,
    SelectedTetIndexOutOfBounds { tet_index: usize, tet_count: usize },
    DuplicateSelectedTetIndex { tet_index: usize },
    Validation(ConstrainedCavityValidationError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityRefillError {
    InvalidOptions,
    Validation(ConstrainedCavityValidationError),
    MissingBoundaryNode {
        node_id: u32,
    },
    DuplicateInteriorNode {
        node_id: u32,
    },
    InteriorNodeReusesBoundaryNode {
        node_id: u32,
    },
    InteriorPointOutsideCavity {
        node_id: u32,
    },
    NoValidCandidate {
        rejected_by_reason: BTreeMap<String, usize>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityRefillTetSplitError {
    InvalidBarycentricCoordinates {
        barycentric: [f64; 3],
    },
    MissingNode {
        node_id: u32,
    },
    FaceIncidenceNotTwo {
        node_ids: [u32; 3],
        incident_tet_count: usize,
    },
    RejectedChildTet {
        node_ids: [u32; 4],
        reason: &'static str,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityRefillTetFlipError {
    MissingNode {
        node_id: u32,
    },
    FaceIncidenceNotTwo {
        node_ids: [u32; 3],
        incident_tet_count: usize,
    },
    EdgeIncidenceNotThree {
        node_ids: [u32; 2],
        incident_tet_count: usize,
    },
    InvalidFlipTopology {
        reason: &'static str,
    },
    RejectedCreatedTet {
        node_ids: [u32; 4],
        reason: &'static str,
    },
}
