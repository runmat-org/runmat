use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use runmat_meshing_core::{
    contracts::{
        MeshingStage, StageEvidence, TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ACCEPTED_COUNT,
        TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ATTEMPT_COUNT,
        TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTED_COUNT,
        TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTION_PREFIX,
    },
    quality::predicate::Point3,
};

#[cfg(test)]
mod diagnostics;
#[cfg(test)]
pub(crate) use diagnostics::*;
mod errors;
pub use errors::*;

pub(super) const MAX_MULTI_INTERIOR_REFILL_NODES: usize = 6;
pub(super) const MAX_MULTI_INTERIOR_REFILL_CANDIDATES: usize = 512;
pub(super) const MAX_CONSTRAINED_CAVITY_EXPANSION_STEPS: usize = 64;
pub(super) const MAX_CAP_SIDE_CONNECTOR_CHAIN_DEPTH: usize = 2;
pub(super) const MAX_CAP_SIDE_CONNECTOR_CHAIN_FACES_PER_DEPTH: usize = 128;
pub(super) const MAX_CAP_SIDE_CONNECTORS_PER_CHAIN_FACE: usize = 2;
pub(super) const MAX_CAP_SIDE_CONNECTOR_CHAIN_CANDIDATES: usize = 512;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavity {
    pub removed_tetrahedron_ids: Vec<u32>,
    pub boundary_faces: Vec<ConstrainedCavityBoundaryFace>,
    #[serde(default)]
    pub protected_node_ids: Vec<u32>,
    pub target_volume_m3: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavityBoundaryFace {
    pub node_ids: [u32; 3],
    #[serde(default)]
    pub outside_tetrahedron_ids: Vec<u32>,
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
pub struct ConstrainedCavityRefillTetrahedron {
    pub node_ids: [u32; 4],
    pub volume_m3: f64,
    pub aspect_ratio: f64,
    pub exact_scaled_jacobian: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CavityTetrahedron {
    pub tetrahedron_id: u32,
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
pub(super) struct ConnectivityTetrahedron {
    pub(super) vertices: [usize; 4],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavityRefill {
    pub tetrahedra: Vec<ConstrainedCavityRefillTetrahedron>,
    pub boundary_faces: Vec<ConstrainedCavityBoundaryFace>,
    #[serde(default)]
    pub inserted_nodes: Vec<ConstrainedCavityNode>,
    pub total_volume_m3: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryExactCoverFaceCountBlockers {
    pub target_face: [u32; 3],
    pub selected_tetrahedron_count: usize,
    pub candidate_count: usize,
    pub blocker_count: usize,
    pub blockers: Vec<BoundaryExactCoverFaceCountBlocker>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryExactCoverFaceCountBlocker {
    pub node_ids: [u32; 4],
    pub exact_scaled_jacobian: f64,
    pub conflicting_faces: Vec<[u32; 3]>,
    pub blocking_selected_tetrahedra: Vec<[u32; 4]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryExactCoverSaturatedComponent {
    pub seed_face: [u32; 3],
    pub saturated_face_count: usize,
    pub component_face_count: usize,
    pub component_tetrahedron_count: usize,
    pub component_faces: Vec<[u32; 3]>,
    pub component_tetrahedra: Vec<[u32; 4]>,
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
    #[serde(default)]
    pub local_reconnection_attempt_count: usize,
    #[serde(default)]
    pub local_reconnection_accepted_count: usize,
    #[serde(default)]
    pub local_reconnection_rejected_count: usize,
    #[serde(default)]
    pub local_reconnection_rejected_by_reason: BTreeMap<String, usize>,
}

impl ConstrainedCavityRefillEvaluation {
    pub fn optimization_stage_evidence(&self) -> StageEvidence {
        let mut evidence = StageEvidence::complete(MeshingStage::Optimization);
        evidence.entity_counts.insert(
            TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ATTEMPT_COUNT.to_string(),
            self.local_reconnection_attempt_count,
        );
        evidence.entity_counts.insert(
            TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ACCEPTED_COUNT.to_string(),
            self.local_reconnection_accepted_count,
        );
        evidence.entity_counts.insert(
            TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTED_COUNT.to_string(),
            self.local_reconnection_rejected_count,
        );
        for (reason, count) in &self.local_reconnection_rejected_by_reason {
            evidence.rejection_counts.insert(
                format!("{TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTION_PREFIX}{reason}"),
                *count,
            );
        }
        evidence
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCavitySourceEdgeSplit {
    pub cavity: ConstrainedCavity,
    pub split_node: ConstrainedCavityNode,
    pub source_tetrahedra: Vec<CavityTetrahedron>,
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
    pub added_tetrahedron_ids: Vec<u32>,
    pub removed_tetrahedron_count_before: usize,
    pub removed_tetrahedron_count_after: usize,
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
    pub expanded_removed_tetrahedron_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstrainedCavitySolidEmptyBoundaryRejectedSplit {
    pub input_faces: Vec<[u32; 3]>,
    pub output_faces: Vec<[u32; 3]>,
    pub split_node_count: usize,
    pub split_step_count: usize,
}
