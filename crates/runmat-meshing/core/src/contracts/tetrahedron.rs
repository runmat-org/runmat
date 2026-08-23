use serde::{Deserialize, Serialize};

use super::{StageEvidence, TopologyEntityId};

pub const TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ATTEMPT_COUNT: &str =
    "optimization_local_reconnection_attempts";
pub const TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ACCEPTED_COUNT: &str =
    "optimization_local_reconnection_accepted";
pub const TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTED_COUNT: &str =
    "optimization_local_reconnection_rejected";
pub const TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_BUDGET_LIMIT_COUNT: &str =
    "optimization_local_reconnection_budget_limited";
pub const TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTION_PREFIX: &str =
    "optimization_local_reconnection_rejected_";
pub const TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ATTEMPT_COUNT: &str =
    "optimization_interior_smoothing_attempts";
pub const TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ACCEPTED_COUNT: &str =
    "optimization_interior_smoothing_accepted";
pub const TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_REJECTED_COUNT: &str =
    "optimization_interior_smoothing_rejected";
pub const TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_BUDGET_LIMIT_COUNT: &str =
    "optimization_interior_smoothing_budget_limited";
pub const TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_REJECTION_PREFIX: &str =
    "optimization_interior_smoothing_rejected_";
pub const TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ATTEMPT_COUNT: &str =
    "optimization_boundary_smoothing_attempts";
pub const TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ACCEPTED_COUNT: &str =
    "optimization_boundary_smoothing_accepted";
pub const TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_REJECTED_COUNT: &str =
    "optimization_boundary_smoothing_rejected";
pub const TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_BUDGET_LIMIT_COUNT: &str =
    "optimization_boundary_smoothing_budget_limited";
pub const TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_REJECTION_PREFIX: &str =
    "optimization_boundary_smoothing_rejected_";
pub const TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_ATTEMPT_COUNT: &str =
    "optimization_sliver_removal_attempts";
pub const TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_ACCEPTED_COUNT: &str =
    "optimization_sliver_removal_accepted";
pub const TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_REJECTED_COUNT: &str =
    "optimization_sliver_removal_rejected";
pub const TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_BUDGET_LIMIT_COUNT: &str =
    "optimization_sliver_removal_budget_limited";
pub const TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_REJECTION_PREFIX: &str =
    "optimization_sliver_removal_rejected_";
pub const TETRAHEDRON_UNTANGLING_PASS_COUNT: &str = "untangling_passes";
pub const TETRAHEDRON_UNTANGLING_INITIAL_NEAR_SINGULAR_COUNT: &str =
    "untangling_initial_near_singular";
pub const TETRAHEDRON_UNTANGLING_FINAL_NEAR_SINGULAR_COUNT: &str = "untangling_final_near_singular";
pub const TETRAHEDRON_UNTANGLING_RELOCATED_SEED_COUNT: &str = "untangling_relocated_seeds";
pub const TETRAHEDRON_UNTANGLING_REJECTION_PREFIX: &str = "untangling_rejected_";
pub const TETRAHEDRON_EXACT_QUALITY_REPAIR_PASS_COUNT: &str = "exact_quality_repair_passes";
pub const TETRAHEDRON_EXACT_QUALITY_SEED_STAR_RELOCATION_COUNT: &str =
    "exact_quality_seed_star_relocations";
pub const TETRAHEDRON_EXACT_QUALITY_UNREPAIRED_TOTAL_COUNT: &str = "exact_quality_unrepaired_total";
pub const TETRAHEDRON_EXACT_QUALITY_UNREPAIRED_INTERIOR_SEED_COUNT: &str =
    "exact_quality_unrepaired_interior_seeds";
pub const TETRAHEDRON_EXACT_QUALITY_REPAIR_REJECTION_PREFIX: &str =
    "exact_quality_repair_rejected_";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetrahedronMesh {
    pub mesh_id: String,
    #[serde(default)]
    pub nodes: Vec<TetrahedronMeshNode>,
    #[serde(default)]
    pub elements: Vec<Tetrahedron4Element>,
    #[serde(default)]
    pub boundary_faces: Vec<TetrahedronBoundaryFace>,
    pub recovery_complete: bool,
    pub quality_optimized: bool,
    pub evidence: StageEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetrahedronMeshNode {
    pub node_id: TopologyEntityId,
    pub coordinates_m: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tetrahedron4Element {
    pub element_id: TopologyEntityId,
    pub node_ids: [TopologyEntityId; 4],
    pub material_region_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetrahedronBoundaryFace {
    pub face_id: TopologyEntityId,
    pub node_ids: [TopologyEntityId; 3],
    pub source_face_id: TopologyEntityId,
    #[serde(default)]
    pub source_edge_ids: [Option<TopologyEntityId>; 3],
}
