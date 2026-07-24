use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryRecoveryPriority {
    Edge,
    Face,
    Provenance,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryRecoveryReason {
    MissingEdge,
    MissingFace,
    OutsideTetrahedronMismatch,
    SourceEdgeMismatch,
    SourceFaceMismatch,
    RegionMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundaryRecoveryQueueItem {
    pub priority: BoundaryRecoveryPriority,
    pub reason: BoundaryRecoveryReason,
    #[serde(default)]
    pub face_node_ids: Option<[u32; 3]>,
    #[serde(default)]
    pub edge_node_ids: Option<[u32; 2]>,
    #[serde(default)]
    pub source_face_id: Option<u32>,
    #[serde(default)]
    pub source_edge_id: Option<u32>,
    #[serde(default)]
    pub outside_tetrahedron_ids: Vec<u32>,
    #[serde(default)]
    pub region_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundaryRecoveryQueue {
    pub items: Vec<BoundaryRecoveryQueueItem>,
    pub missing_edge_count: usize,
    pub missing_face_count: usize,
    #[serde(default)]
    pub interface_mismatch_count: usize,
    pub provenance_mismatch_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryRecoveryQueueError {
    DegenerateBoundaryFace { node_ids: [u32; 3] },
    DuplicateBoundaryFace { node_ids: [u32; 3] },
}
