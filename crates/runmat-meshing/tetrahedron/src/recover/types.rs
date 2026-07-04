use serde::{Deserialize, Serialize};

use runmat_meshing_core::contracts::{StageEvidence, TetrahedronMesh, TopologyEntityId};
use runmat_meshing_plc::validate::PlcValidationError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TetrahedronRecoveryError {
    InvalidProtectedBoundaryComplex {
        error: PlcValidationError,
    },
    EmptyTetrahedronMesh,
    IncompleteRecovery {
        missing_item_count: usize,
        missing_source_face_item_count: usize,
        missing_source_edge_item_count: usize,
        missing_material_interface_item_count: usize,
    },
}

impl std::fmt::Display for TetrahedronRecoveryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidProtectedBoundaryComplex { error } => {
                write!(
                    formatter,
                    "Tetrahedron recovery requires a validated PLC: {error}"
                )
            }
            Self::EmptyTetrahedronMesh => write!(
                formatter,
                "Tetrahedron recovery requires a non-empty Tetrahedron mesh"
            ),
            Self::IncompleteRecovery {
                missing_item_count,
                missing_source_face_item_count,
                missing_source_edge_item_count,
                missing_material_interface_item_count,
            } => write!(
                formatter,
                "Tetrahedron recovery is incomplete: {missing_item_count} missing constraints ({missing_source_face_item_count} source faces, {missing_source_edge_item_count} source edges, {missing_material_interface_item_count} material interfaces)"
            ),
        }
    }
}

impl std::error::Error for TetrahedronRecoveryError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetrahedronRecoveryQueue {
    #[serde(default)]
    pub items: Vec<TetrahedronRecoveryQueueItem>,
    pub evidence: StageEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetrahedronRecoveryResult {
    pub tetrahedron_mesh: TetrahedronMesh,
    pub recovery_queue: TetrahedronRecoveryQueue,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetrahedronRecoveryQueueItem {
    pub item_id: String,
    pub kind: TetrahedronRecoveryKind,
    pub status: TetrahedronRecoveryStatus,
    #[serde(default)]
    pub source_entity_id: Option<TopologyEntityId>,
    #[serde(default)]
    pub protected_edge_node_ids: Option<[TopologyEntityId; 2]>,
    #[serde(default)]
    pub material_interface_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TetrahedronRecoveryKind {
    SourceFace,
    SourceEdge,
    MaterialInterface,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TetrahedronRecoveryStatus {
    Recovered,
    Missing,
}
