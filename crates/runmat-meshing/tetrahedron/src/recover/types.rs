use serde::{Deserialize, Serialize};

use runmat_meshing_core::contracts::{StageEvidence, TopologyEntityId};
use runmat_meshing_plc::validate::PlcValidationError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TetrahedronRecoveryError {
    InvalidProtectedBoundaryComplex { error: PlcValidationError },
    EmptyTetrahedronMesh,
    MissingSourceFaceRecovery { face_id: String },
    MissingSourceEdgeRecovery { edge_id: String },
    MissingMaterialInterfaceRecovery { material_interface_id: String },
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
            Self::MissingSourceFaceRecovery { face_id } => {
                write!(formatter, "source face {face_id} is not recovered")
            }
            Self::MissingSourceEdgeRecovery { edge_id } => {
                write!(formatter, "source edge {edge_id} is not recovered")
            }
            Self::MissingMaterialInterfaceRecovery {
                material_interface_id,
            } => write!(
                formatter,
                "material interface {material_interface_id} is not recovered"
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
pub struct TetrahedronRecoveryQueueItem {
    pub item_id: String,
    pub kind: TetrahedronRecoveryKind,
    pub status: TetrahedronRecoveryStatus,
    #[serde(default)]
    pub source_entity_id: Option<TopologyEntityId>,
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
}
