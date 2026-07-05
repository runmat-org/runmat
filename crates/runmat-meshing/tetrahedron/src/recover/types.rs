use serde::{Deserialize, Serialize};

use runmat_meshing_core::contracts::{
    MeshingStage, StageEvidence, TetrahedronMesh, TopologyEntityId,
};
use runmat_meshing_plc::validate::PlcValidationError;

#[derive(Debug, Clone, PartialEq)]
pub enum TetrahedronRecoveryError {
    InvalidProtectedBoundaryComplex {
        error: PlcValidationError,
    },
    EmptyTetrahedronMesh,
    TetrahedronMeshEvidenceStageMismatch {
        stage: MeshingStage,
    },
    DuplicateTetrahedronMeshNode {
        node_id: TopologyEntityId,
    },
    NonFiniteTetrahedronMeshNode {
        node_id: TopologyEntityId,
    },
    TetrahedronElementStageMismatch {
        element_id: TopologyEntityId,
    },
    DuplicateTetrahedronElement {
        element_id: TopologyEntityId,
    },
    TetrahedronElementReferencesUnknownNode {
        element_id: TopologyEntityId,
        node_id: TopologyEntityId,
    },
    TetrahedronElementHasRepeatedNode {
        element_id: TopologyEntityId,
    },
    TetrahedronElementEmptyMaterialRegion {
        element_id: TopologyEntityId,
    },
    TetrahedronBoundaryFaceStageMismatch {
        face_id: TopologyEntityId,
    },
    DuplicateTetrahedronBoundaryFace {
        face_id: TopologyEntityId,
    },
    TetrahedronBoundaryFaceReferencesUnknownNode {
        face_id: TopologyEntityId,
        node_id: TopologyEntityId,
    },
    TetrahedronBoundaryFaceHasRepeatedNode {
        face_id: TopologyEntityId,
    },
    TetrahedronBoundaryFaceSourceFaceStageMismatch {
        face_id: TopologyEntityId,
        source_face_id: TopologyEntityId,
    },
    TetrahedronBoundaryFaceSourceEdgeStageMismatch {
        face_id: TopologyEntityId,
        source_edge_id: TopologyEntityId,
    },
    IncompleteRecovery {
        missing_item_count: usize,
        missing_source_face_item_count: usize,
        missing_source_edge_item_count: usize,
        missing_material_interface_item_count: usize,
        recovery_evidence: StageEvidence,
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
            Self::TetrahedronMeshEvidenceStageMismatch { stage } => write!(
                formatter,
                "Tetrahedron recovery requires TetrahedronMesh evidence, got {stage:?}"
            ),
            Self::DuplicateTetrahedronMeshNode { node_id } => {
                write!(formatter, "Tetrahedron mesh contains duplicate node {}", node_id.id)
            }
            Self::NonFiniteTetrahedronMeshNode { node_id } => write!(
                formatter,
                "Tetrahedron mesh node {} has non-finite coordinates",
                node_id.id
            ),
            Self::TetrahedronElementStageMismatch { element_id } => write!(
                formatter,
                "Tetrahedron element {} is not a TetrahedronMesh entity",
                element_id.id
            ),
            Self::DuplicateTetrahedronElement { element_id } => write!(
                formatter,
                "Tetrahedron mesh contains duplicate element {}",
                element_id.id
            ),
            Self::TetrahedronElementReferencesUnknownNode {
                element_id,
                node_id,
            } => write!(
                formatter,
                "Tetrahedron element {} references unknown node {}",
                element_id.id, node_id.id
            ),
            Self::TetrahedronElementHasRepeatedNode { element_id } => write!(
                formatter,
                "Tetrahedron element {} has repeated nodes",
                element_id.id
            ),
            Self::TetrahedronElementEmptyMaterialRegion { element_id } => write!(
                formatter,
                "Tetrahedron element {} has empty material-region ownership",
                element_id.id
            ),
            Self::TetrahedronBoundaryFaceStageMismatch { face_id } => write!(
                formatter,
                "Tetrahedron boundary face {} is not a PLC or TetrahedronMesh entity",
                face_id.id
            ),
            Self::DuplicateTetrahedronBoundaryFace { face_id } => write!(
                formatter,
                "Tetrahedron mesh contains duplicate boundary face {}",
                face_id.id
            ),
            Self::TetrahedronBoundaryFaceReferencesUnknownNode { face_id, node_id } => write!(
                formatter,
                "Tetrahedron boundary face {} references unknown node {}",
                face_id.id, node_id.id
            ),
            Self::TetrahedronBoundaryFaceHasRepeatedNode { face_id } => write!(
                formatter,
                "Tetrahedron boundary face {} has repeated nodes",
                face_id.id
            ),
            Self::TetrahedronBoundaryFaceSourceFaceStageMismatch {
                face_id,
                source_face_id,
            } => write!(
                formatter,
                "Tetrahedron boundary face {} has non-surface source face {}",
                face_id.id, source_face_id.id
            ),
            Self::TetrahedronBoundaryFaceSourceEdgeStageMismatch {
                face_id,
                source_edge_id,
            } => write!(
                formatter,
                "Tetrahedron boundary face {} has non-curve source edge {}",
                face_id.id, source_edge_id.id
            ),
            Self::IncompleteRecovery {
                missing_item_count,
                missing_source_face_item_count,
                missing_source_edge_item_count,
                missing_material_interface_item_count,
                ..
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
    pub source_face_node_ids: Option<[TopologyEntityId; 3]>,
    #[serde(default)]
    pub source_face_topology: Option<TetrahedronSourceFaceTopology>,
    #[serde(default)]
    pub protected_edge_node_ids: Option<[TopologyEntityId; 2]>,
    #[serde(default)]
    pub protected_edge_topology: Option<TetrahedronProtectedEdgeTopology>,
    #[serde(default)]
    pub material_interface_topology: Option<TetrahedronMaterialInterfaceTopology>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TetrahedronSourceFaceTopology {
    BoundaryFace,
    VolumeFace,
    InteriorFace,
    Absent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TetrahedronProtectedEdgeTopology {
    BoundaryEdge,
    VolumeEdge,
    InteriorEdge,
    Absent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TetrahedronMaterialInterfaceTopology {
    BoundaryOwned,
    InteriorFace,
    AbsentPartition,
}
