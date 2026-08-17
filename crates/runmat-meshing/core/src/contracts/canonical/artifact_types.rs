use serde::{Deserialize, Serialize};

use super::{ElementOrder, GeometryRevisionRef, MeshingRequest, PersistentEntityId, StableDigest};

pub const ANALYSIS_MESH_ARTIFACT_SCHEMA_VERSION: u16 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryTriangleOrder {
    Tri3,
    Tri6,
}

impl BoundaryTriangleOrder {
    pub(super) const fn node_count(self) -> usize {
        match self {
            Self::Tri3 => 3,
            Self::Tri6 => 6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryFaceRole {
    Exterior,
    MaterialInterface,
    ContactPrimary,
    ContactSecondary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverMeshNode {
    pub node_id: u64,
    pub coordinates_m: [f64; 3],
    pub provenance: Vec<PersistentEntityId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverVolumeElement {
    pub element_id: u64,
    pub order: ElementOrder,
    pub node_ids: Vec<u64>,
    pub region_id: PersistentEntityId,
    pub material_id: String,
    pub provenance: Vec<PersistentEntityId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverBoundaryFace {
    pub face_id: u64,
    pub order: BoundaryTriangleOrder,
    pub node_ids: Vec<u64>,
    pub adjacent_volume_element_ids: Vec<u64>,
    pub role: BoundaryFaceRole,
    pub provenance: Vec<PersistentEntityId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverBoundaryEdge {
    pub edge_id: u64,
    pub node_ids: [u64; 2],
    pub adjacent_boundary_face_ids: Vec<u64>,
    pub provenance: Vec<PersistentEntityId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshNeighbor {
    pub element_id: u64,
    pub local_face_index: u8,
    pub adjacent_element_id: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshRegion {
    pub region_id: PersistentEntityId,
    pub material_id: String,
    pub element_ids: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialInterface {
    pub interface_id: String,
    pub side_a_region_id: PersistentEntityId,
    pub side_b_region_id: PersistentEntityId,
    pub boundary_face_ids: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContactPair {
    pub contact_id: PersistentEntityId,
    pub primary_boundary_face_ids: Vec<u64>,
    pub secondary_boundary_face_ids: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldTopologyLocation {
    Node,
    VolumeElement,
    BoundaryFace,
    BoundaryEdge,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldTopologyMap {
    pub topology_id: String,
    pub location: FieldTopologyLocation,
    pub ordered_entity_ids: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverMeshTopology {
    pub nodes: Vec<SolverMeshNode>,
    pub volume_elements: Vec<SolverVolumeElement>,
    pub neighbors: Vec<MeshNeighbor>,
    pub boundary_faces: Vec<SolverBoundaryFace>,
    pub boundary_edges: Vec<SolverBoundaryEdge>,
    pub regions: Vec<MeshRegion>,
    pub material_interfaces: Vec<MaterialInterface>,
    pub contacts: Vec<ContactPair>,
    pub field_topologies: Vec<FieldTopologyMap>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverMeshArtifact {
    pub schema_version: u16,
    pub canonical_digest: StableDigest,
    pub root_stage_manifest_digest: StableDigest,
    pub geometry: GeometryRevisionRef,
    pub resolved_request: MeshingRequest,
    pub topology: SolverMeshTopology,
}
