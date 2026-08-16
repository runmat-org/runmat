use serde::{Deserialize, Serialize};

use super::{
    GeometryRevisionRef, MeshElementOrderV2, MeshingRequestV2, PersistentEntityId, StableDigest,
};

pub const ANALYSIS_MESH_ARTIFACT_SCHEMA_VERSION: u16 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryTriangleOrderV2 {
    Tri3,
    Tri6,
}

impl BoundaryTriangleOrderV2 {
    pub(super) const fn node_count(self) -> usize {
        match self {
            Self::Tri3 => 3,
            Self::Tri6 => 6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryFaceRoleV2 {
    Exterior,
    MaterialInterface,
    ContactPrimary,
    ContactSecondary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisMeshNodeV2 {
    pub node_id: u64,
    pub coordinates_m: [f64; 3],
    pub provenance: Vec<PersistentEntityId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisVolumeElementV2 {
    pub element_id: u64,
    pub order: MeshElementOrderV2,
    pub node_ids: Vec<u64>,
    pub region_id: PersistentEntityId,
    pub material_id: String,
    pub provenance: Vec<PersistentEntityId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisBoundaryFaceV2 {
    pub face_id: u64,
    pub order: BoundaryTriangleOrderV2,
    pub node_ids: Vec<u64>,
    pub adjacent_volume_element_ids: Vec<u64>,
    pub role: BoundaryFaceRoleV2,
    pub provenance: Vec<PersistentEntityId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisBoundaryEdgeV2 {
    pub edge_id: u64,
    pub node_ids: [u64; 2],
    pub adjacent_boundary_face_ids: Vec<u64>,
    pub provenance: Vec<PersistentEntityId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshNeighborV2 {
    pub element_id: u64,
    pub local_face_index: u8,
    pub adjacent_element_id: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshRegionV2 {
    pub region_id: PersistentEntityId,
    pub material_id: String,
    pub element_ids: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialInterfaceV2 {
    pub interface_id: String,
    pub side_a_region_id: PersistentEntityId,
    pub side_b_region_id: PersistentEntityId,
    pub boundary_face_ids: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContactPairV2 {
    pub contact_id: PersistentEntityId,
    pub primary_boundary_face_ids: Vec<u64>,
    pub secondary_boundary_face_ids: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldTopologyLocationV2 {
    Node,
    VolumeElement,
    BoundaryFace,
    BoundaryEdge,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldTopologyMapV2 {
    pub topology_id: String,
    pub location: FieldTopologyLocationV2,
    pub ordered_entity_ids: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisMeshTopologyV2 {
    pub nodes: Vec<AnalysisMeshNodeV2>,
    pub volume_elements: Vec<AnalysisVolumeElementV2>,
    pub neighbors: Vec<MeshNeighborV2>,
    pub boundary_faces: Vec<AnalysisBoundaryFaceV2>,
    pub boundary_edges: Vec<AnalysisBoundaryEdgeV2>,
    pub regions: Vec<MeshRegionV2>,
    pub material_interfaces: Vec<MaterialInterfaceV2>,
    pub contacts: Vec<ContactPairV2>,
    pub field_topologies: Vec<FieldTopologyMapV2>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisMeshArtifactV2 {
    pub schema_version: u16,
    pub canonical_digest: StableDigest,
    pub root_stage_manifest_digest: StableDigest,
    pub geometry: GeometryRevisionRef,
    pub resolved_request: MeshingRequestV2,
    pub topology: AnalysisMeshTopologyV2,
}
