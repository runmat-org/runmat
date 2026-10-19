use serde::{Deserialize, Serialize};

use super::{ElementOrder, GeometryRevisionRef, MeshingRequest, PersistentEntityId, StableDigest};

pub const SOLVER_MESH_ARTIFACT_SCHEMA_VERSION: u16 = 8;
/// Tet10 midside-node order after the four corners: 01, 12, 20, 03, 13, 23.
pub const TETRAHEDRON_MIDSIDE_EDGE_CORNERS: [[usize; 2]; 6] =
    [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryEdgeOrder {
    Line2,
    Line3,
}

impl BoundaryEdgeOrder {
    pub(super) const fn node_count(self) -> usize {
        match self {
            Self::Line2 => 2,
            Self::Line3 => 3,
        }
    }
}

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
    ConformalInterface,
    ContactPrimary,
    ContactSecondary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverMeshNode {
    pub node_id: u64,
    pub stable_identity: StableDigest,
    pub coordinates_m: [f64; 3],
    pub provenance: Vec<PersistentEntityId>,
    pub exact_parameters: Vec<SolverNodeExactParameter>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SolverNodeExactParameter {
    Curve {
        source_edge_id: PersistentEntityId,
        parameter: f64,
    },
    Surface {
        source_face_id: PersistentEntityId,
        chart_id: StableDigest,
        evaluator_uv: [f64; 2],
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverVolumeElement {
    pub element_id: u64,
    pub stable_identity: StableDigest,
    pub order: ElementOrder,
    pub node_ids: Vec<u64>,
    pub region_id: PersistentEntityId,
    pub provenance: Vec<PersistentEntityId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverBoundaryFace {
    pub face_id: u64,
    pub stable_identity: StableDigest,
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
    pub stable_identity: StableDigest,
    pub order: BoundaryEdgeOrder,
    pub node_ids: Vec<u64>,
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
    pub element_ids: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConformalInterface {
    /// The authoritative exact face shared by both conformal regions.
    pub source_face_id: PersistentEntityId,
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
    pub conformal_interfaces: Vec<ConformalInterface>,
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
