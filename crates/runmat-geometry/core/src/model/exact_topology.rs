use serde::{Deserialize, Serialize};

use super::{
    CurveEvaluatorId, MassPropertiesEvaluatorId, PcurveEvaluatorId, PersistentEntityId,
    SurfaceEvaluatorId, TrimClassifierId,
};

pub const EXACT_BREP_TOPOLOGY_SCHEMA_VERSION: u16 = 2;
pub const EXACT_CONTACT_PAIRING_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TopologicalOrientation {
    Forward,
    Reversed,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GeometryTransform(pub [f64; 16]);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactBRepTopology {
    pub schema_version: u16,
    pub root_assembly_id: PersistentEntityId,
    pub assemblies: Vec<ExactAssembly>,
    pub instances: Vec<ExactInstance>,
    pub bodies: Vec<ExactBody>,
    pub lumps: Vec<ExactLump>,
    pub solids: Vec<ExactSolid>,
    pub regions: Vec<ExactRegion>,
    pub shells: Vec<ExactShell>,
    pub faces: Vec<ExactFace>,
    pub wires: Vec<ExactWire>,
    pub coedges: Vec<ExactCoedge>,
    pub edges: Vec<ExactEdge>,
    pub vertices: Vec<ExactVertex>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interfaces: Vec<ExactSharedInterface>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contacts: Vec<ExactContactPair>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactAssembly {
    pub id: PersistentEntityId,
    /// Content identity of the imported part/assembly definition. Multiple
    /// occurrence nodes may bind the same definition without sharing semantic IDs.
    pub definition_digest: [u8; 32],
    pub body_ids: Vec<PersistentEntityId>,
    pub child_instance_ids: Vec<PersistentEntityId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactInstance {
    pub id: PersistentEntityId,
    pub parent_assembly_id: PersistentEntityId,
    pub instantiated_assembly_id: PersistentEntityId,
    pub transform: GeometryTransform,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactBody {
    pub id: PersistentEntityId,
    pub mass_properties_evaluator_id: MassPropertiesEvaluatorId,
    pub lump_ids: Vec<PersistentEntityId>,
    pub is_sheet_body: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sheet_shell_ids: Vec<PersistentEntityId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactLump {
    pub id: PersistentEntityId,
    pub solid_ids: Vec<PersistentEntityId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactSolid {
    pub id: PersistentEntityId,
    pub outer_shell_id: PersistentEntityId,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub void_shell_ids: Vec<PersistentEntityId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactRegion {
    pub id: PersistentEntityId,
    pub solid_id: PersistentEntityId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactShell {
    pub id: PersistentEntityId,
    pub orientation: TopologicalOrientation,
    pub face_uses: Vec<OrientedEntityUse>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactFace {
    pub id: PersistentEntityId,
    pub orientation: TopologicalOrientation,
    pub surface_evaluator_id: SurfaceEvaluatorId,
    pub trim_classifier_id: TrimClassifierId,
    pub outer_wire_id: PersistentEntityId,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inner_wire_ids: Vec<PersistentEntityId>,
    pub periodic_u: bool,
    pub periodic_v: bool,
    pub has_singularity: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactWire {
    pub id: PersistentEntityId,
    pub orientation: TopologicalOrientation,
    pub coedge_ids: Vec<PersistentEntityId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactCoedge {
    pub id: PersistentEntityId,
    pub face_id: PersistentEntityId,
    pub edge_id: PersistentEntityId,
    /// Direction of the edge in the face-local ordered wire traversal. This already includes the
    /// wire traversal direction and must not be composed with `ExactWire::orientation` again.
    pub orientation: TopologicalOrientation,
    pub pcurve_evaluator_id: PcurveEvaluatorId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seam_image: Option<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactEdge {
    pub id: PersistentEntityId,
    pub curve_evaluator_id: CurveEvaluatorId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_vertex_id: Option<PersistentEntityId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_vertex_id: Option<PersistentEntityId>,
    pub is_closed: bool,
    pub is_periodic: bool,
    pub is_degenerate: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactVertex {
    pub id: PersistentEntityId,
    pub point_m: [f64; 3],
    pub tolerance_m: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrientedEntityUse {
    pub entity_id: PersistentEntityId,
    pub orientation: TopologicalOrientation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactSharedInterface {
    pub face_id: PersistentEntityId,
    pub side_a_region_id: PersistentEntityId,
    pub side_b_region_id: PersistentEntityId,
    pub side_a_orientation: TopologicalOrientation,
    pub side_b_orientation: TopologicalOrientation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactContactPair {
    pub id: PersistentEntityId,
    pub side_a_face_ids: Vec<PersistentEntityId>,
    pub side_b_face_ids: Vec<PersistentEntityId>,
    /// Version of the canonical exact source-face pairing identity below.
    pub pairing_schema_version: u16,
    /// Domain-separated digest of both canonical persistent face-ID sets.
    pub pairing_contract_digest: [u8; 32],
}
