use serde::{Deserialize, Serialize};

use super::{
    CurveEvaluatorIdV2, MassPropertiesEvaluatorIdV2, PcurveEvaluatorIdV2, PersistentEntityId,
    SurfaceEvaluatorIdV2, TrimClassifierIdV2,
};

pub const EXACT_BREP_TOPOLOGY_SCHEMA_VERSION: u16 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TopologicalOrientationV2 {
    Forward,
    Reversed,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GeometryTransformV2(pub [f64; 16]);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactBRepTopologyV2 {
    pub schema_version: u16,
    pub root_assembly_id: PersistentEntityId,
    pub assemblies: Vec<ExactAssemblyV2>,
    pub instances: Vec<ExactInstanceV2>,
    pub bodies: Vec<ExactBodyV2>,
    pub lumps: Vec<ExactLumpV2>,
    pub solids: Vec<ExactSolidV2>,
    pub shells: Vec<ExactShellV2>,
    pub faces: Vec<ExactFaceV2>,
    pub wires: Vec<ExactWireV2>,
    pub coedges: Vec<ExactCoedgeV2>,
    pub edges: Vec<ExactEdgeV2>,
    pub vertices: Vec<ExactVertexV2>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interfaces: Vec<ExactSharedInterfaceV2>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contacts: Vec<ExactContactPairV2>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactAssemblyV2 {
    pub id: PersistentEntityId,
    /// Content identity of the imported part/assembly definition. Multiple
    /// occurrence nodes may bind the same definition without sharing semantic IDs.
    pub definition_digest: [u8; 32],
    pub body_ids: Vec<PersistentEntityId>,
    pub child_instance_ids: Vec<PersistentEntityId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactInstanceV2 {
    pub id: PersistentEntityId,
    pub parent_assembly_id: PersistentEntityId,
    pub instantiated_assembly_id: PersistentEntityId,
    pub transform: GeometryTransformV2,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactBodyV2 {
    pub id: PersistentEntityId,
    pub mass_properties_evaluator_id: MassPropertiesEvaluatorIdV2,
    pub lump_ids: Vec<PersistentEntityId>,
    pub is_sheet_body: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sheet_shell_ids: Vec<PersistentEntityId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactLumpV2 {
    pub id: PersistentEntityId,
    pub solid_ids: Vec<PersistentEntityId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactSolidV2 {
    pub id: PersistentEntityId,
    pub outer_shell_id: PersistentEntityId,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub void_shell_ids: Vec<PersistentEntityId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactShellV2 {
    pub id: PersistentEntityId,
    pub orientation: TopologicalOrientationV2,
    pub face_uses: Vec<OrientedEntityUseV2>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactFaceV2 {
    pub id: PersistentEntityId,
    pub orientation: TopologicalOrientationV2,
    pub surface_evaluator_id: SurfaceEvaluatorIdV2,
    pub trim_classifier_id: TrimClassifierIdV2,
    pub outer_wire_id: PersistentEntityId,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inner_wire_ids: Vec<PersistentEntityId>,
    pub periodic_u: bool,
    pub periodic_v: bool,
    pub has_singularity: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactWireV2 {
    pub id: PersistentEntityId,
    pub orientation: TopologicalOrientationV2,
    pub coedge_ids: Vec<PersistentEntityId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactCoedgeV2 {
    pub id: PersistentEntityId,
    pub face_id: PersistentEntityId,
    pub edge_id: PersistentEntityId,
    pub orientation: TopologicalOrientationV2,
    pub pcurve_evaluator_id: PcurveEvaluatorIdV2,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seam_image: Option<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactEdgeV2 {
    pub id: PersistentEntityId,
    pub curve_evaluator_id: CurveEvaluatorIdV2,
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
pub struct ExactVertexV2 {
    pub id: PersistentEntityId,
    pub point_m: [f64; 3],
    pub tolerance_m: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrientedEntityUseV2 {
    pub entity_id: PersistentEntityId,
    pub orientation: TopologicalOrientationV2,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactSharedInterfaceV2 {
    pub face_id: PersistentEntityId,
    pub side_a_region_id: PersistentEntityId,
    pub side_b_region_id: PersistentEntityId,
    pub side_a_orientation: TopologicalOrientationV2,
    pub side_b_orientation: TopologicalOrientationV2,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactContactPairV2 {
    pub id: PersistentEntityId,
    pub side_a_face_ids: Vec<PersistentEntityId>,
    pub side_b_face_ids: Vec<PersistentEntityId>,
    pub pairing_contract_digest: [u8; 32],
}
