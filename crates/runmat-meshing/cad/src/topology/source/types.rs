use runmat_geometry_core::{CadCurveEvaluationSample, CadFaceEvaluationSample};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CadTopologySource {
    SemanticCad,
    GenericCadMesh,
    MeshFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CadEntityKind {
    Vertex,
    Edge,
    Loop,
    Face,
    Shell,
    Volume,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CadEntityId {
    pub kind: CadEntityKind,
    pub id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadVertex {
    pub entity_id: CadEntityId,
    pub source_vertex_id: u32,
    pub coordinates_m: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadEdge {
    pub entity_id: CadEntityId,
    pub source_edge_id: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imported_curve_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluator_id: Option<String>,
    #[serde(default)]
    pub evaluator_supports_point_evaluation: bool,
    #[serde(default)]
    pub evaluator_supports_projection: bool,
    #[serde(default)]
    pub evaluator_supports_tangent: bool,
    #[serde(default)]
    pub evaluator_supports_curvature: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evaluator_samples: Vec<CadCurveEvaluationSample>,
    pub vertex_ids: [String; 2],
    pub adjacent_face_ids: Vec<String>,
    pub length_m: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadFace {
    pub entity_id: CadEntityId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imported_face_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluator_id: Option<String>,
    #[serde(default)]
    pub evaluator_supports_point_evaluation: bool,
    #[serde(default)]
    pub evaluator_supports_projection: bool,
    #[serde(default)]
    pub evaluator_supports_normal: bool,
    #[serde(default)]
    pub evaluator_supports_derivatives: bool,
    #[serde(default)]
    pub evaluator_supports_curvature: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluator_reference_point_m: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluator_unit_normal: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evaluator_samples: Vec<CadFaceEvaluationSample>,
    pub source_face_ids: Vec<u32>,
    pub source_edge_ids: Vec<u32>,
    #[serde(default)]
    pub loop_ids: Vec<String>,
    pub loop_edge_ids: Vec<String>,
    #[serde(default)]
    pub region_ids: Vec<String>,
    #[serde(default)]
    pub material_region_ids: Vec<String>,
    pub area_m2: f64,
    pub unit_normal: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadLoop {
    pub entity_id: CadEntityId,
    pub face_id: String,
    pub edge_ids: Vec<String>,
    pub is_outer: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadShell {
    pub entity_id: CadEntityId,
    pub face_ids: Vec<String>,
    pub closed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadVolume {
    pub entity_id: CadEntityId,
    pub shell_ids: Vec<String>,
    pub region_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadTopologyReport {
    pub source: CadTopologySource,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub face_count: usize,
    pub shell_count: usize,
    pub volume_count: usize,
    pub semantic_face_count: usize,
    pub imported_face_count: usize,
    pub evaluator_face_count: usize,
    #[serde(default)]
    pub imported_curve_count: usize,
    #[serde(default)]
    pub evaluator_curve_count: usize,
    pub generic_face_count: usize,
    #[serde(default)]
    pub loop_count: usize,
    #[serde(default)]
    pub hole_loop_count: usize,
    pub closed_shell_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadTopologyModel {
    pub source_geometry_id: String,
    pub source_geometry_revision: u32,
    pub source_geometry_sha256: Option<String>,
    pub source: CadTopologySource,
    pub vertices: Vec<CadVertex>,
    pub edges: Vec<CadEdge>,
    #[serde(default)]
    pub loops: Vec<CadLoop>,
    pub faces: Vec<CadFace>,
    pub shells: Vec<CadShell>,
    pub volumes: Vec<CadVolume>,
    pub report: CadTopologyReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CadTopologyError {
    EmptyTopology,
    DuplicateEntityId {
        kind: CadEntityKind,
        id: String,
    },
    EntityKindMismatch {
        expected: CadEntityKind,
        actual: CadEntityKind,
        id: String,
    },
    MissingEntityReference {
        owner_kind: CadEntityKind,
        owner_id: String,
        reference_kind: CadEntityKind,
        reference_id: String,
    },
    LoopFaceMismatch {
        loop_id: String,
        expected_face_id: String,
        actual_face_id: String,
    },
    MissingFaceLoopReference {
        face_id: String,
        loop_id: String,
    },
    EvaluatorMetadataWithoutImportedFace {
        face_id: String,
    },
    EvaluatorMetadataWithoutImportedCurve {
        edge_id: String,
    },
    EvaluatorCapabilityWithoutEvaluator {
        face_id: String,
        capability: &'static str,
    },
    CurveEvaluatorCapabilityWithoutEvaluator {
        edge_id: String,
        capability: &'static str,
    },
    InvalidCurveEvaluatorSample {
        edge_id: String,
        sample_index: usize,
        reason: &'static str,
    },
    ReportCountMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl std::fmt::Display for CadTopologyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTopology => write!(formatter, "source topology has no vertices or faces"),
            Self::DuplicateEntityId { kind, id } => {
                write!(formatter, "duplicate CAD {kind:?} entity id {id}")
            }
            Self::EntityKindMismatch {
                expected,
                actual,
                id,
            } => write!(
                formatter,
                "CAD entity {id} has kind {actual:?}, expected {expected:?}"
            ),
            Self::MissingEntityReference {
                owner_kind,
                owner_id,
                reference_kind,
                reference_id,
            } => write!(
                formatter,
                "CAD {owner_kind:?} {owner_id} references missing {reference_kind:?} {reference_id}"
            ),
            Self::LoopFaceMismatch {
                loop_id,
                expected_face_id,
                actual_face_id,
            } => write!(
                formatter,
                "CAD loop {loop_id} is listed by face {expected_face_id} but belongs to face {actual_face_id}"
            ),
            Self::MissingFaceLoopReference { face_id, loop_id } => write!(
                formatter,
                "CAD loop {loop_id} belongs to face {face_id} but the face does not list it"
            ),
            Self::EvaluatorMetadataWithoutImportedFace { face_id } => write!(
                formatter,
                "CAD face {face_id} carries evaluator metadata without an imported face handle"
            ),
            Self::EvaluatorMetadataWithoutImportedCurve { edge_id } => write!(
                formatter,
                "CAD edge {edge_id} carries evaluator metadata without an imported curve handle"
            ),
            Self::EvaluatorCapabilityWithoutEvaluator {
                face_id,
                capability,
            } => write!(
                formatter,
                "CAD face {face_id} declares evaluator capability {capability} without an evaluator id"
            ),
            Self::CurveEvaluatorCapabilityWithoutEvaluator {
                edge_id,
                capability,
            } => write!(
                formatter,
                "CAD edge {edge_id} declares evaluator capability {capability} without an evaluator id"
            ),
            Self::InvalidCurveEvaluatorSample {
                edge_id,
                sample_index,
                reason,
            } => write!(
                formatter,
                "CAD edge {edge_id} has invalid evaluator sample {sample_index}: {reason}"
            ),
            Self::ReportCountMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "CAD topology report field {field} is {actual}, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for CadTopologyError {}
