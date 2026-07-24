use serde::{Deserialize, Serialize};

use super::{AssemblyNode, MaterialEvidence};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceGeometryKind {
    Mesh,
    Cad,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceGeometry {
    pub kind: SourceGeometryKind,
    pub assembly: Option<AssemblyNode>,
    pub material_evidence: Vec<MaterialEvidence>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cad_evaluators: Vec<CadEvaluatorSet>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CadEvaluatorSet {
    pub evaluator_id: String,
    pub backend: String,
    pub format_name: String,
    pub requires_source_geometry: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub faces: Vec<CadFaceEvaluator>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub curves: Vec<CadCurveEvaluator>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CadFaceEvaluator {
    pub evaluator_id: String,
    pub imported_face_id: u64,
    pub name: String,
    pub supports_point_evaluation: bool,
    pub supports_projection: bool,
    pub supports_normal: bool,
    pub supports_derivatives: bool,
    pub supports_curvature: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_point_m: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_unit_normal: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evaluation_samples: Vec<CadFaceEvaluationSample>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CadCurveEvaluator {
    pub evaluator_id: String,
    pub imported_curve_id: u64,
    pub name: String,
    pub supports_point_evaluation: bool,
    pub supports_projection: bool,
    pub supports_tangent: bool,
    pub supports_curvature: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evaluation_samples: Vec<CadCurveEvaluationSample>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CadFaceEvaluationSampleSource {
    BackendQuery,
    TessellationEstimate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CadFaceEvaluationSample {
    pub source: CadFaceEvaluationSampleSource,
    pub point_m: [f64; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uv: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projected_point_m: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit_normal: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection_error_m: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CadCurveEvaluationSampleSource {
    BackendQuery,
    TessellationEstimate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CadCurveEvaluationSample {
    pub source: CadCurveEvaluationSampleSource,
    pub parameter: f64,
    pub point_m: [f64; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projected_point_m: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tangent_m: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curvature_1_per_m: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection_error_m: Option<f64>,
}
