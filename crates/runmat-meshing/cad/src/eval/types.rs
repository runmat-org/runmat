use serde::{Deserialize, Serialize};

use crate::math::Point3;
use runmat_geometry_core::CadFaceEvaluationSample;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CadEvaluationSource {
    ParametricCad,
    ImportedEvaluatorSamples,
    PlanarFacetApproximation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadFaceEvaluationFrame {
    pub face_id: String,
    pub source_face_id: u32,
    pub origin_m: Point3,
    pub u_axis: Point3,
    pub v_axis: Point3,
    pub unit_normal: Point3,
    pub area_m2: f64,
    #[serde(default)]
    pub evaluator_backed: bool,
    #[serde(default)]
    pub exact_query_backed: bool,
    #[serde(default)]
    pub live_query_backed: bool,
    #[serde(default)]
    pub evaluator_sample_count: usize,
    #[serde(default)]
    pub evaluator_rejected_sample_count: usize,
    #[serde(default)]
    pub evaluator_max_projection_error_m: f64,
    #[serde(default)]
    pub evaluator_samples: Vec<CadFaceEvaluationSample>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub u_derivative_m_per_uv: Option<Point3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub v_derivative_m_per_uv: Option<Point3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_curvature_estimate_1_per_m: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uv_bounds: Option<[[f64; 2]; 2]>,
    #[serde(default)]
    pub uv_bounds_sample_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uv_domain_source: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CadFaceProjection {
    pub point_m: Point3,
    pub uv: [f64; 2],
    pub distance_m: f64,
    pub unit_normal: Point3,
    #[serde(default)]
    pub uv_in_bounds: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadEvaluationReport {
    pub source: CadEvaluationSource,
    pub face_frame_count: usize,
    pub evaluator_face_count: usize,
    #[serde(default)]
    pub live_query_face_count: usize,
    #[serde(default)]
    pub exact_query_face_count: usize,
    #[serde(default)]
    pub point_evaluation_supported_face_count: usize,
    #[serde(default)]
    pub projection_supported_face_count: usize,
    #[serde(default)]
    pub normal_supported_face_count: usize,
    #[serde(default)]
    pub derivative_supported_face_count: usize,
    #[serde(default)]
    pub curvature_supported_face_count: usize,
    #[serde(default)]
    pub missing_exact_query_face_count: usize,
    #[serde(default)]
    pub missing_derivative_query_face_count: usize,
    #[serde(default)]
    pub missing_curvature_query_face_count: usize,
    #[serde(default)]
    pub evaluator_sample_count: usize,
    #[serde(default)]
    pub evaluator_rejected_sample_count: usize,
    pub normal_query_count: usize,
    pub projection_query_count: usize,
    #[serde(default)]
    pub derivative_query_count: usize,
    #[serde(default)]
    pub curvature_query_count: usize,
    pub max_projection_error_m: f64,
    pub max_normal_deviation: f64,
    #[serde(default)]
    pub uv_domain_face_count: usize,
    #[serde(default)]
    pub uv_projection_out_of_bounds_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_curvature_estimate_1_per_m: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadEvaluationModel {
    pub source_geometry_id: String,
    pub source_geometry_revision: u32,
    pub source: CadEvaluationSource,
    pub face_frames: Vec<CadFaceEvaluationFrame>,
    pub report: CadEvaluationReport,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CadEvaluationError {
    EmptyFaces,
    MissingSourceFace { source_face_id: u32 },
    MissingSourceVertex { vertex_id: u32 },
    DegenerateFace { source_face_id: u32 },
}

impl std::fmt::Display for CadEvaluationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyFaces => write!(formatter, "CAD evaluation model has no faces"),
            Self::MissingSourceFace { source_face_id } => {
                write!(formatter, "source face {source_face_id} is missing")
            }
            Self::MissingSourceVertex { vertex_id } => {
                write!(formatter, "source vertex {vertex_id} is missing")
            }
            Self::DegenerateFace { source_face_id } => {
                write!(formatter, "source face {source_face_id} is degenerate")
            }
        }
    }
}

impl std::error::Error for CadEvaluationError {}

#[derive(Debug, Clone, PartialEq)]
pub struct CadFaceEvaluationRequest<'a> {
    pub face_id: &'a str,
    pub source_face_id: u32,
    pub imported_face_id: Option<u64>,
    pub evaluator_id: Option<&'a str>,
    pub supports_point_evaluation: bool,
    pub supports_projection: bool,
    pub supports_normal: bool,
    pub supports_derivatives: bool,
    pub supports_curvature: bool,
    pub reference_point_m: Point3,
    pub reference_unit_normal: Point3,
}

pub trait CadFaceEvaluatorProvider {
    fn evaluate_face(&self, request: &CadFaceEvaluationRequest<'_>)
        -> Vec<CadFaceEvaluationSample>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NoopCadFaceEvaluatorProvider;

impl CadFaceEvaluatorProvider for NoopCadFaceEvaluatorProvider {
    fn evaluate_face(
        &self,
        _request: &CadFaceEvaluationRequest<'_>,
    ) -> Vec<CadFaceEvaluationSample> {
        Vec::new()
    }
}
