use serde::{Deserialize, Serialize};

use runmat_meshing_cad::CadCurveEvaluationSample;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CurveDiscretizationOptions {
    pub target_size_m: f64,
    pub min_segments_per_edge: usize,
    pub max_segments_per_edge: usize,
}

impl Default for CurveDiscretizationOptions {
    fn default() -> Self {
        Self {
            target_size_m: 0.05,
            min_segments_per_edge: 1,
            max_segments_per_edge: 256,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveNode {
    pub node_id: u32,
    pub source_edge_id: u32,
    pub parameter: f64,
    pub coordinates_m: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveElement {
    pub element_id: u32,
    pub source_edge_id: u32,
    pub node_ids: [u32; 2],
    pub length_m: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveDiscretization {
    pub nodes: Vec<CurveNode>,
    pub elements: Vec<CurveElement>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadCurveEdgeProvenance {
    pub source_edge_id: u32,
    pub cad_edge_id: String,
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
    #[serde(default)]
    pub evaluator_sample_count: usize,
    #[serde(default)]
    pub live_query_backed: bool,
    #[serde(default)]
    pub live_query_sample_count: usize,
    #[serde(default)]
    pub rejected_evaluator_sample_count: usize,
    #[serde(default)]
    pub curvature_sample_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curvature_limited_target_size_m: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadCurveDiscretization {
    pub curves: CurveDiscretization,
    #[serde(default)]
    pub edge_provenance: Vec<CadCurveEdgeProvenance>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CadCurveEvaluationRequest<'a> {
    pub cad_edge_id: &'a str,
    pub source_edge_id: u32,
    pub imported_curve_id: Option<u64>,
    pub evaluator_id: Option<&'a str>,
    pub supports_point_evaluation: bool,
    pub supports_projection: bool,
    pub supports_tangent: bool,
    pub supports_curvature: bool,
    pub parameters: &'a [f64],
}

pub trait CadCurveEvaluatorProvider {
    fn evaluate_curve(
        &self,
        request: &CadCurveEvaluationRequest<'_>,
    ) -> Vec<CadCurveEvaluationSample>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NoopCadCurveEvaluatorProvider;

impl CadCurveEvaluatorProvider for NoopCadCurveEvaluatorProvider {
    fn evaluate_curve(
        &self,
        _request: &CadCurveEvaluationRequest<'_>,
    ) -> Vec<CadCurveEvaluationSample> {
        Vec::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CurveDiscretizationError {
    InvalidTargetSize,
    InvalidSegmentBounds,
    MissingEdgeVertex { edge_id: u32, node_id: u32 },
    MissingCadEdge { source_edge_id: u32 },
}

impl std::fmt::Display for CurveDiscretizationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTargetSize => {
                write!(formatter, "curve target_size_m must be finite and positive")
            }
            Self::InvalidSegmentBounds => write!(
                formatter,
                "curve segment bounds must satisfy 1 <= min_segments_per_edge <= max_segments_per_edge"
            ),
            Self::MissingEdgeVertex { edge_id, node_id } => write!(
                formatter,
                "source edge {edge_id} references missing topology vertex {node_id}"
            ),
            Self::MissingCadEdge { source_edge_id } => {
                write!(
                    formatter,
                    "CAD topology is missing source edge {source_edge_id}"
                )
            }
        }
    }
}

impl std::error::Error for CurveDiscretizationError {}
