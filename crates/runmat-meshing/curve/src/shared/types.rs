use runmat_geometry_core::{ParameterRange, PersistentEntityId, TopologicalOrientation};
use runmat_meshing_core::{MetricSourceKind, StableDigest};
use serde::{Deserialize, Serialize};

use super::SharedCurveError;

pub const SHARED_CURVE_MESH_SCHEMA_VERSION: u16 = 3;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SharedCurveMesh {
    pub schema_version: u16,
    pub edges: Vec<SharedCurve>,
}

impl SharedCurveMesh {
    pub fn validate_against(
        &self,
        topology: &runmat_geometry_core::ExactBRepTopology,
    ) -> Result<(), SharedCurveError> {
        super::validation::validate_shared_curve_mesh(self, topology)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SharedCurve {
    pub source_edge_id: PersistentEntityId,
    pub parameter_range: ParameterRange,
    pub nodes: Vec<SharedCurveNode>,
    pub face_uses: Vec<SharedCurveFaceUse>,
    pub requested: CurveResolutionPolicy,
    pub achieved: CurveResolutionEvidence,
    pub metric_resolution: CurveMetricResolutionEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SharedCurveNode {
    pub node_id: StableDigest,
    pub source_vertex_id: Option<PersistentEntityId>,
    pub parameter: f64,
    pub arc_length_m: f64,
    pub coordinates_m: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SharedCurveFaceUse {
    pub coedge_id: PersistentEntityId,
    pub face_id: PersistentEntityId,
    pub orientation: TopologicalOrientation,
    pub seam_image: Option<u8>,
    /// One face-local UV image per shared 3D node, in the edge's canonical parameter order.
    pub node_uv: Vec<[f64; 2]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurveResolutionPolicy {
    pub maximum_chordal_deviation_m: f64,
    pub maximum_tangent_change_rad: f64,
    pub minimum_metric_edge_length: f64,
    pub maximum_metric_edge_length: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurveResolutionEvidence {
    pub maximum_chordal_deviation_m: f64,
    pub maximum_tangent_change_rad: f64,
    pub minimum_metric_edge_length: f64,
    pub maximum_metric_edge_length: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CurveMetricResolutionEvidence {
    Evaluated {
        /// Canonical union of metric sources active at every constructive sample.
        active_sources: Vec<MetricSourceKind>,
        evaluation_count: u64,
        /// Evaluation-weighted count of applicable canonical request contributions.
        applied_contribution_count: u32,
        minimum_tangent_target_size_m: f64,
        maximum_tangent_target_size_m: f64,
        /// Evaluation-weighted contribution counts reported by the metric authority.
        clipped_contribution_count: u32,
        rejected_contribution_count: u32,
    },
    /// No tangent metric exists because the exact edge collapses to one 3D point.
    DegenerateTopologicalCollapse,
}
