use runmat_geometry_core::{
    ExactBRepTopology, ExactCurveEvaluator, ExactPcurveEvaluator, GeometryEvaluationControl,
    PersistentEntityId,
};
use runmat_meshing_core::{MetricSourceKind, MetricTensor3};

use crate::{shared::SharedCurveError, CurveResolutionPolicy};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SharedCurveDiscretizationOptions {
    pub resolution: CurveResolutionPolicy,
    pub maximum_nodes_per_edge: u32,
    pub maximum_subdivision_depth: u16,
    pub geometry_absolute_error_m: f64,
    pub pcurve_absolute_error: f64,
    pub arc_length_absolute_error_m: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct CurveMetricQuery<'a> {
    pub edge_id: &'a PersistentEntityId,
    pub parameter: f64,
    pub point_m: [f64; 3],
    pub unit_tangent: [f64; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct CurveMetricEvaluation {
    pub metric: MetricTensor3,
    pub active_sources: Vec<MetricSourceKind>,
    pub applied_contribution_count: u32,
    pub clipped_contribution_count: u32,
    pub rejected_contribution_count: u32,
}

pub trait CurveMetricField: Send + Sync {
    fn evaluate(
        &self,
        query: CurveMetricQuery<'_>,
    ) -> Result<CurveMetricEvaluation, SharedCurveError>;
}

#[derive(Clone, Copy)]
pub struct SharedCurveEvaluationContext<'a> {
    pub topology: &'a ExactBRepTopology,
    pub curves: &'a dyn ExactCurveEvaluator,
    pub pcurves: &'a dyn ExactPcurveEvaluator,
    pub metric_field: &'a dyn CurveMetricField,
    pub control: &'a dyn GeometryEvaluationControl,
}

impl<'a> SharedCurveEvaluationContext<'a> {
    pub fn new(
        topology: &'a ExactBRepTopology,
        curves: &'a dyn ExactCurveEvaluator,
        pcurves: &'a dyn ExactPcurveEvaluator,
        metric_field: &'a dyn CurveMetricField,
        control: &'a dyn GeometryEvaluationControl,
    ) -> Self {
        Self {
            topology,
            curves,
            pcurves,
            metric_field,
            control,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UniformCurveMetric {
    metric: MetricTensor3,
}

impl UniformCurveMetric {
    pub fn from_target_size_m(target_size_m: f64) -> Result<Self, SharedCurveError> {
        let metric = MetricTensor3::isotropic_length_m(target_size_m).map_err(|error| {
            SharedCurveError::invalid_request("uniform metric", error.to_string())
        })?;
        Ok(Self { metric })
    }
}

impl CurveMetricField for UniformCurveMetric {
    fn evaluate(
        &self,
        _query: CurveMetricQuery<'_>,
    ) -> Result<CurveMetricEvaluation, SharedCurveError> {
        Ok(CurveMetricEvaluation {
            metric: self.metric,
            active_sources: vec![MetricSourceKind::Global],
            applied_contribution_count: 0,
            clipped_contribution_count: 0,
            rejected_contribution_count: 0,
        })
    }
}
