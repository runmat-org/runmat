use runmat_geometry_core::{GeometryEvaluationErrorKind, PersistentEntityId};
use runmat_meshing_core::{MetricSourceKind, MetricTensor3};

use crate::CurveResolutionPolicy;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SharedCurveDiscretizationOptions {
    pub resolution: CurveResolutionPolicy,
    pub maximum_nodes_per_edge: u32,
    pub maximum_subdivision_depth: u16,
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
    pub clipped_contribution_count: u32,
    pub rejected_contribution_count: u32,
}

pub trait CurveMetricField: Send + Sync {
    fn evaluate(
        &self,
        query: CurveMetricQuery<'_>,
    ) -> Result<CurveMetricEvaluation, SharedCurveDiscretizationError>;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UniformCurveMetric {
    metric: MetricTensor3,
}

impl UniformCurveMetric {
    pub fn from_target_size_m(target_size_m: f64) -> Result<Self, SharedCurveDiscretizationError> {
        let metric = MetricTensor3::isotropic_length_m(target_size_m).map_err(|error| {
            SharedCurveDiscretizationError::invalid("uniform metric", error.to_string())
        })?;
        Ok(Self { metric })
    }
}

impl CurveMetricField for UniformCurveMetric {
    fn evaluate(
        &self,
        _query: CurveMetricQuery<'_>,
    ) -> Result<CurveMetricEvaluation, SharedCurveDiscretizationError> {
        Ok(CurveMetricEvaluation {
            metric: self.metric,
            active_sources: vec![MetricSourceKind::Global],
            clipped_contribution_count: 0,
            rejected_contribution_count: 0,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedCurveDiscretizationError {
    pub edge_id: Option<PersistentEntityId>,
    pub kind: SharedCurveDiscretizationErrorKind,
    pub field: String,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedCurveDiscretizationErrorKind {
    InvalidRequest,
    GeometryEvaluation(GeometryEvaluationErrorKind),
    MetricEvaluation,
    ResourceLimit,
    UnsatisfiedConstraint,
    InvalidResult,
}

impl SharedCurveDiscretizationError {
    pub(crate) fn invalid(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            edge_id: None,
            kind: SharedCurveDiscretizationErrorKind::InvalidRequest,
            field: field.into(),
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for SharedCurveDiscretizationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(edge) = &self.edge_id {
            write!(
                formatter,
                "shared curve {:?} for {:?}, invalid {}: {}",
                self.kind, edge, self.field, self.reason
            )
        } else {
            write!(
                formatter,
                "shared curve {:?}, invalid {}: {}",
                self.kind, self.field, self.reason
            )
        }
    }
}

impl std::error::Error for SharedCurveDiscretizationError {}
