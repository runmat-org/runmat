use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{
    ExactCurveEvaluator, ExactEdge, GeometryEvaluationControl, GeometryTransform,
};
use runmat_meshing_core::{MetricSourceKind, MetricTensor3};

use crate::shared::{validation::metric_source_rank, CurveMetricResolutionEvidence};

use super::{
    error::{edge_error, geometry_error},
    math::{average_metric, metric_length, normalize, point_segment_distance, sub, tangent_angle},
    types::{
        CurveMetricField, CurveMetricQuery, SharedCurveDiscretizationError,
        SharedCurveDiscretizationErrorKind,
    },
};

#[derive(Clone, Copy)]
pub(super) struct EvaluatedPoint {
    pub parameter: f64,
    pub point_m: [f64; 3],
    tangent: [f64; 3],
    metric: MetricTensor3,
}

pub(super) struct IntervalEvidence {
    pub midpoint: EvaluatedPoint,
    pub chordal_deviation_m: f64,
    pub tangent_change_rad: f64,
    pub metric_length: f64,
}

pub(super) fn interval_evidence(
    cache: &mut EvaluationCache<'_>,
    left: EvaluatedPoint,
    right: EvaluatedPoint,
) -> Result<IntervalEvidence, SharedCurveDiscretizationError> {
    let span = right.parameter - left.parameter;
    let quarter = cache.sample(left.parameter + span * 0.25)?;
    let midpoint = cache.sample(left.parameter + span * 0.5)?;
    let three_quarter = cache.sample(left.parameter + span * 0.75)?;
    let points = [left, quarter, midpoint, three_quarter, right];
    let chordal_deviation_m = points[1..4]
        .iter()
        .map(|sample| point_segment_distance(sample.point_m, left.point_m, right.point_m))
        .fold(0.0, f64::max);
    let tangent_change_rad = points
        .iter()
        .enumerate()
        .flat_map(|(left_index, left)| {
            points[left_index + 1..]
                .iter()
                .map(move |right| tangent_angle(left.tangent, right.tangent))
        })
        .fold(0.0, f64::max);
    let metric_length = points
        .windows(2)
        .map(|pair| {
            metric_length(
                sub(pair[1].point_m, pair[0].point_m),
                average_metric(pair[0].metric, pair[1].metric),
            )
        })
        .sum();
    Ok(IntervalEvidence {
        midpoint,
        chordal_deviation_m,
        tangent_change_rad,
        metric_length,
    })
}

pub(super) struct EvaluationCache<'a> {
    pub edge: &'a ExactEdge,
    curves: &'a dyn ExactCurveEvaluator,
    metric_field: &'a dyn CurveMetricField,
    control: &'a dyn GeometryEvaluationControl,
    transform: GeometryTransform,
    samples: BTreeMap<u64, EvaluatedPoint>,
    sources: BTreeSet<u8>,
    source_by_rank: BTreeMap<u8, MetricSourceKind>,
    minimum_target_size_m: f64,
    maximum_target_size_m: f64,
    clipped_contribution_count: u32,
    rejected_contribution_count: u32,
}

impl<'a> EvaluationCache<'a> {
    pub fn new(
        edge: &'a ExactEdge,
        curves: &'a dyn ExactCurveEvaluator,
        metric_field: &'a dyn CurveMetricField,
        control: &'a dyn GeometryEvaluationControl,
        transform: GeometryTransform,
    ) -> Self {
        Self {
            edge,
            curves,
            metric_field,
            control,
            transform,
            samples: BTreeMap::new(),
            sources: BTreeSet::new(),
            source_by_rank: BTreeMap::new(),
            minimum_target_size_m: f64::INFINITY,
            maximum_target_size_m: 0.0,
            clipped_contribution_count: 0,
            rejected_contribution_count: 0,
        }
    }

    pub fn sample(
        &mut self,
        parameter: f64,
    ) -> Result<EvaluatedPoint, SharedCurveDiscretizationError> {
        if let Some(sample) = self.samples.get(&parameter.to_bits()) {
            return Ok(*sample);
        }
        self.control
            .checkpoint()
            .map_err(|error| geometry_error(self.edge, error))?;
        let local_point = self
            .curves
            .point(&self.edge.curve_evaluator_id, parameter, self.control)
            .map_err(|error| geometry_error(self.edge, error))?;
        let local_tangent = self
            .curves
            .unit_tangent(&self.edge.curve_evaluator_id, parameter, self.control)
            .map_err(|error| geometry_error(self.edge, error))?;
        let point_m = self.transform.transform_point(local_point);
        let tangent =
            normalize(self.transform.transform_vector(local_tangent)).ok_or_else(|| {
                edge_error(
                    self.edge,
                    SharedCurveDiscretizationErrorKind::InvalidResult,
                    "curve tangent",
                    "transformed exact tangent is not finite and nonzero",
                )
            })?;
        let evaluation = self
            .metric_field
            .evaluate(CurveMetricQuery {
                edge_id: &self.edge.id,
                parameter,
                point_m,
                unit_tangent: tangent,
            })
            .map_err(|mut error| {
                if error.edge_id.is_none() {
                    error.edge_id = Some(self.edge.id.clone());
                }
                error
            })?;
        evaluation.metric.validate().map_err(|error| {
            edge_error(
                self.edge,
                SharedCurveDiscretizationErrorKind::MetricEvaluation,
                "curve metric",
                error.to_string(),
            )
        })?;
        if evaluation.active_sources.is_empty() {
            return Err(edge_error(
                self.edge,
                SharedCurveDiscretizationErrorKind::MetricEvaluation,
                "curve metric sources",
                "a resolved metric must name at least one active source",
            ));
        }
        for source in evaluation.active_sources {
            let rank = metric_source_rank(source);
            self.sources.insert(rank);
            self.source_by_rank.insert(rank, source);
        }
        let density = metric_length(tangent, evaluation.metric);
        if !density.is_finite() || density <= 0.0 {
            return Err(edge_error(
                self.edge,
                SharedCurveDiscretizationErrorKind::MetricEvaluation,
                "curve tangent metric",
                "directional metric density must be finite and positive",
            ));
        }
        let target_size_m = 1.0 / density;
        self.minimum_target_size_m = self.minimum_target_size_m.min(target_size_m);
        self.maximum_target_size_m = self.maximum_target_size_m.max(target_size_m);
        self.clipped_contribution_count = self
            .clipped_contribution_count
            .saturating_add(evaluation.clipped_contribution_count);
        self.rejected_contribution_count = self
            .rejected_contribution_count
            .saturating_add(evaluation.rejected_contribution_count);
        let sample = EvaluatedPoint {
            parameter,
            point_m,
            tangent,
            metric: evaluation.metric,
        };
        self.control
            .consume_allocation_bytes(std::mem::size_of::<(u64, EvaluatedPoint)>() as u64)
            .map_err(|error| geometry_error(self.edge, error))?;
        self.samples.insert(parameter.to_bits(), sample);
        Ok(sample)
    }

    pub fn metric_evidence(&self) -> CurveMetricResolutionEvidence {
        CurveMetricResolutionEvidence {
            active_sources: self
                .sources
                .iter()
                .map(|rank| self.source_by_rank[rank])
                .collect(),
            evaluation_count: self.samples.len() as u64,
            minimum_tangent_target_size_m: self.minimum_target_size_m,
            maximum_tangent_target_size_m: self.maximum_target_size_m,
            clipped_contribution_count: self.clipped_contribution_count,
            rejected_contribution_count: self.rejected_contribution_count,
        }
    }
}
