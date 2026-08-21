use std::collections::BTreeMap;

use runmat_geometry_core::{
    ExactCurveEvaluator, ExactEdge, GeometryEvaluationControl, GeometryTransform,
};
use runmat_meshing_core::{MetricSourceKind, MetricTensor3};

use super::super::{
    discretize::{
        average_metric, edge_error, geometry_error, metric_length, normalize,
        point_segment_distance, sub, tangent_angle, CurveMetricField, CurveMetricQuery,
    },
    validation::metric_source_rank,
    CurveMetricResolutionEvidence, SharedCurve, SharedCurveError, SharedCurveErrorKind,
};

#[derive(Clone, Copy)]
struct Witness {
    point_m: [f64; 3],
    tangent: [f64; 3],
    metric: MetricTensor3,
}

pub(super) struct ValidationSampler<'a> {
    edge: &'a ExactEdge,
    curves: &'a dyn ExactCurveEvaluator,
    metric_field: &'a dyn CurveMetricField,
    control: &'a dyn GeometryEvaluationControl,
    transform: GeometryTransform,
    samples: BTreeMap<u64, Witness>,
    sources: BTreeMap<u8, MetricSourceKind>,
    minimum_target_size_m: f64,
    maximum_target_size_m: f64,
    applied_count: u32,
    clipped_count: u32,
    rejected_count: u32,
}

impl<'a> ValidationSampler<'a> {
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
            sources: BTreeMap::new(),
            minimum_target_size_m: f64::INFINITY,
            maximum_target_size_m: 0.0,
            applied_count: 0,
            clipped_count: 0,
            rejected_count: 0,
        }
    }

    pub fn sample_count(&self) -> u64 {
        self.samples.len() as u64
    }

    fn sample(&mut self, parameter: f64) -> Result<Witness, SharedCurveError> {
        if let Some(sample) = self.samples.get(&parameter.to_bits()) {
            return Ok(*sample);
        }
        self.control
            .checkpoint()
            .map_err(|error| geometry_error(self.edge, error))?;
        let point_m = self
            .curves
            .point(&self.edge.curve_evaluator_id, parameter, self.control)
            .map(|point| self.transform.transform_point(point))
            .map_err(|error| geometry_error(self.edge, error))?;
        let tangent = self
            .curves
            .unit_tangent(&self.edge.curve_evaluator_id, parameter, self.control)
            .map(|tangent| self.transform.transform_vector(tangent))
            .map_err(|error| geometry_error(self.edge, error))?;
        let tangent = normalize(tangent).ok_or_else(|| {
            mismatch(
                self.edge,
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
            .map_err(|error| error.for_edge(&self.edge.id))?;
        evaluation.metric.validate().map_err(|error| {
            edge_error(
                self.edge,
                SharedCurveErrorKind::MetricEvaluation,
                "curve metric",
                error.to_string(),
            )
        })?;
        let density = metric_length(tangent, evaluation.metric);
        if evaluation.active_sources.is_empty() || !density.is_finite() || density <= 0.0 {
            return Err(edge_error(
                self.edge,
                SharedCurveErrorKind::MetricEvaluation,
                "curve metric",
                "metric sources and tangent density must be valid",
            ));
        }
        for source in evaluation.active_sources {
            self.sources.insert(metric_source_rank(source), source);
        }
        let target = 1.0 / density;
        self.minimum_target_size_m = self.minimum_target_size_m.min(target);
        self.maximum_target_size_m = self.maximum_target_size_m.max(target);
        self.clipped_count = self
            .clipped_count
            .saturating_add(evaluation.clipped_contribution_count);
        self.applied_count = self
            .applied_count
            .saturating_add(evaluation.applied_contribution_count);
        self.rejected_count = self
            .rejected_count
            .saturating_add(evaluation.rejected_contribution_count);
        let witness = Witness {
            point_m,
            tangent,
            metric: evaluation.metric,
        };
        self.samples.insert(parameter.to_bits(), witness);
        Ok(witness)
    }

    pub fn validate_metric_evidence(&self, curve: &SharedCurve) -> Result<(), SharedCurveError> {
        let sources = self.sources.values().copied().collect::<Vec<_>>();
        let CurveMetricResolutionEvidence::Evaluated {
            active_sources,
            evaluation_count,
            applied_contribution_count,
            minimum_tangent_target_size_m,
            maximum_tangent_target_size_m,
            clipped_contribution_count,
            rejected_contribution_count,
        } = &curve.metric_resolution
        else {
            return Err(mismatch(
                self.edge,
                "curve metric resolution evidence",
                "a nondegenerate edge requires evaluated tangent metric evidence",
            ));
        };
        if *active_sources != sources
            || *evaluation_count != self.samples.len() as u64
            || *applied_contribution_count != self.applied_count
            || (*minimum_tangent_target_size_m - self.minimum_target_size_m).abs() > 1.0e-12
            || (*maximum_tangent_target_size_m - self.maximum_target_size_m).abs() > 1.0e-12
            || *clipped_contribution_count != self.clipped_count
            || *rejected_contribution_count != self.rejected_count
        {
            return Err(mismatch(
                self.edge,
                "curve metric resolution evidence",
                "recorded metric samples, sources, targets, or contribution counts are not reproducible",
            ));
        }
        Ok(())
    }
}

pub(super) fn validate_interval(
    sampler: &mut ValidationSampler<'_>,
    start: f64,
    end: f64,
) -> Result<(f64, f64, f64), SharedCurveError> {
    let span = end - start;
    let parameters = [
        start,
        start + span * 0.125,
        start + span * 0.25,
        start + span * 0.375,
        start + span * 0.5,
        start + span * 0.625,
        start + span * 0.75,
        start + span * 0.875,
        end,
    ];
    let points = parameters
        .into_iter()
        .map(|parameter| sampler.sample(parameter))
        .collect::<Result<Vec<_>, _>>()?;
    let chordal = points[1..8]
        .iter()
        .map(|point| point_segment_distance(point.point_m, points[0].point_m, points[8].point_m))
        .fold(0.0, f64::max);
    let tangent = points
        .iter()
        .enumerate()
        .flat_map(|(left_index, left)| {
            points[left_index + 1..]
                .iter()
                .map(move |right| tangent_angle(left.tangent, right.tangent))
        })
        .fold(0.0, f64::max);
    let metric = points
        .windows(2)
        .map(|pair| {
            metric_length(
                sub(pair[1].point_m, pair[0].point_m),
                average_metric(pair[0].metric, pair[1].metric),
            )
        })
        .sum();
    Ok((chordal, tangent, metric))
}

fn mismatch(
    edge: &ExactEdge,
    field: impl Into<String>,
    reason: impl Into<String>,
) -> SharedCurveError {
    edge_error(edge, SharedCurveErrorKind::GeometricMismatch, field, reason)
}
