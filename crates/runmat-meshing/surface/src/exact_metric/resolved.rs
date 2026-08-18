use std::collections::BTreeMap;

use runmat_geometry_core::{
    ExactBRepTopology, ExactSurfaceEvaluator, GeometryEvaluationControl, GeometryTransform,
    PersistentEntityId, SurfaceEvaluatorId,
};
use runmat_meshing_core::{MetricContributionScope, MetricFieldRequest};
use runmat_meshing_size::{
    grading::grade_metric_evaluations,
    incidence::TopologyMetricIncidence,
    metric::{MetricTensor3, ResolvedMetricEvaluation, ResolvedMetricField},
};

use super::{
    ExactFaceMetricError, ExactFaceMetricErrorKind, ExactFaceMetricEvaluation,
    ParametricMetricTensor,
};
use crate::ExactFaceChartParameterization;

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedFaceMetricField {
    by_face: BTreeMap<PersistentEntityId, FaceMetricRecord>,
}

#[derive(Clone, Debug, PartialEq)]
struct FaceMetricRecord {
    surface_evaluator_id: SurfaceEvaluatorId,
    world_transform: GeometryTransform,
    resolved: ResolvedMetricEvaluation,
}

impl ResolvedFaceMetricField {
    pub fn new(
        topology: &ExactBRepTopology,
        request: &MetricFieldRequest,
    ) -> Result<Self, ExactFaceMetricError> {
        request
            .validate()
            .map_err(|error| invalid_request(error.to_string()))?;
        let resolver = ResolvedMetricField::new(request)
            .map_err(|error| invalid_request(error.to_string()))?;
        let incidence = TopologyMetricIncidence::new(topology);
        for contribution in &request.contributions {
            let entity_id = match &contribution.scope {
                MetricContributionScope::Region { region_id } => region_id,
                MetricContributionScope::Entity { entity_id } => entity_id,
            };
            if !incidence.knows(entity_id) {
                return Err(invalid_request(format!(
                    "metric contribution references unknown exact entity {entity_id:?}"
                )));
            }
        }

        let mut resolved_by_face = BTreeMap::new();
        for face in &topology.faces {
            let incident = incidence.incident_face_entities(face);
            let resolved = resolver.resolve(&incident).map_err(|error| {
                ExactFaceMetricError::new(
                    ExactFaceMetricErrorKind::InvalidRequest,
                    Some(&face.id),
                    error.to_string(),
                )
            })?;
            if resolved_by_face.insert(face.id.clone(), resolved).is_some() {
                return Err(ExactFaceMetricError::new(
                    ExactFaceMetricErrorKind::InvalidRequest,
                    Some(&face.id),
                    "exact topology contains a duplicate face identity",
                ));
            }
        }
        grade_metric_evaluations(
            request.maximum_grading_ratio,
            &TopologyMetricIncidence::face_adjacency(topology),
            &mut resolved_by_face,
        )
        .map_err(|error| invalid_request(error.to_string()))?;

        let mut by_face = BTreeMap::new();
        for face in &topology.faces {
            let world_transform = topology.world_transform_for(&face.id).map_err(|error| {
                ExactFaceMetricError::new(
                    ExactFaceMetricErrorKind::InvalidRequest,
                    Some(&face.id),
                    error.to_string(),
                )
            })?;
            by_face.insert(
                face.id.clone(),
                FaceMetricRecord {
                    surface_evaluator_id: face.surface_evaluator_id.clone(),
                    world_transform,
                    resolved: resolved_by_face
                        .remove(&face.id)
                        .expect("resolved every exact face"),
                },
            );
        }
        Ok(Self { by_face })
    }

    pub fn evaluate(
        &self,
        source_face_id: &PersistentEntityId,
        uv: [f64; 2],
        evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
        control: &dyn GeometryEvaluationControl,
    ) -> Result<ExactFaceMetricEvaluation, ExactFaceMetricError> {
        self.evaluate_parameterized(
            source_face_id,
            uv,
            &ExactFaceChartParameterization::EvaluatorParameters,
            evaluator,
            control,
        )
    }

    pub fn evaluate_parameterized(
        &self,
        source_face_id: &PersistentEntityId,
        uv: [f64; 2],
        parameterization: &ExactFaceChartParameterization,
        evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
        control: &dyn GeometryEvaluationControl,
    ) -> Result<ExactFaceMetricEvaluation, ExactFaceMetricError> {
        if uv.iter().any(|value| !value.is_finite()) {
            return Err(ExactFaceMetricError::new(
                ExactFaceMetricErrorKind::InvalidEvaluation,
                Some(source_face_id),
                "face metric UV must be finite",
            ));
        }
        let record = self.by_face.get(source_face_id).ok_or_else(|| {
            ExactFaceMetricError::new(
                ExactFaceMetricErrorKind::UnknownFace,
                Some(source_face_id),
                "face is absent from the resolved exact metric field",
            )
        })?;
        parameterization.validate().map_err(|reason| {
            ExactFaceMetricError::new(
                ExactFaceMetricErrorKind::InvalidEvaluation,
                Some(source_face_id),
                reason,
            )
        })?;
        let sample = parameterized_sample(
            parameterization,
            evaluator,
            &record.surface_evaluator_id,
            source_face_id,
            uv,
            control,
        )?;
        let evaluator_uv = sample.evaluator_uv;
        let point_m = record.world_transform.transform_point(sample.point_m);
        let derivative_u_m = record
            .world_transform
            .transform_vector(sample.derivatives[0]);
        let derivative_v_m = record
            .world_transform
            .transform_vector(sample.derivatives[1]);
        if point_m
            .iter()
            .chain(&derivative_u_m)
            .chain(&derivative_v_m)
            .any(|value| !value.is_finite())
        {
            return Err(ExactFaceMetricError::new(
                ExactFaceMetricErrorKind::InvalidEvaluation,
                Some(source_face_id),
                "transformed surface derivatives are not finite",
            ));
        }
        let physical_metric = pullback_identity(derivative_u_m, derivative_v_m);
        let sizing_metric = pullback_metric(record.resolved.metric, derivative_u_m, derivative_v_m);
        physical_metric.validate().map_err(|reason| {
            ExactFaceMetricError::new(
                ExactFaceMetricErrorKind::InvalidEvaluation,
                Some(source_face_id),
                format!("invalid exact first fundamental form: {reason}"),
            )
        })?;
        sizing_metric.validate().map_err(|reason| {
            ExactFaceMetricError::new(
                ExactFaceMetricErrorKind::InvalidEvaluation,
                Some(source_face_id),
                format!("invalid pulled-back sizing metric: {reason}"),
            )
        })?;

        Ok(ExactFaceMetricEvaluation {
            source_face_id: source_face_id.clone(),
            uv,
            evaluator_uv,
            point_m,
            derivative_u_m,
            derivative_v_m,
            physical_metric,
            sizing_metric,
            active_sources: record.resolved.active_sources.clone(),
            applied_contribution_count: record.resolved.applied_contribution_count,
            clipped_contribution_count: record.resolved.clipped_contribution_count,
            rejected_contribution_count: record.resolved.rejected_contribution_count,
        })
    }
}

struct ParameterizedSample {
    evaluator_uv: [f64; 2],
    point_m: [f64; 3],
    derivatives: [[f64; 3]; 2],
}

fn parameterized_sample(
    parameterization: &ExactFaceChartParameterization,
    evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
    evaluator_id: &SurfaceEvaluatorId,
    source_face_id: &PersistentEntityId,
    coordinates: [f64; 2],
    control: &dyn GeometryEvaluationControl,
) -> Result<ParameterizedSample, ExactFaceMetricError> {
    match parameterization {
        ExactFaceChartParameterization::EvaluatorParameters => {
            let evaluator_uv =
                evaluator_parameters(evaluator, evaluator_id, source_face_id, coordinates)?;
            let derivatives = evaluator
                .derivatives(evaluator_id, evaluator_uv, control)
                .map_err(|error| geometry_error(source_face_id, error))?;
            Ok(ParameterizedSample {
                evaluator_uv,
                point_m: derivatives.point_m,
                derivatives: [derivatives.du_m, derivatives.dv_m],
            })
        }
        ExactFaceChartParameterization::ClosestPointProjection {
            differential_step_m,
            projection_tolerance_m,
            ..
        } => {
            let project = |coordinates| {
                let query = parameterization
                    .chart_plane_point(coordinates)
                    .ok_or_else(|| {
                        invalid_evaluation(source_face_id, "projected chart has no plane point")
                    })?;
                evaluator
                    .closest_point(evaluator_id, query, *projection_tolerance_m, control)
                    .map_err(|error| geometry_error(source_face_id, error))
            };
            let center = project(coordinates)?;
            let samples = [0, 1]
                .map(|axis| {
                    let mut before = coordinates;
                    let mut after = coordinates;
                    before[axis] -= differential_step_m;
                    after[axis] += differential_step_m;
                    let before = project(before)?;
                    let after = project(after)?;
                    Ok(std::array::from_fn(|component| {
                        (after.point_m[component] - before.point_m[component])
                            / (2.0 * differential_step_m)
                    }))
                })
                .into_iter()
                .collect::<Result<Vec<[f64; 3]>, ExactFaceMetricError>>()?;
            if center
                .uv
                .iter()
                .chain(&center.point_m)
                .chain(samples.iter().flatten())
                .any(|value| !value.is_finite())
            {
                return Err(invalid_evaluation(
                    source_face_id,
                    "projected chart evaluation is not finite",
                ));
            }
            Ok(ParameterizedSample {
                evaluator_uv: center.uv,
                point_m: center.point_m,
                derivatives: [samples[0], samples[1]],
            })
        }
    }
}

/// Maps chart-local periodic images into the evaluator's admitted parameter
/// bounds without changing the authoritative chart coordinates retained in
/// metric evidence.
pub(super) fn evaluator_parameters(
    evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
    evaluator_id: &SurfaceEvaluatorId,
    source_face_id: &PersistentEntityId,
    uv: [f64; 2],
) -> Result<[f64; 2], ExactFaceMetricError> {
    let bounds = evaluator
        .parameter_bounds(evaluator_id)
        .map_err(|error| geometry_error(source_face_id, error))?;
    let periodicity = evaluator
        .periodicity(evaluator_id)
        .map_err(|error| geometry_error(source_face_id, error))?;
    let mut result = uv;
    for axis in 0..2 {
        let bound = bounds[axis];
        if !bound.start.is_finite() || !bound.end.is_finite() || bound.start >= bound.end {
            return Err(invalid_evaluation(
                source_face_id,
                "surface evaluator returned invalid parameter bounds",
            ));
        }
        let Some(period) = periodicity[axis] else {
            continue;
        };
        if !period.is_finite() || period <= 0.0 || period > bound.end - bound.start {
            return Err(invalid_evaluation(
                source_face_id,
                "surface evaluator returned periodicity inconsistent with its bounds",
            ));
        }
        if result[axis] < bound.start || result[axis] > bound.end {
            result[axis] = bound.start + (result[axis] - bound.start).rem_euclid(period);
        }
    }
    Ok(result)
}

fn pullback_identity(first: [f64; 3], second: [f64; 3]) -> ParametricMetricTensor {
    ParametricMetricTensor {
        uu: dot(first, first),
        uv: dot(first, second),
        vv: dot(second, second),
    }
}

fn pullback_metric(
    metric: MetricTensor3,
    first: [f64; 3],
    second: [f64; 3],
) -> ParametricMetricTensor {
    ParametricMetricTensor {
        uu: metric_dot(first, metric, first),
        uv: metric_dot(first, metric, second),
        vv: metric_dot(second, metric, second),
    }
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn metric_dot(left: [f64; 3], metric: MetricTensor3, right: [f64; 3]) -> f64 {
    left[0] * (metric.xx * right[0] + metric.xy * right[1] + metric.xz * right[2])
        + left[1] * (metric.xy * right[0] + metric.yy * right[1] + metric.yz * right[2])
        + left[2] * (metric.xz * right[0] + metric.yz * right[1] + metric.zz * right[2])
}

fn invalid_request(reason: impl Into<String>) -> ExactFaceMetricError {
    ExactFaceMetricError::new(ExactFaceMetricErrorKind::InvalidRequest, None, reason)
}

fn invalid_evaluation(
    source_face_id: &PersistentEntityId,
    reason: impl Into<String>,
) -> ExactFaceMetricError {
    ExactFaceMetricError::new(
        ExactFaceMetricErrorKind::InvalidEvaluation,
        Some(source_face_id),
        reason,
    )
}

fn geometry_error(
    source_face_id: &PersistentEntityId,
    error: runmat_geometry_core::GeometryEvaluationError,
) -> ExactFaceMetricError {
    ExactFaceMetricError::new(
        ExactFaceMetricErrorKind::GeometryEvaluation(error.kind),
        Some(source_face_id),
        error.reason,
    )
}
