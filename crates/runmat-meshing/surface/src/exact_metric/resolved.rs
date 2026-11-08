use std::collections::BTreeMap;
use std::sync::OnceLock;

use runmat_geometry_core::{
    ExactSurfaceEvaluator, GeometryEvaluationControl, GeometryTransform, ParameterRange,
    PersistentEntityId, SurfaceEvaluatorId,
};
use runmat_meshing_size::metric::{MetricTensor3, ResolvedMetricEvaluation};

mod boundary;
mod preparation;

use super::{
    ExactFaceMetricError, ExactFaceMetricErrorKind, ExactFaceMetricEvaluation,
    ParametricMetricTensor,
};
use crate::ExactFaceChartParameterization;

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedFaceMetricField {
    by_face: BTreeMap<PersistentEntityId, FaceMetricRecord>,
    resolver: std::sync::Arc<runmat_meshing_size::metric::ResolvedMetricField>,
}

#[derive(Clone, Debug)]
struct FaceMetricRecord {
    surface_evaluator_id: SurfaceEvaluatorId,
    world_transform: GeometryTransform,
    /// Uniform part of the face sizing; entity-local sources are graded from it pointwise.
    resolved: ResolvedMetricEvaluation,
    local: Vec<runmat_meshing_size::metric::LocalMetricSource>,
    boundary_metrics: Vec<boundary::EdgeMetric>,
    provenance: std::sync::Arc<super::ExactFaceMetricProvenance>,
    evaluator_parameter_domain: OnceLock<EvaluatorParameterDomain>,
}

impl PartialEq for FaceMetricRecord {
    fn eq(&self, other: &Self) -> bool {
        self.surface_evaluator_id == other.surface_evaluator_id
            && self.world_transform == other.world_transform
            && self.resolved == other.resolved
            && self.local == other.local
            && self.boundary_metrics == other.boundary_metrics
            && self.provenance == other.provenance
    }
}

#[derive(Clone, Debug, PartialEq)]
struct EvaluatorParameterDomain {
    bounds: [ParameterRange; 2],
    periodicity: [Option<f64>; 2],
}

impl ResolvedFaceMetricField {
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
            record,
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
        let local_metric = self
            .resolver
            .evaluate_at(record.resolved.metric, &record.local, point_m)
            .map_err(|error| invalid_evaluation(source_face_id, error.to_string()))?;
        let sizing_metric = pullback_metric(local_metric, derivative_u_m, derivative_v_m);
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
            provenance: std::sync::Arc::clone(&record.provenance),
            uv,
            evaluator_uv,
            point_m,
            derivative_u_m,
            derivative_v_m,
            physical_metric,
            sizing_metric,
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
    record: &FaceMetricRecord,
    source_face_id: &PersistentEntityId,
    coordinates: [f64; 2],
    control: &dyn GeometryEvaluationControl,
) -> Result<ParameterizedSample, ExactFaceMetricError> {
    let evaluator_id = &record.surface_evaluator_id;
    match parameterization {
        ExactFaceChartParameterization::EvaluatorParameters => {
            let evaluator_uv =
                cached_evaluator_parameters(evaluator, record, source_face_id, coordinates)?;
            let derivatives = evaluator
                .derivatives(evaluator_id, evaluator_uv, control)
                .map_err(|error| geometry_error(source_face_id, error))?;
            Ok(ParameterizedSample {
                evaluator_uv,
                point_m: *derivatives.point_m.canonical_values(),
                derivatives: [
                    derivatives
                        .du_m
                        .with_canonical_storage(|values, _axes| *values),
                    derivatives
                        .dv_m
                        .with_canonical_storage(|values, _axes| *values),
                ],
            })
        }
        ExactFaceChartParameterization::AffineEvaluatorParameters {
            evaluator_units_per_chart_unit,
            ..
        } => {
            let raw_uv = parameterization
                .evaluator_parameters(coordinates)
                .map_err(|reason| invalid_evaluation(source_face_id, reason))?;
            let evaluator_uv =
                cached_evaluator_parameters(evaluator, record, source_face_id, raw_uv)?;
            let derivatives = evaluator
                .derivatives(evaluator_id, evaluator_uv, control)
                .map_err(|error| geometry_error(source_face_id, error))?;
            Ok(ParameterizedSample {
                evaluator_uv,
                point_m: *derivatives.point_m.canonical_values(),
                derivatives: [
                    derivatives
                        .du_m
                        .with_canonical_storage(|values, _axes| *values)
                        .map(|value| value * evaluator_units_per_chart_unit[0]),
                    derivatives
                        .dv_m
                        .with_canonical_storage(|values, _axes| *values)
                        .map(|value| value * evaluator_units_per_chart_unit[1]),
                ],
            })
        }
        ExactFaceChartParameterization::PolarEvaluatorParameters {
            angular_axis,
            radial_axis,
            radial_direction,
            differential_step,
            ..
        } => {
            let evaluator_uv = parameterization
                .evaluator_parameters(coordinates)
                .map_err(|reason| invalid_evaluation(source_face_id, reason))?;
            let center = evaluator
                .derivatives(evaluator_id, evaluator_uv, control)
                .map_err(|error| geometry_error(source_face_id, error))?;
            let radius_squared = coordinates[0] * coordinates[0] + coordinates[1] * coordinates[1];
            let derivatives = if radius_squared == 0.0 {
                [0, 1]
                    .map(|axis| {
                        let mut sample_coordinates = [0.0; 2];
                        sample_coordinates[axis] = *differential_step;
                        let sample_uv =
                            parameterization
                                .evaluator_parameters(sample_coordinates)
                                .map_err(|reason| invalid_evaluation(source_face_id, reason))?;
                        let sample = evaluator
                            .point(evaluator_id, sample_uv, control)
                            .map_err(|error| geometry_error(source_face_id, error))?;
                        Ok(std::array::from_fn(|component| {
                            (sample.canonical_values()[component]
                                - center.point_m.canonical_values()[component])
                                / differential_step
                        }))
                    })
                    .into_iter()
                    .collect::<Result<Vec<[f64; 3]>, ExactFaceMetricError>>()?
                    .try_into()
                    .expect("two polar chart derivative samples")
            } else {
                let radius = radius_squared.sqrt();
                let mut jacobian = [[0.0; 2]; 2];
                jacobian[*angular_axis as usize] = [
                    -coordinates[1] / radius_squared,
                    coordinates[0] / radius_squared,
                ];
                jacobian[*radial_axis as usize] = [
                    radial_direction * coordinates[0] / radius,
                    radial_direction * coordinates[1] / radius,
                ];
                [0, 1].map(|chart_axis| {
                    std::array::from_fn(|component| {
                        center.du_m.with_canonical_storage(|values, _axes| *values)[component]
                            * jacobian[0][chart_axis]
                            + center.dv_m.with_canonical_storage(|values, _axes| *values)[component]
                                * jacobian[1][chart_axis]
                    })
                })
            };
            if evaluator_uv
                .iter()
                .chain(center.point_m.canonical_values())
                .chain(derivatives.iter().flatten())
                .any(|value| !value.is_finite())
            {
                return Err(invalid_evaluation(
                    source_face_id,
                    "polar chart evaluation is not finite",
                ));
            }
            Ok(ParameterizedSample {
                evaluator_uv,
                point_m: *center.point_m.canonical_values(),
                derivatives,
            })
        }
    }
}

/// Maps chart-local periodic images into the evaluator's admitted parameter
/// bounds without changing the authoritative chart coordinates retained in
/// metric evidence.
fn cached_evaluator_parameters(
    evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
    record: &FaceMetricRecord,
    source_face_id: &PersistentEntityId,
    uv: [f64; 2],
) -> Result<[f64; 2], ExactFaceMetricError> {
    if record.evaluator_parameter_domain.get().is_none() {
        let domain = EvaluatorParameterDomain {
            bounds: evaluator
                .parameter_bounds(&record.surface_evaluator_id)
                .map_err(|error| geometry_error(source_face_id, error))?,
            periodicity: evaluator
                .periodicity(&record.surface_evaluator_id)
                .map_err(|error| geometry_error(source_face_id, error))?,
        };
        let _ = record.evaluator_parameter_domain.set(domain);
    }
    let domain = record
        .evaluator_parameter_domain
        .get()
        .expect("surface parameter domain initialized");
    map_evaluator_parameters(domain, source_face_id, uv)
}

pub(super) fn evaluator_parameters(
    evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
    evaluator_id: &SurfaceEvaluatorId,
    source_face_id: &PersistentEntityId,
    uv: [f64; 2],
) -> Result<[f64; 2], ExactFaceMetricError> {
    let domain = EvaluatorParameterDomain {
        bounds: evaluator
            .parameter_bounds(evaluator_id)
            .map_err(|error| geometry_error(source_face_id, error))?,
        periodicity: evaluator
            .periodicity(evaluator_id)
            .map_err(|error| geometry_error(source_face_id, error))?,
    };
    map_evaluator_parameters(&domain, source_face_id, uv)
}

fn map_evaluator_parameters(
    domain: &EvaluatorParameterDomain,
    source_face_id: &PersistentEntityId,
    uv: [f64; 2],
) -> Result<[f64; 2], ExactFaceMetricError> {
    let mut result = uv;
    for (axis, value) in result.iter_mut().enumerate() {
        let bound = &domain.bounds[axis];
        if !bound.start.is_finite() || !bound.end.is_finite() || bound.start >= bound.end {
            return Err(invalid_evaluation(
                source_face_id,
                "surface evaluator returned invalid parameter bounds",
            ));
        }
        let Some(period) = domain.periodicity[axis] else {
            continue;
        };
        if !period.is_finite() || period <= 0.0 {
            return Err(invalid_evaluation(
                source_face_id,
                "surface evaluator returned periodicity inconsistent with its bounds",
            ));
        }
        // A chart coordinate within roundoff of the trimmed bounds is on them; wrapping it by a
        // period would carry it to the far side of a partial-period face.
        let roundoff = f64::EPSILON.sqrt() * bound.start.abs().max(bound.end.abs()).max(1.0);
        if *value < bound.start && bound.start - *value <= roundoff {
            *value = bound.start;
        } else if *value > bound.end && *value - bound.end <= roundoff {
            *value = bound.end;
        }
        if *value < bound.start || *value > bound.end {
            let original = *value;
            *value = bound.start + (*value - bound.start).rem_euclid(period);
            let tolerance = f64::EPSILON.sqrt()
                * bound
                    .start
                    .abs()
                    .max(bound.end.abs())
                    .max(value.abs())
                    .max(1.0);
            if *value > bound.end + tolerance {
                return Err(invalid_evaluation(
                    source_face_id,
                    format!(
                        "periodic chart coordinate {original} on axis {axis} (period {period}) has no equivalent inside the trimmed face bounds [{}, {}]",
                        bound.start, bound.end
                    ),
                ));
            }
            *value = value.min(bound.end);
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

#[cfg(test)]
mod periodic_bounds_tests {
    use super::*;
    use runmat_geometry_core::PersistentEntityKind;

    // witness: FEA-MESH-DISCRETIZATION-001/surface_discretization @ shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial; boundary: periodic chart coordinates within roundoff of partial-period trimmed bounds (hv-electrode rounded quarter torus)
    #[test]
    fn roundoff_outside_partial_period_bounds_lands_on_the_bound() {
        let face = PersistentEntityId::new(PersistentEntityKind::Face, "face".into(), Vec::new());
        let start = 4.712388980384587;
        let end = std::f64::consts::TAU;
        let domain = EvaluatorParameterDomain {
            bounds: [
                ParameterRange { start, end },
                ParameterRange {
                    start: 0.0,
                    end: 1.0,
                },
            ],
            periodicity: [Some(std::f64::consts::TAU), None],
        };
        let mapped = map_evaluator_parameters(&domain, &face, [4.712388980384277, 0.5]).unwrap();
        assert_eq!(mapped, [start, 0.5]);
        assert!(map_evaluator_parameters(&domain, &face, [3.0, 0.5]).is_err());
    }
}
