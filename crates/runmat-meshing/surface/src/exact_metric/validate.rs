use runmat_geometry_core::{
    ExactBRepTopology, ExactSurfaceEvaluator, GeometryEvaluationControl, PersistentEntityId,
};
use runmat_meshing_core::MetricFieldRequest;
use runmat_meshing_size::metric::MetricTensor3;

use super::{
    resolved::evaluator_parameters, ExactFaceMetricError, ExactFaceMetricErrorKind,
    ExactFaceMetricEvaluation, ParametricMetricTensor,
};
use crate::ExactFaceChartParameterization;

pub fn validate_exact_face_metric_evaluation(
    evaluation: &ExactFaceMetricEvaluation,
    topology: &ExactBRepTopology,
    request: &MetricFieldRequest,
    evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
    control: &dyn GeometryEvaluationControl,
) -> Result<(), ExactFaceMetricError> {
    validate_exact_face_metric_evaluation_in_parameterization(
        evaluation,
        topology,
        request,
        &ExactFaceChartParameterization::EvaluatorParameters,
        evaluator,
        control,
    )
}

pub fn validate_exact_face_metric_evaluation_in_parameterization(
    evaluation: &ExactFaceMetricEvaluation,
    topology: &ExactBRepTopology,
    request: &MetricFieldRequest,
    parameterization: &ExactFaceChartParameterization,
    evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
    control: &dyn GeometryEvaluationControl,
) -> Result<(), ExactFaceMetricError> {
    let prepared =
        super::prepared_validation::PreparedFaceMetricValidation::new(topology, request, control)?;
    prepared.validate(evaluation, parameterization, evaluator, control)
}

pub(super) fn validate_prepared_face_metric_evaluation(
    evaluation: &ExactFaceMetricEvaluation,
    reference: &std::collections::BTreeMap<
        PersistentEntityId,
        super::prepared_validation::FaceMetricReference<'_>,
    >,
    resolver: &runmat_meshing_size::metric::ResolvedMetricField,
    parameterization: &ExactFaceChartParameterization,
    evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
    control: &dyn GeometryEvaluationControl,
) -> Result<(), ExactFaceMetricError> {
    if evaluation
        .uv
        .iter()
        .chain(&evaluation.evaluator_uv)
        .any(|value| !value.is_finite())
    {
        return Err(invalid(
            &evaluation.provenance.source_face_id,
            "metric evaluation chart or evaluator UV is not finite",
        ));
    }
    let record = reference
        .get(&evaluation.provenance.source_face_id)
        .ok_or_else(|| {
            ExactFaceMetricError::new(
                ExactFaceMetricErrorKind::UnknownFace,
                Some(&evaluation.provenance.source_face_id),
                "metric evaluation face is absent from exact topology",
            )
        })?;
    let face = record.face;
    parameterization
        .validate()
        .map_err(|reason| invalid(&evaluation.provenance.source_face_id, reason))?;
    let expected = independent_sample(
        parameterization,
        evaluator,
        &face.surface_evaluator_id,
        &evaluation.provenance.source_face_id,
        evaluation.uv,
        control,
    )?;
    if evaluation.evaluator_uv != expected.evaluator_uv {
        return Err(invalid(
            &face.id,
            "metric evaluation uses a noncanonical evaluator image",
        ));
    }
    let transform = record.world_transform;
    let expected_point = transform.transform_point(expected.point_m);
    let expected_u = transform.transform_vector(expected.derivatives[0]);
    let expected_v = transform.transform_vector(expected.derivatives[1]);
    if evaluation.point_m != expected_point
        || evaluation.derivative_u_m != expected_u
        || evaluation.derivative_v_m != expected_v
    {
        return Err(invalid(
            &face.id,
            "metric evaluation differs from exact world-space surface derivatives",
        ));
    }

    let physical_metric = ParametricMetricTensor {
        uu: scalar_product(expected_u, expected_u),
        uv: scalar_product(expected_u, expected_v),
        vv: scalar_product(expected_v, expected_v),
    };
    physical_metric.validate().map_err(|reason| {
        invalid(
            &face.id,
            format!("invalid first fundamental form: {reason}"),
        )
    })?;
    if evaluation.physical_metric != physical_metric {
        return Err(invalid(
            &face.id,
            "reported first fundamental form is inconsistent",
        ));
    }

    let expected = &record.provenance;
    let local_metric = resolver
        .evaluate_at(record.metric.metric, &record.local, expected_point)
        .map_err(|error| invalid(&face.id, error.to_string()))?;
    let sizing_metric = independent_pullback(local_metric, expected_u, expected_v);
    sizing_metric
        .validate()
        .map_err(|reason| invalid(&face.id, format!("invalid sizing pullback: {reason}")))?;
    if evaluation.sizing_metric != sizing_metric
        || evaluation.provenance.active_sources != expected.active_sources
        || evaluation.provenance.applied_contribution_count != expected.applied_contribution_count
        || evaluation.provenance.clipped_contribution_count
            != record.metric.clipped_contribution_count
        || evaluation.provenance.rejected_contribution_count
            != record.metric.rejected_contribution_count
    {
        return Err(invalid(
            &face.id,
            "reported face sizing metric or provenance is inconsistent",
        ));
    }
    Ok(())
}

struct IndependentSample {
    evaluator_uv: [f64; 2],
    point_m: [f64; 3],
    derivatives: [[f64; 3]; 2],
}

fn independent_sample(
    parameterization: &ExactFaceChartParameterization,
    evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
    evaluator_id: &runmat_geometry_core::SurfaceEvaluatorId,
    source_face_id: &PersistentEntityId,
    coordinates: [f64; 2],
    control: &dyn GeometryEvaluationControl,
) -> Result<IndependentSample, ExactFaceMetricError> {
    match parameterization {
        ExactFaceChartParameterization::EvaluatorParameters => {
            let evaluator_uv =
                evaluator_parameters(evaluator, evaluator_id, source_face_id, coordinates)?;
            let derivatives = evaluator
                .derivatives(evaluator_id, evaluator_uv, control)
                .map_err(|error| geometry(source_face_id, error))?;
            Ok(IndependentSample {
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
                .map_err(|reason| invalid(source_face_id, reason))?;
            let evaluator_uv =
                evaluator_parameters(evaluator, evaluator_id, source_face_id, raw_uv)?;
            let derivatives = evaluator
                .derivatives(evaluator_id, evaluator_uv, control)
                .map_err(|error| geometry(source_face_id, error))?;
            Ok(IndependentSample {
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
                .map_err(|reason| invalid(source_face_id, reason))?;
            let center = evaluator
                .derivatives(evaluator_id, evaluator_uv, control)
                .map_err(|error| geometry(source_face_id, error))?;
            let radius_squared = coordinates[0] * coordinates[0] + coordinates[1] * coordinates[1];
            let derivatives = if radius_squared == 0.0 {
                let mut result = [[0.0; 3]; 2];
                for axis in 0..2 {
                    let mut sample_coordinates = [0.0; 2];
                    sample_coordinates[axis] = *differential_step;
                    let sample_uv = parameterization
                        .evaluator_parameters(sample_coordinates)
                        .map_err(|reason| invalid(source_face_id, reason))?;
                    let sample = evaluator
                        .point(evaluator_id, sample_uv, control)
                        .map_err(|error| geometry(source_face_id, error))?;
                    result[axis] = std::array::from_fn(|component| {
                        (sample.canonical_values()[component]
                            - center.point_m.canonical_values()[component])
                            / differential_step
                    });
                }
                result
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
                return Err(invalid(
                    source_face_id,
                    "polar chart evaluation is not finite",
                ));
            }
            Ok(IndependentSample {
                evaluator_uv,
                point_m: *center.point_m.canonical_values(),
                derivatives,
            })
        }
    }
}

fn geometry(
    face_id: &PersistentEntityId,
    error: runmat_geometry_core::GeometryEvaluationError,
) -> ExactFaceMetricError {
    ExactFaceMetricError::new(
        ExactFaceMetricErrorKind::GeometryEvaluation(error.kind),
        Some(face_id),
        error.reason,
    )
}

fn independent_pullback(
    metric: MetricTensor3,
    derivative_u: [f64; 3],
    derivative_v: [f64; 3],
) -> ParametricMetricTensor {
    let apply = |vector: [f64; 3]| {
        [
            metric.xx * vector[0] + metric.xy * vector[1] + metric.xz * vector[2],
            metric.xy * vector[0] + metric.yy * vector[1] + metric.yz * vector[2],
            metric.xz * vector[0] + metric.yz * vector[1] + metric.zz * vector[2],
        ]
    };
    let metric_u = apply(derivative_u);
    let metric_v = apply(derivative_v);
    ParametricMetricTensor {
        uu: scalar_product(derivative_u, metric_u),
        uv: scalar_product(derivative_u, metric_v),
        vv: scalar_product(derivative_v, metric_v),
    }
}

fn scalar_product(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter().zip(right).map(|(a, b)| a * b).sum()
}

fn invalid(face_id: &PersistentEntityId, reason: impl Into<String>) -> ExactFaceMetricError {
    ExactFaceMetricError::new(
        ExactFaceMetricErrorKind::InvalidEvaluation,
        Some(face_id),
        reason,
    )
}
