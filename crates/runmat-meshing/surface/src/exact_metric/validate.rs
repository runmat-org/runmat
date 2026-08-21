use std::collections::BTreeMap;

use runmat_geometry_core::{
    ExactBRepTopology, ExactSurfaceEvaluator, GeometryEvaluationControl, PersistentEntityId,
};
use runmat_meshing_core::{MetricContributionScope, MetricFieldRequest};
use runmat_meshing_size::{
    grading::grade_metric_evaluations,
    incidence::TopologyMetricIncidence,
    metric::{MetricTensor3, ResolvedMetricField},
};

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
    if evaluation
        .uv
        .iter()
        .chain(&evaluation.evaluator_uv)
        .any(|value| !value.is_finite())
    {
        return Err(invalid(
            &evaluation.source_face_id,
            "metric evaluation chart or evaluator UV is not finite",
        ));
    }
    let face = topology
        .faces
        .iter()
        .find(|face| face.id == evaluation.source_face_id)
        .ok_or_else(|| {
            ExactFaceMetricError::new(
                ExactFaceMetricErrorKind::UnknownFace,
                Some(&evaluation.source_face_id),
                "metric evaluation face is absent from exact topology",
            )
        })?;
    parameterization
        .validate()
        .map_err(|reason| invalid(&evaluation.source_face_id, reason))?;
    let expected = independent_sample(
        parameterization,
        evaluator,
        &face.surface_evaluator_id,
        &evaluation.source_face_id,
        evaluation.uv,
        control,
    )?;
    if evaluation.evaluator_uv != expected.evaluator_uv {
        return Err(invalid(
            &face.id,
            "metric evaluation uses a noncanonical evaluator image",
        ));
    }
    let transform = topology.world_transform_for(&face.id).map_err(|error| {
        ExactFaceMetricError::new(
            ExactFaceMetricErrorKind::InvalidRequest,
            Some(&face.id),
            error.to_string(),
        )
    })?;
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

    let expected = resolve_face_metrics(topology, request)?
        .remove(&face.id)
        .ok_or_else(|| invalid(&face.id, "resolved metric does not contain the exact face"))?;
    let sizing_metric = independent_pullback(expected.metric, expected_u, expected_v);
    sizing_metric
        .validate()
        .map_err(|reason| invalid(&face.id, format!("invalid sizing pullback: {reason}")))?;
    if evaluation.sizing_metric != sizing_metric
        || evaluation.active_sources != expected.active_sources
        || evaluation.applied_contribution_count != expected.applied_contribution_count
        || evaluation.clipped_contribution_count != expected.clipped_contribution_count
        || evaluation.rejected_contribution_count != expected.rejected_contribution_count
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
                point_m: derivatives.point_m,
                derivatives: [derivatives.du_m, derivatives.dv_m],
            })
        }
        ExactFaceChartParameterization::ClosestPointProjection {
            origin_m,
            axes,
            differential_step_m,
            projection_tolerance_m,
        } => {
            let project = |coordinates: [f64; 2]| {
                let query = std::array::from_fn(|axis| {
                    origin_m[axis] + coordinates[0] * axes[0][axis] + coordinates[1] * axes[1][axis]
                });
                evaluator
                    .closest_point(evaluator_id, query, *projection_tolerance_m, control)
                    .map_err(|error| geometry(source_face_id, error))
            };
            let center = project(coordinates)?;
            let mut derivatives = [[0.0; 3]; 2];
            for axis in 0..2 {
                let mut before = coordinates;
                let mut after = coordinates;
                before[axis] -= differential_step_m;
                after[axis] += differential_step_m;
                let before = project(before)?;
                let after = project(after)?;
                derivatives[axis] = std::array::from_fn(|component| {
                    (after.point_m[component] - before.point_m[component])
                        / (2.0 * differential_step_m)
                });
            }
            if center
                .uv
                .iter()
                .chain(&center.point_m)
                .chain(derivatives.iter().flatten())
                .any(|value| !value.is_finite())
            {
                return Err(invalid(
                    source_face_id,
                    "projected chart evaluation is not finite",
                ));
            }
            Ok(IndependentSample {
                evaluator_uv: center.uv,
                point_m: center.point_m,
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

fn resolve_face_metrics(
    topology: &ExactBRepTopology,
    request: &MetricFieldRequest,
) -> Result<
    BTreeMap<PersistentEntityId, runmat_meshing_size::metric::ResolvedMetricEvaluation>,
    ExactFaceMetricError,
> {
    request.validate().map_err(|error| {
        ExactFaceMetricError::new(
            ExactFaceMetricErrorKind::InvalidRequest,
            None,
            error.to_string(),
        )
    })?;
    let resolver = ResolvedMetricField::new(request).map_err(|error| {
        ExactFaceMetricError::new(
            ExactFaceMetricErrorKind::InvalidRequest,
            None,
            error.to_string(),
        )
    })?;
    let incidence = TopologyMetricIncidence::new(topology);
    for contribution in &request.contributions {
        let entity = match &contribution.scope {
            MetricContributionScope::Region { region_id } => region_id,
            MetricContributionScope::Entity { entity_id } => entity_id,
        };
        if !incidence.knows(entity) {
            return Err(ExactFaceMetricError::new(
                ExactFaceMetricErrorKind::InvalidRequest,
                None,
                format!("metric contribution references unknown exact entity {entity:?}"),
            ));
        }
    }
    let mut by_face = topology
        .faces
        .iter()
        .map(|face| {
            resolver
                .resolve(&incidence.incident_face_entities(face))
                .map(|metric| (face.id.clone(), metric))
                .map_err(|error| invalid(&face.id, error.to_string()))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    grade_metric_evaluations(
        request.maximum_grading_ratio,
        &TopologyMetricIncidence::face_adjacency(topology),
        &mut by_face,
    )
    .map_err(|error| {
        ExactFaceMetricError::new(
            ExactFaceMetricErrorKind::InvalidRequest,
            None,
            error.to_string(),
        )
    })?;
    Ok(by_face)
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
