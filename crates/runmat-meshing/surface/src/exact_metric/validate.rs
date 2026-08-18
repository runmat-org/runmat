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
    ExactFaceMetricError, ExactFaceMetricErrorKind, ExactFaceMetricEvaluation,
    ParametricMetricTensor,
};

pub fn validate_exact_face_metric_evaluation(
    evaluation: &ExactFaceMetricEvaluation,
    topology: &ExactBRepTopology,
    request: &MetricFieldRequest,
    evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
    control: &dyn GeometryEvaluationControl,
) -> Result<(), ExactFaceMetricError> {
    if evaluation.uv.iter().any(|value| !value.is_finite()) {
        return Err(invalid(
            &evaluation.source_face_id,
            "metric evaluation UV is not finite",
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
    let derivatives = ExactSurfaceEvaluator::derivatives(
        evaluator,
        &face.surface_evaluator_id,
        evaluation.uv,
        control,
    )
    .map_err(|error| {
        ExactFaceMetricError::new(
            ExactFaceMetricErrorKind::GeometryEvaluation(error.kind),
            Some(&evaluation.source_face_id),
            error.reason,
        )
    })?;
    let transform = topology.world_transform_for(&face.id).map_err(|error| {
        ExactFaceMetricError::new(
            ExactFaceMetricErrorKind::InvalidRequest,
            Some(&face.id),
            error.to_string(),
        )
    })?;
    let expected_point = transform.transform_point(derivatives.point_m);
    let expected_u = transform.transform_vector(derivatives.du_m);
    let expected_v = transform.transform_vector(derivatives.dv_m);
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
