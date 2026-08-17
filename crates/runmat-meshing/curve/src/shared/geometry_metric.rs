use runmat_geometry_core::{
    ExactBRepTopology, ExactCurveEvaluator, GeometryEvaluationControl, GeometryEvaluationError,
};
use runmat_meshing_core::{
    CurveQualityTargets, MetricContribution, MetricContributionScope, MetricFieldRequest,
    MetricSourceKind, MetricTensor3,
};

use super::{SharedCurveError, SharedCurveErrorKind};

const CURVATURE_SAMPLES_PER_EDGE: u32 = 9;

/// Adds deterministic exact-curve curvature constraints to the resolved metric request.
/// Constructive chord and tangent validation remains authoritative; this source makes the same
/// geometry demand visible to curve, surface, and later volume metric consumers.
pub fn derive_curve_geometry_metric(
    topology: &ExactBRepTopology,
    evaluator: &(impl ExactCurveEvaluator + ?Sized),
    control: &dyn GeometryEvaluationControl,
    request: &MetricFieldRequest,
    quality: CurveQualityTargets,
) -> Result<MetricFieldRequest, SharedCurveError> {
    let mut contributions = Vec::new();
    for edge in &topology.edges {
        if edge.is_degenerate {
            continue;
        }
        let range = evaluator
            .parameter_range(&edge.curve_evaluator_id)
            .map_err(|error| geometry_error(edge, error))?;
        let transform = topology.world_transform_for(&edge.id).map_err(|error| {
            SharedCurveError::invalid_request(
                "curve metric occurrence transform",
                error.to_string(),
            )
            .for_edge(&edge.id)
        })?;
        let mut maximum_curvature = 0.0_f64;
        for sample in 0..CURVATURE_SAMPLES_PER_EDGE {
            control
                .consume_iterations(1)
                .map_err(|error| geometry_error(edge, error))?;
            let fraction = f64::from(sample) / f64::from(CURVATURE_SAMPLES_PER_EDGE - 1);
            let parameter = range.start + (range.end - range.start) * fraction;
            let derivatives = evaluator
                .derivatives(&edge.curve_evaluator_id, parameter, control)
                .map_err(|error| geometry_error(edge, error))?;
            let first = transform.transform_vector(derivatives.first_m);
            let second = transform.transform_vector(derivatives.second_m);
            let speed = norm(first);
            let curvature = norm(cross(first, second)) / speed.powi(3);
            if !curvature.is_finite() || speed <= 0.0 {
                return Err(SharedCurveError::new(
                    SharedCurveErrorKind::GeometryEvaluation(
                        runmat_geometry_core::GeometryEvaluationErrorKind::InvalidResult,
                    ),
                    "curve curvature metric",
                    "transformed derivatives do not define finite regular curvature",
                )
                .for_edge(&edge.id));
            }
            maximum_curvature = maximum_curvature.max(curvature);
        }
        if maximum_curvature == 0.0 {
            continue;
        }
        let target_size_m = curvature_target_size(maximum_curvature, quality).ok_or_else(|| {
            SharedCurveError::invalid_request(
                "curve curvature metric",
                "quality targets do not produce a finite positive curvature size",
            )
            .for_edge(&edge.id)
        })?;
        contributions.push(MetricContribution {
            source: MetricSourceKind::Curve,
            scope: MetricContributionScope::Entity {
                entity_id: edge.id.clone(),
            },
            metric: MetricTensor3::isotropic_length_m(target_size_m).map_err(|error| {
                SharedCurveError::invalid_request("curve curvature metric", error.to_string())
                    .for_edge(&edge.id)
            })?,
        });
    }
    request
        .intersect_contributions(contributions)
        .map_err(|error| {
            SharedCurveError::invalid_request("curve geometry metric", error.to_string())
        })
}

fn curvature_target_size(curvature: f64, quality: CurveQualityTargets) -> Option<f64> {
    let radius = curvature.recip();
    let deviation = quality.maximum_chordal_deviation_m;
    let chord_squared = 8.0 * radius * deviation - 4.0 * deviation.powi(2);
    let chord = if deviation < radius && chord_squared > 0.0 {
        chord_squared.sqrt()
    } else {
        f64::INFINITY
    };
    let tangent = quality.maximum_tangent_change_degrees.to_radians() / curvature;
    let target = chord.min(tangent);
    (target.is_finite() && target > 0.0).then_some(target)
}

fn geometry_error(
    edge: &runmat_geometry_core::ExactEdge,
    error: GeometryEvaluationError,
) -> SharedCurveError {
    SharedCurveError::new(
        SharedCurveErrorKind::GeometryEvaluation(error.kind),
        "curve curvature metric evaluation",
        error.reason,
    )
    .for_edge(&edge.id)
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn norm(value: [f64; 3]) -> f64 {
    value
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt()
}
