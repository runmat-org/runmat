use runmat_geometry_core::{
    surface_principal_curvature, ExactBRepTopology, ExactCurveEvaluator, ExactPcurveEvaluator,
    ExactSurfaceEvaluator, GeometryEvaluationControl, GeometryEvaluationError, SurfaceDerivatives,
};
use runmat_meshing_core::{
    CurveQualityTargets, MetricContribution, MetricContributionScope, MetricFieldRequest,
    MetricSourceKind, MetricTensor3, SurfaceQualityTargets,
};

use std::collections::BTreeMap;

use super::{SharedCurveError, SharedCurveErrorKind};

const CURVATURE_SAMPLES_PER_EDGE: u32 = 9;

/// Adds deterministic exact-curve curvature constraints to the resolved metric request.
/// Constructive chord and tangent validation remains authoritative; this source makes the same
/// geometry demand visible to curve, surface, and later volume metric consumers.
pub fn derive_curve_geometry_metric(
    topology: &ExactBRepTopology,
    evaluator: &(impl ExactCurveEvaluator + ExactPcurveEvaluator + ExactSurfaceEvaluator + ?Sized),
    control: &dyn GeometryEvaluationControl,
    request: &MetricFieldRequest,
    curve_quality: CurveQualityTargets,
    surface_quality: SurfaceQualityTargets,
) -> Result<MetricFieldRequest, SharedCurveError> {
    let mut contributions = Vec::new();
    let mut face_curvature = BTreeMap::new();
    for edge in &topology.edges {
        if edge.is_degenerate {
            continue;
        }
        let range = ExactCurveEvaluator::parameter_range(evaluator, &edge.curve_evaluator_id)
            .map_err(|error| geometry_error(edge, error))?;
        let transform = topology.world_transform_for(&edge.id).map_err(|error| {
            SharedCurveError::invalid_request(
                "curve metric occurrence transform",
                error.to_string(),
            )
            .for_edge(&edge.id)
        })?;
        let mut maximum_curvature = 0.0_f64;
        let mut minimum_witness_chord = f64::INFINITY;
        let mut previous_point = None;
        let mut is_feature = topology
            .coedges
            .iter()
            .filter(|coedge| coedge.edge_id == edge.id)
            .count()
            != 2;
        for sample in 0..CURVATURE_SAMPLES_PER_EDGE {
            control
                .consume_iterations(1)
                .map_err(|error| geometry_error(edge, error))?;
            let fraction = f64::from(sample) / f64::from(CURVATURE_SAMPLES_PER_EDGE - 1);
            let parameter = range.start + (range.end - range.start) * fraction;
            let derivatives = ExactCurveEvaluator::derivatives(
                evaluator,
                &edge.curve_evaluator_id,
                parameter,
                control,
            )
            .map_err(|error| geometry_error(edge, error))?;
            let first = transform.transform_vector(derivatives.first_m);
            let second = transform.transform_vector(derivatives.second_m);
            let point = transform.transform_point(derivatives.point_m);
            if let Some(previous) = previous_point {
                let chord = distance(previous, point);
                if chord > 0.0 {
                    minimum_witness_chord = minimum_witness_chord.min(chord);
                }
            }
            previous_point = Some(point);
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
            let normals = sample_incident_faces(
                topology,
                edge,
                evaluator,
                control,
                parameter,
                &mut face_curvature,
            )?;
            is_feature |= normals.iter().enumerate().any(|(index, left)| {
                normals[index + 1..].iter().any(|right| {
                    normal_angle_degrees(*left, *right)
                        > surface_quality.maximum_normal_deviation_degrees
                })
            });
        }
        if maximum_curvature > 0.0 {
            let target_size_m = curvature_target_size(
                maximum_curvature,
                curve_quality.maximum_chordal_deviation_m,
                curve_quality.maximum_tangent_change_degrees,
            )
            .ok_or_else(|| {
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
        if is_feature && minimum_witness_chord.is_finite() {
            contributions.push(MetricContribution {
                source: MetricSourceKind::Feature,
                scope: MetricContributionScope::Entity {
                    entity_id: edge.id.clone(),
                },
                metric: MetricTensor3::isotropic_length_m(minimum_witness_chord).map_err(
                    |error| {
                        SharedCurveError::invalid_request(
                            "exact edge feature metric",
                            error.to_string(),
                        )
                        .for_edge(&edge.id)
                    },
                )?,
            });
        }
    }
    for (face_id, maximum_curvature) in face_curvature {
        if maximum_curvature == 0.0 {
            continue;
        }
        let target_size_m = curvature_target_size(
            maximum_curvature,
            surface_quality.maximum_chordal_deviation_m,
            surface_quality.maximum_normal_deviation_degrees,
        )
        .ok_or_else(|| {
            SharedCurveError::invalid_request(
                "face-induced curve metric",
                "surface quality targets do not produce a finite positive curvature size",
            )
        })?;
        contributions.push(MetricContribution {
            source: MetricSourceKind::Face,
            scope: MetricContributionScope::Entity {
                entity_id: face_id.clone(),
            },
            metric: MetricTensor3::isotropic_length_m(target_size_m).map_err(|error| {
                SharedCurveError::invalid_request("face-induced curve metric", error.to_string())
            })?,
        });
    }
    request
        .intersect_contributions(contributions)
        .map_err(|error| {
            SharedCurveError::invalid_request("curve geometry metric", error.to_string())
        })
}

fn sample_incident_faces(
    topology: &ExactBRepTopology,
    edge: &runmat_geometry_core::ExactEdge,
    evaluator: &(impl ExactPcurveEvaluator + ExactSurfaceEvaluator + ?Sized),
    control: &dyn GeometryEvaluationControl,
    parameter: f64,
    maximum_by_face: &mut BTreeMap<runmat_geometry_core::PersistentEntityId, f64>,
) -> Result<Vec<[f64; 3]>, SharedCurveError> {
    let mut normals = Vec::new();
    for coedge in topology
        .coedges
        .iter()
        .filter(|coedge| coedge.edge_id == edge.id)
    {
        let face = topology
            .faces
            .iter()
            .find(|face| face.id == coedge.face_id)
            .expect("admitted coedge face");
        let uv =
            ExactPcurveEvaluator::point(evaluator, &coedge.pcurve_evaluator_id, parameter, control)
                .map_err(|error| geometry_error(edge, error))?;
        let derivatives =
            ExactSurfaceEvaluator::derivatives(evaluator, &face.surface_evaluator_id, uv, control)
                .map_err(|error| geometry_error(edge, error))?;
        let transform = topology.world_transform_for(&face.id).map_err(|error| {
            SharedCurveError::invalid_request("face metric occurrence transform", error.to_string())
                .for_edge(&edge.id)
        })?;
        let curvature = surface_principal_curvature(&SurfaceDerivatives {
            point_m: transform.transform_point(derivatives.point_m),
            du_m: transform.transform_vector(derivatives.du_m),
            dv_m: transform.transform_vector(derivatives.dv_m),
            duu_m: transform.transform_vector(derivatives.duu_m),
            duv_m: transform.transform_vector(derivatives.duv_m),
            dvv_m: transform.transform_vector(derivatives.dvv_m),
        })
        .map_err(|error| geometry_error(edge, error))?;
        let maximum = curvature
            .minimum_1_per_m
            .abs()
            .max(curvature.maximum_1_per_m.abs());
        maximum_by_face
            .entry(face.id.clone())
            .and_modify(|current| *current = current.max(maximum))
            .or_insert(maximum);
        normals.push(
            normalize(cross(
                transform.transform_vector(derivatives.du_m),
                transform.transform_vector(derivatives.dv_m),
            ))
            .ok_or_else(|| {
                SharedCurveError::new(
                    SharedCurveErrorKind::GeometryEvaluation(
                        runmat_geometry_core::GeometryEvaluationErrorKind::InvalidResult,
                    ),
                    "face-induced curve metric normal",
                    "transformed surface derivatives do not define a finite normal",
                )
                .for_edge(&edge.id)
            })?,
        );
    }
    Ok(normals)
}

fn curvature_target_size(
    curvature: f64,
    maximum_deviation_m: f64,
    maximum_angle_degrees: f64,
) -> Option<f64> {
    let radius = curvature.recip();
    let deviation = maximum_deviation_m;
    let chord_squared = 8.0 * radius * deviation - 4.0 * deviation.powi(2);
    let chord = if deviation < radius && chord_squared > 0.0 {
        chord_squared.sqrt()
    } else {
        f64::INFINITY
    };
    let tangent = maximum_angle_degrees.to_radians() / curvature;
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

fn normalize(value: [f64; 3]) -> Option<[f64; 3]> {
    let length = norm(value);
    (length.is_finite() && length > 0.0).then(|| value.map(|component| component / length))
}

fn normal_angle_degrees(left: [f64; 3], right: [f64; 3]) -> f64 {
    let dot = left
        .into_iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f64>()
        .abs()
        .clamp(0.0, 1.0);
    dot.acos().to_degrees()
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    norm([left[0] - right[0], left[1] - right[1], left[2] - right[2]])
}
