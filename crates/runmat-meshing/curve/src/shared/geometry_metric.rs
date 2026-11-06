use runmat_geometry_core::{
    ExactBRepTopology, ExactCurveEvaluator, ExactPcurveEvaluator, ExactSurfaceEvaluator,
    GeometryEvaluationControl, GeometryEvaluationError,
};
use runmat_meshing_core::{
    CurveQualityTargets, MetricContribution, MetricContributionScope, MetricFieldRequest,
    MetricSourceKind, SurfaceQualityTargets,
};

use runmat_meshing_size::metric::DirectionalMetricEnvelope;
use std::collections::{BTreeMap, BTreeSet};

use super::{SharedCurveError, SharedCurveErrorKind};

mod feature;
mod proximity;
mod relations;
mod surface_sampling;
use relations::TopologyRelations;

use proximity::{
    derive_face_proximity_contributions, derive_proximity_contributions, ordered_pair,
    ContactProximityPairs, EdgeWitness, FaceWitness,
};

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
    let relations = TopologyRelations::new(topology, control).map_err(|error| {
        SharedCurveError::new(
            SharedCurveErrorKind::GeometryEvaluation(error.kind),
            "curve topology preparation",
            error.to_string(),
        )
    })?;
    let mut contributions = Vec::new();
    let mut face_metrics = surface_sampling::FaceMetricSamples::new(
        &topology.faces,
        control,
        request.global_metric,
        surface_quality,
    )?;
    let shell_solids = topology
        .solids
        .iter()
        .flat_map(|solid| {
            std::iter::once(&solid.outer_shell_id)
                .chain(&solid.void_shell_ids)
                .map(move |shell| (shell.clone(), solid.id.clone()))
        })
        .collect::<BTreeMap<_, _>>();
    let mut face_solids = BTreeMap::<
        runmat_geometry_core::PersistentEntityId,
        BTreeSet<runmat_geometry_core::PersistentEntityId>,
    >::new();
    for shell in &topology.shells {
        if let Some(solid) = shell_solids.get(&shell.id) {
            for face_use in &shell.face_uses {
                face_solids
                    .entry(face_use.entity_id.clone())
                    .or_default()
                    .insert(solid.clone());
            }
        }
    }
    let mut edge_witnesses = Vec::new();
    // Gaps between facing surfaces are measured only when the request authors
    // `cells_across_gap`, so default requests keep their meshes.
    let measure_face_gaps = request.cells_across_gap.is_some();
    let mut face_witnesses =
        BTreeMap::<runmat_geometry_core::PersistentEntityId, FaceWitness>::new();
    for edge in &topology.edges {
        if edge.is_degenerate {
            continue;
        }
        let range = ExactCurveEvaluator::parameter_range(evaluator, &edge.curve_evaluator_id)
            .map_err(|error| geometry_error(edge, error))?;
        let transform = relations.world_transform_for(&edge.id).map_err(|error| {
            SharedCurveError::invalid_request(
                "curve metric occurrence transform",
                error.to_string(),
            )
            .for_edge(&edge.id)
        })?;
        let mut curve_metric =
            DirectionalMetricEnvelope::new(request.global_metric).map_err(|error| {
                SharedCurveError::invalid_request("curve metric baseline", error.to_string())
            })?;
        let mut has_curvature = false;
        let mut witness_points = Vec::with_capacity(CURVATURE_SAMPLES_PER_EDGE as usize);
        let mut is_feature = relations.edge_coedges(&edge.id).len() != 2;
        for sample in 0..CURVATURE_SAMPLES_PER_EDGE {
            control
                .consume_iterations(1)
                .map_err(|error| geometry_error(edge, error))?;
            let fraction = f64::from(sample) / f64::from(CURVATURE_SAMPLES_PER_EDGE - 1);
            let parameter = range
                .parameter_at_fraction(fraction)
                .map_err(|error| geometry_error(edge, error))?;
            let derivatives = ExactCurveEvaluator::derivatives(
                evaluator,
                &edge.curve_evaluator_id,
                parameter,
                control,
            )
            .map_err(|error| geometry_error(edge, error))?;
            let derivatives = derivatives
                .transformed(&transform)
                .map_err(|error| geometry_error(edge, error))?;
            let first = derivatives
                .first_m
                .with_canonical_storage(|values, _axes| *values);
            let second = derivatives
                .second_m
                .with_canonical_storage(|values, _axes| *values);
            let point = *derivatives.point_m.canonical_values();
            witness_points.push(point);
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
            if curvature > 0.0 {
                let size = curvature_target_size(
                    curvature,
                    curve_quality.maximum_chordal_deviation_m.canonical_value(),
                    curve_quality.maximum_tangent_change_degrees,
                )
                .ok_or_else(|| {
                    SharedCurveError::invalid_request(
                        "curve curvature metric",
                        "invalid directional curvature size",
                    )
                    .for_edge(&edge.id)
                })?;
                curve_metric.include(first, size).map_err(|error| {
                    SharedCurveError::invalid_request("curve curvature metric", error.to_string())
                        .for_edge(&edge.id)
                })?;
                has_curvature = true;
            }
            let normals = face_metrics.sample(&relations, edge, evaluator, control, parameter)?;
            is_feature |= normals.iter().enumerate().any(|(index, left)| {
                normals[index + 1..].iter().any(|right| {
                    normal_angle_degrees(left.normal, right.normal)
                        > surface_quality.maximum_normal_deviation_degrees
                })
            });
            if measure_face_gaps {
                for sample in normals {
                    face_witnesses
                        .entry(sample.face_id.clone())
                        .or_insert_with(|| FaceWitness::new(sample.face_id))
                        .samples
                        .push((sample.point_m, sample.normal));
                }
            }
        }
        if has_curvature {
            contributions.push(MetricContribution {
                source: MetricSourceKind::Curve,
                scope: MetricContributionScope::Entity {
                    entity_id: edge.id.clone(),
                },
                metric: curve_metric.metric().map_err(|error| {
                    SharedCurveError::invalid_request("curve metric envelope", error.to_string())
                        .for_edge(&edge.id)
                })?,
            });
        }
        if let Some(feature_span_m) = is_feature
            .then(|| feature::witness_span_m(&witness_points))
            .flatten()
        {
            contributions.push(MetricContribution {
                source: MetricSourceKind::Feature,
                scope: MetricContributionScope::Entity {
                    entity_id: edge.id.clone(),
                },
                metric: feature::directional_metric(
                    request.global_metric,
                    &witness_points,
                    feature_span_m,
                )
                .map_err(|error| {
                    SharedCurveError::invalid_request(
                        "exact edge feature metric",
                        error.to_string(),
                    )
                    .for_edge(&edge.id)
                })?,
            });
        }
        edge_witnesses.push(EdgeWitness {
            edge_id: edge.id.clone(),
            tolerance_m: edge.tolerance_m.canonical_value(),
            endpoint_ids: edge
                .start_vertex_id
                .iter()
                .chain(&edge.end_vertex_id)
                .cloned()
                .collect(),
            endpoint_vertices: [edge.start_vertex_id.clone(), edge.end_vertex_id.clone()],
            face_ids: relations
                .edge_coedges(&edge.id)
                .map(|coedge| coedge.face_id.clone())
                .collect(),
            solid_ids: relations
                .edge_coedges(&edge.id)
                .filter_map(|coedge| face_solids.get(&coedge.face_id))
                .flatten()
                .cloned()
                .collect(),
            points_m: witness_points,
        });
    }
    contributions.extend(face_metrics.into_contributions()?);
    let contact_pairs = contact_proximity_pairs(topology, &relations);
    let search_radius_m = request
        .global_metric
        .conservative_minimum_length_m()
        .map_err(|error| {
            SharedCurveError::invalid_request("curve proximity radius", error.to_string())
        })?
        * f64::from(request.resolved_cells_across_gap());
    // Exact edge samples locate edge-local sizing; consumers grade it by distance from the
    // edge instead of applying it across each face the edge bounds.
    let mut supports = edge_witnesses
        .iter()
        .map(|witness| runmat_meshing_size::metric::MetricEntitySupport {
            entity_id: witness.edge_id.clone(),
            points_m: witness.points_m.clone(),
        })
        .collect::<Vec<_>>();
    // A gap that closes at an edge's endpoint is sized from that vertex (see
    // `closing_endpoint`); the vertex's exact point is its support.
    let vertex_points = edge_witnesses
        .iter()
        .flat_map(|witness| {
            let first = witness.points_m.first().copied();
            let last = witness.points_m.last().copied();
            [
                witness.endpoint_vertices[0].clone().zip(first),
                witness.endpoint_vertices[1].clone().zip(last),
            ]
        })
        .flatten()
        .collect::<BTreeMap<_, _>>();
    if measure_face_gaps {
        for edge in &topology.edges {
            for coedge in relations.edge_coedges(&edge.id) {
                if let Some(witness) = face_witnesses.get_mut(&coedge.face_id) {
                    witness.tolerance_m =
                        witness.tolerance_m.max(edge.tolerance_m.canonical_value());
                    witness.vertex_ids.extend(
                        edge.start_vertex_id
                            .iter()
                            .chain(&edge.end_vertex_id)
                            .cloned(),
                    );
                }
            }
        }
        for (face_id, witness) in &mut face_witnesses {
            if let Some(solids) = face_solids.get(face_id) {
                witness.solid_ids = solids.clone();
            }
        }
        contributions.extend(derive_face_proximity_contributions(
            face_witnesses.into_values().collect(),
            &contact_pairs,
            search_radius_m,
            request.resolved_cells_across_gap(),
            control,
        )?);
    }
    let proximity = derive_proximity_contributions(
        edge_witnesses,
        &contact_pairs,
        request.global_metric,
        search_radius_m,
        request.resolved_cells_across_gap(),
        request.maximum_grading_ratio,
        control,
    )?;
    for contribution in &proximity {
        if let MetricContributionScope::Entity { entity_id } = &contribution.scope {
            if let Some(point) = vertex_points.get(entity_id) {
                supports.push(runmat_meshing_size::metric::MetricEntitySupport {
                    entity_id: entity_id.clone(),
                    points_m: vec![*point],
                });
            }
        }
    }
    contributions.extend(proximity);
    let mut resolved = request
        .intersect_contributions(contributions)
        .map_err(|error| {
            SharedCurveError::invalid_request("curve geometry metric", error.to_string())
        })?;
    // Every regular edge carries a support: edge-scoped sizing and the uniform sizing of a
    // finer adjacent face both grade from it.
    let mut entity_supports = std::mem::take(&mut resolved.entity_supports);
    entity_supports.extend(supports);
    entity_supports.sort_by(|left, right| left.entity_id.cmp(&right.entity_id));
    entity_supports.dedup_by(|left, right| left.entity_id == right.entity_id);
    resolved.entity_supports = entity_supports;
    resolved.validate().map_err(|error| {
        SharedCurveError::invalid_request("curve geometry metric supports", error.to_string())
    })?;
    Ok(resolved)
}

fn contact_proximity_pairs(
    topology: &ExactBRepTopology,
    relations: &TopologyRelations<'_>,
) -> ContactProximityPairs {
    if topology.contacts.is_empty() {
        return ContactProximityPairs::default();
    }
    let contact_faces = topology
        .contacts
        .iter()
        .flat_map(|contact| {
            contact
                .side_a_face_ids
                .iter()
                .chain(&contact.side_b_face_ids)
        })
        .collect::<BTreeSet<_>>();
    let vertices_by_face = contact_faces
        .into_iter()
        .map(|face_id| {
            let vertices = relations
                .face_coedges(face_id)
                .map(|coedge| relations.edge(&coedge.edge_id))
                .flat_map(|edge| {
                    edge.start_vertex_id
                        .iter()
                        .chain(&edge.end_vertex_id)
                        .cloned()
                })
                .collect::<BTreeSet<_>>();
            (face_id.clone(), vertices)
        })
        .collect::<BTreeMap<_, _>>();
    let mut pairs = ContactProximityPairs::default();
    for contact in &topology.contacts {
        for left in &contact.side_a_face_ids {
            for right in &contact.side_b_face_ids {
                pairs
                    .faces
                    .insert(ordered_pair(left.clone(), right.clone()));
                let left_vertices = &vertices_by_face[left];
                let right_vertices = &vertices_by_face[right];
                pairs.vertices.extend(left_vertices.iter().flat_map(|left| {
                    right_vertices
                        .iter()
                        .map(move |right| ordered_pair(left.clone(), right.clone()))
                }));
            }
        }
    }
    pairs
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

#[cfg(test)]
mod directional_tests;
