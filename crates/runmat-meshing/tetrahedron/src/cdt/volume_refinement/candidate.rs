use crate::cdt::topology::VolumeTopologyRead;
use std::collections::BTreeSet;

use runmat_meshing_core::quality::predicate::{orient3d, PredicateSign};

#[path = "candidate/interior.rs"]
mod interior;
mod queries;
pub(super) use queries::CandidateQueries;

pub(super) use interior::canonical_interior_points;

use super::metric_circumcenter::metric_circumcenter;
use super::{
    candidate_identity, error, CandidateWork, DelaunayRefinementCandidateKind,
    DelaunayTetrahedronQuality, DelaunayVolumeRefinementCandidate,
    DelaunayVolumeRefinementCandidateError, DelaunayVolumeRefinementCandidateErrorKind,
    DelaunayVolumeTopology,
};

#[derive(Clone, Copy)]
struct CandidateChoice {
    kind: DelaunayRefinementCandidateKind,
    coordinates_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
    construction_node_identities: [runmat_meshing_core::StableDigest; 4],
    construction_point_index: u16,
}

pub(super) fn construct_candidate(
    topology: &DelaunayVolumeTopology,
    queries: &CandidateQueries,
    quality: &DelaunayTetrahedronQuality,
    quality_options: crate::cdt::DelaunayVolumeQualityOptions,
    fixed_node_identities: &[runmat_meshing_core::StableDigest],
    excluded_node_identities: &BTreeSet<runmat_meshing_core::StableDigest>,
    work: &mut CandidateWork<'_>,
) -> Result<DelaunayVolumeRefinementCandidate, DelaunayVolumeRefinementCandidateError> {
    construct_candidate_on_topology(
        &queries.topology(topology),
        quality,
        quality_options,
        fixed_node_identities,
        excluded_node_identities,
        work,
    )
}

pub(in crate::cdt::volume_refinement) fn construct_candidate_on_topology(
    topology: &impl VolumeTopologyRead,
    quality: &DelaunayTetrahedronQuality,
    quality_options: crate::cdt::DelaunayVolumeQualityOptions,
    fixed_node_identities: &[runmat_meshing_core::StableDigest],
    excluded_node_identities: &BTreeSet<runmat_meshing_core::StableDigest>,
    work: &mut CandidateWork<'_>,
) -> Result<DelaunayVolumeRefinementCandidate, DelaunayVolumeRefinementCandidateError> {
    work.evaluate()?;
    let missing = || {
        error(
            DelaunayVolumeRefinementCandidateErrorKind::InvalidTopology,
            "quality source tetrahedron is absent from topology",
        )
    };
    let vertices = quality.node_identities.map(|id| topology.node_index(id));
    let [Some(a), Some(b), Some(c), Some(d)] = vertices else {
        return Err(missing());
    };
    let tetrahedron_index = topology
        .tetrahedron_index([a, b, c, d])
        .ok_or_else(missing)? as usize;
    let mut points = topology
        .tetrahedron(tetrahedron_index as u32)
        .expect("resolved quality source")
        .vertex_indices
        .map(|vertex| {
            topology.nodes()[vertex as usize]
                .coordinates_m
                .canonical_value()
        });
    // Topology indices are canonically ordered by persistent identity for the
    // published contract. Candidate arithmetic is geometric, so its operand
    // order must instead be invariant under a source-revision identity change.
    points.sort_by(compare_points);
    let mut choices = Vec::new();
    if let Some(center) = metric_circumcenter(points, quality.resolved_metric) {
        // A Delaunay insertion removes the complete circumsphere cavity; it does
        // not split only the source tetrahedron. Consequently, the source's four
        // hypothetical point-split children are not an admission criterion. The
        // prepared cavity and resulting quality are validated downstream.
        let center_is_admissible = strictly_inside_region_from(
            topology,
            tetrahedron_index,
            &quality.region_id,
            center,
            work.cancellation,
        )?;
        if center_is_admissible {
            choices.push(CandidateChoice {
                kind: DelaunayRefinementCandidateKind::MetricCircumcenter,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(center),
                construction_node_identities: quality.node_identities,
                construction_point_index: 0,
            });
        } else if quality.metric_radius_edge_ratio / quality_options.maximum_radius_edge_ratio
            > quality.maximum_metric_edge_length / quality_options.maximum_metric_edge_length
        {
            choices.extend(
                metric_offcenter_points(
                    topology,
                    tetrahedron_index,
                    &quality.region_id,
                    points,
                    center,
                    work.cancellation,
                )?
                .into_iter()
                .enumerate()
                .filter_map(|(index, coordinates_m)| {
                    Some(CandidateChoice {
                        kind: DelaunayRefinementCandidateKind::MetricOffcenter,
                        coordinates_m: runmat_types::physical::Quantity::<
                            runmat_types::physical::kind::Position,
                            _,
                        >::new(coordinates_m),
                        construction_node_identities: quality.node_identities,
                        construction_point_index: u16::try_from(index).ok()?,
                    })
                }),
            );
        }
    }
    work.evaluate()?;
    // Preserve the original family order, but construct only the prefix needed
    // to find an available point. A valid circumcenter needs no interior lattice.
    let choices = choices.into_iter().map(Ok).chain(
        interior::CanonicalInteriorPoints::new(points)
            .enumerate()
            .filter_map(|(index, coordinates)| {
                let construction_point_index = u16::try_from(index).ok()?;
                Some(coordinates.map(|coordinates_m| CandidateChoice {
                    kind: DelaunayRefinementCandidateKind::InteriorPoint,
                    coordinates_m: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Position,
                        _,
                    >::new(coordinates_m),
                    construction_node_identities: quality.node_identities,
                    construction_point_index,
                }))
            }),
    );
    let (choice, identity) = first_available_choice(
        choices,
        quality,
        topology,
        excluded_node_identities,
        work.cancellation,
    )?
    .ok_or_else(|| {
            let source_roles = quality.node_identities.map(|identity| {
                let index = topology.node_index(identity).expect("quality source corner");
                (topology.vertex_is_boundary(index),
                    fixed_node_identities.binary_search(&identity).is_ok())
            });
            error(
                DelaunayVolumeRefinementCandidateErrorKind::NumericalFailure,
                format!(
                    "source tetrahedron {points:?} has no unused representable canonical interior point; source node (boundary, fixed) roles are {source_roles:?}; metric edge ratio {}, radius-edge ratio {}, scaled Jacobian {}, and aggregate violation ratio {}",
                    quality.maximum_metric_edge_length,
                    quality.metric_radius_edge_ratio,
                    quality.metric_scaled_jacobian,
                    quality.refinement_violation_ratio,
                ),
            )
        })?;
    if identity == runmat_meshing_core::StableDigest::ZERO {
        return Err(error(
            DelaunayVolumeRefinementCandidateErrorKind::InvalidCandidate,
            "refinement candidate identity is zero",
        ));
    }
    Ok(DelaunayVolumeRefinementCandidate {
        node: super::DelaunayVolumeNode::explicit(identity, choice.coordinates_m.canonical_value()),
        kind: choice.kind,
        construction_node_identities: choice.construction_node_identities,
        construction_point_index: choice.construction_point_index,
        source_node_identities: quality.node_identities,
        region_id: quality.region_id.clone(),
        incident_metric_entity_ids: quality.incident_metric_entity_ids.clone(),
        resolved_metric: quality.resolved_metric,
        source_violation_ratio: quality.refinement_violation_ratio,
    })
}

fn compare_points(left: &[f64; 3], right: &[f64; 3]) -> std::cmp::Ordering {
    left.iter()
        .zip(right.iter())
        .find_map(|(left, right)| (left != right).then(|| left.total_cmp(right)))
        .unwrap_or(std::cmp::Ordering::Equal)
}

pub(in crate::cdt::volume_refinement) fn metric_offcenter_points(
    topology: &impl VolumeTopologyRead,
    source_index: usize,
    region_id: &runmat_geometry_core::PersistentEntityId,
    points: [[f64; 3]; 4],
    circumcenter: [f64; 3],
    cancellation: &dyn runmat_meshing_core::MeshingCancellationSignal,
) -> Result<Vec<[f64; 3]>, DelaunayVolumeRefinementCandidateError> {
    const BRACKET_STEPS: u32 = 32;
    const BISECTION_STEPS: u32 = 48;
    const INWARD_FRACTIONS: [f64; 6] = [0.9375, 0.875, 0.75, 0.5, 0.25, 0.125];

    let centroid = average(points.into_iter());
    let point_at = |fraction: f64| {
        std::array::from_fn(|axis| {
            centroid[axis] + fraction * (circumcenter[axis] - centroid[axis])
        })
    };
    let mut inside = 0.0;
    let mut outside = 1.0;
    for step in 1..=BRACKET_STEPS {
        let fraction = step as f64 / BRACKET_STEPS as f64;
        if strictly_inside_region_from(
            topology,
            source_index,
            region_id,
            point_at(fraction),
            cancellation,
        )? {
            inside = fraction;
        } else {
            outside = fraction;
            break;
        }
    }
    if inside == 1.0 {
        return Ok(Vec::new());
    }
    for _ in 0..BISECTION_STEPS {
        let middle = inside + (outside - inside) * 0.5;
        if middle == inside || middle == outside {
            break;
        }
        if strictly_inside_region_from(
            topology,
            source_index,
            region_id,
            point_at(middle),
            cancellation,
        )? {
            inside = middle;
        } else {
            outside = middle;
        }
    }
    let mut candidates = Vec::new();
    for fraction in INWARD_FRACTIONS {
        let candidate = point_at(inside * fraction);
        if !candidates.contains(&candidate)
            && strictly_inside_region_from(
                topology,
                source_index,
                region_id,
                candidate,
                cancellation,
            )?
        {
            candidates.push(candidate);
        }
    }
    Ok(candidates)
}

fn first_available_choice(
    choices: impl IntoIterator<Item = Result<CandidateChoice, DelaunayVolumeRefinementCandidateError>>,
    quality: &DelaunayTetrahedronQuality,
    topology: &impl VolumeTopologyRead,
    excluded_node_identities: &BTreeSet<runmat_meshing_core::StableDigest>,
    cancellation: &dyn runmat_meshing_core::MeshingCancellationSignal,
) -> Result<
    Option<(CandidateChoice, runmat_meshing_core::StableDigest)>,
    DelaunayVolumeRefinementCandidateError,
> {
    for choice in choices {
        if cancellation.is_cancelled() {
            return Err(error(
                DelaunayVolumeRefinementCandidateErrorKind::Cancelled,
                "cancelled",
            ));
        }
        let choice = choice?;
        let identity = candidate_identity(
            quality.node_identities,
            choice.kind,
            choice.construction_point_index,
        );
        if !excluded_node_identities.contains(&identity)
            && topology.node_index(identity).is_none()
            && !topology.contains_position(choice.coordinates_m.canonical_value())
        {
            return Ok(Some((choice, identity)));
        }
    }
    Ok(None)
}

fn average(points: impl Iterator<Item = [f64; 3]>) -> [f64; 3] {
    let (sum, count) = points.fold(([0.0; 3], 0_u64), |(mut sum, count), point| {
        for axis in 0..3 {
            sum[axis] += point[axis];
        }
        (sum, count + 1)
    });
    sum.map(|value| value / count as f64)
}

pub(super) fn strictly_inside_region_from(
    topology: &impl VolumeTopologyRead,
    source_index: usize,
    region_id: &runmat_geometry_core::PersistentEntityId,
    candidate: [f64; 3],
    cancellation: &dyn runmat_meshing_core::MeshingCancellationSignal,
) -> Result<bool, DelaunayVolumeRefinementCandidateError> {
    let mut visited = BTreeSet::new();
    let mut current = source_index;
    while visited.insert(current) {
        if cancellation.is_cancelled() {
            return Err(error(
                DelaunayVolumeRefinementCandidateErrorKind::Cancelled,
                "cancelled",
            ));
        }
        let tetrahedron = topology.tetrahedron(current as u32).ok_or_else(|| {
            error(
                DelaunayVolumeRefinementCandidateErrorKind::InvalidTopology,
                "candidate point-location walk reached an absent tetrahedron",
            )
        })?;
        let points = tetrahedron.vertex_indices.map(|vertex| {
            topology.nodes()[vertex as usize]
                .coordinates_m
                .canonical_value()
        });
        let mut crossing = None;
        for replace in 0..4 {
            let mut simplex = points;
            simplex[replace] = candidate;
            match orient3d(simplex).map_err(|failure| {
                error(
                    DelaunayVolumeRefinementCandidateErrorKind::NumericalFailure,
                    format!("candidate orientation failed: {failure:?}"),
                )
            })? {
                PredicateSign::Positive => {}
                PredicateSign::Zero => return Ok(false),
                PredicateSign::Negative => {
                    crossing = Some(replace);
                    break;
                }
            }
        }
        let Some(opposite) = crossing else {
            return Ok(tetrahedron.region_id.as_ref() == Some(region_id));
        };
        let Some(neighbor) = tetrahedron.neighbors[opposite] else {
            return Ok(false);
        };
        current = neighbor as usize;
    }

    // Exact orientation normally makes the adjacency walk decisive. Retain a
    // complete fallback for degenerate complexes where choosing the first of
    // several crossed faces cycles, so this optimization cannot change admission.
    for slot in 0..topology.tetrahedron_slot_count() {
        if cancellation.is_cancelled() {
            return Err(error(
                DelaunayVolumeRefinementCandidateErrorKind::Cancelled,
                "cancelled",
            ));
        }
        let Some(tetrahedron) = topology.tetrahedron(slot as u32) else {
            continue;
        };
        if tetrahedron.region_id.as_ref() != Some(region_id) {
            continue;
        }
        let points = tetrahedron.vertex_indices.map(|vertex| {
            topology.nodes()[vertex as usize]
                .coordinates_m
                .canonical_value()
        });
        if strictly_inside(points, candidate)? {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(super) fn strictly_inside(
    points: [[f64; 3]; 4],
    candidate: [f64; 3],
) -> Result<bool, DelaunayVolumeRefinementCandidateError> {
    // Candidate arithmetic uses geometric operand order, which can have either
    // orientation. Strict containment compares every child with its source.
    let source_sign = orient3d(points).map_err(|failure| {
        error(
            DelaunayVolumeRefinementCandidateErrorKind::NumericalFailure,
            format!("candidate source orientation failed: {failure:?}"),
        )
    })?;
    if source_sign == PredicateSign::Zero {
        return Ok(false);
    }
    for replace in 0..4 {
        let mut simplex = points;
        simplex[replace] = candidate;
        let sign = orient3d(simplex).map_err(|failure| {
            error(
                DelaunayVolumeRefinementCandidateErrorKind::NumericalFailure,
                format!("candidate orientation failed: {failure:?}"),
            )
        })?;
        if sign != source_sign {
            return Ok(false);
        }
    }
    Ok(true)
}
