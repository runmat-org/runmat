use std::collections::BTreeSet;

use runmat_meshing_core::{
    quality::predicate::{tetrahedron_centroid, PointInClosedSurface},
    MeshingCancellationSignal,
};

use super::{
    boundary_nodes::{
        boundary_node_coordinates, candidate_respects_protected_boundary_distance,
        cavity_boundary_node_centroid, cavity_boundary_node_ids, cavity_boundary_triangles,
        classify_cavity_point, resolved_cavity_tolerance,
    },
    caps::local_cap_apex_candidates,
    geometry::face_centroid,
    refill_tetrahedra::raw_refill_tetrahedron_with_rejection_reason,
    topology::sorted_face,
    validation::{validate_constrained_cavity, validate_refill_options},
    ConstrainedCavity, ConstrainedCavityNode, ConstrainedCavityRefillError,
    ConstrainedCavityRefillOptions, ConstrainedCavitySteinerCandidateBudget,
    RefillTetrahedronError,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ConstrainedCavitySteinerCandidateDiagnostics {
    pub evaluated: u64,
    pub duplicate_position: u64,
    pub protected_distance: u64,
    pub point_on_boundary: u64,
    pub point_outside: u64,
    pub tetrahedron_centroid_on_boundary: u64,
    pub tetrahedron_centroid_outside: u64,
    pub tetrahedron_below_minimum_volume: u64,
    pub tetrahedron_above_maximum_aspect_ratio: u64,
    pub tetrahedron_below_minimum_scaled_jacobian: u64,
    pub accepted_before_deduplication: u64,
}

/// Builds a canonical quality-ranked inventory of strictly interior points. This function only
/// proposes geometry; the owning CDT mutation path must insert and independently revalidate it.
pub fn generate_constrained_cavity_interior_steiner_candidates(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
    budget: ConstrainedCavitySteinerCandidateBudget,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<[f64; 3]>, ConstrainedCavityRefillError> {
    generate_constrained_cavity_interior_steiner_candidates_with_diagnostics(
        cavity,
        nodes,
        options,
        budget,
        cancellation,
    )
    .map(|(candidates, _)| candidates)
}

pub(crate) fn generate_constrained_cavity_interior_steiner_candidates_with_diagnostics(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
    budget: ConstrainedCavitySteinerCandidateBudget,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<
    (Vec<[f64; 3]>, ConstrainedCavitySteinerCandidateDiagnostics),
    ConstrainedCavityRefillError,
> {
    validate_refill_options(options)?;
    validate_budget(budget)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_nodes = boundary_node_coordinates(cavity, nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_nodes)?;
    let classification_tolerance =
        resolved_cavity_tolerance(&boundary_nodes, options.classification_tolerance);
    let Some(cavity_centroid) = cavity_boundary_node_centroid(cavity, &boundary_nodes) else {
        return Ok((
            Vec::new(),
            ConstrainedCavitySteinerCandidateDiagnostics::default(),
        ));
    };
    let maximum_candidates = usize::try_from(budget.maximum_candidates).unwrap_or(usize::MAX);
    let global_candidates = if classification_tolerance.absolute_m.canonical_value() == 0.0 {
        global_interior_candidates(
            cavity,
            &boundary_nodes,
            &boundary_triangles,
            classification_tolerance,
            maximum_candidates,
            usize::try_from(budget.maximum_evaluations).unwrap_or(usize::MAX),
        )
    } else {
        Vec::new()
    };
    let candidate_node_id = unused_node_id(nodes)?;
    let existing_points = nodes
        .iter()
        .map(|node| node.coordinates_m.canonical_value())
        .collect::<Vec<_>>();
    let mut faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<Vec<_>>();
    faces.sort_unstable();
    faces.dedup();
    let mut evaluated = 0_u64;
    let mut diagnostics = ConstrainedCavitySteinerCandidateDiagnostics::default();
    let mut scored = Vec::<(f64, [u32; 3], [f64; 3], &'static str)>::new();

    for face in faces {
        let Some(surface_point) = face_centroid(face, &boundary_nodes) else {
            continue;
        };
        let candidates = global_candidates
            .iter()
            .copied()
            .map(|coordinates_m| super::caps::LocalCapApexCandidate {
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(coordinates_m),
                source: "boundary_combination",
            })
            .chain(std::iter::once(super::caps::LocalCapApexCandidate {
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(cavity_centroid),
                source: "cavity_centroid",
            }))
            .chain(
                local_cap_apex_candidates(face, surface_point, cavity_centroid, &boundary_nodes)
                    .map_err(ConstrainedCavityRefillError::Accounting)?,
            );
        for candidate in candidates {
            evaluated = evaluated.checked_add(1).ok_or_else(|| {
                resource("constrained cavity Steiner candidate counter overflowed")
            })?;
            diagnostics.evaluated = evaluated;
            if evaluated > budget.maximum_evaluations {
                return Err(resource(
                    "constrained cavity Steiner candidate-evaluation limit exceeded",
                ));
            }
            if evaluated.is_multiple_of(budget.cancellation_check_interval)
                && cancellation.is_cancelled()
            {
                return Err(ConstrainedCavityRefillError::Cancelled);
            }
            let point = candidate.coordinates_m.canonical_value();
            if existing_points.contains(&point) {
                diagnostics.duplicate_position += 1;
                continue;
            }
            if !candidate_respects_protected_boundary_distance(
                cavity,
                &boundary_nodes,
                point,
                options,
            ) {
                diagnostics.protected_distance += 1;
                continue;
            }
            match classify_cavity_point(
                cavity,
                &boundary_nodes,
                &boundary_triangles,
                point,
                classification_tolerance,
            ) {
                PointInClosedSurface::Inside => {}
                PointInClosedSurface::OnBoundary => {
                    diagnostics.point_on_boundary += 1;
                    continue;
                }
                PointInClosedSurface::Outside => {
                    diagnostics.point_outside += 1;
                    continue;
                }
            }
            let points = [
                boundary_nodes[&face[0]],
                boundary_nodes[&face[1]],
                boundary_nodes[&face[2]],
                point,
            ];
            match classify_cavity_point(
                cavity,
                &boundary_nodes,
                &boundary_triangles,
                tetrahedron_centroid(points),
                classification_tolerance,
            ) {
                PointInClosedSurface::Inside => {}
                PointInClosedSurface::OnBoundary => {
                    diagnostics.tetrahedron_centroid_on_boundary += 1;
                    continue;
                }
                PointInClosedSurface::Outside => {
                    diagnostics.tetrahedron_centroid_outside += 1;
                    continue;
                }
            }
            let tetrahedron = match raw_refill_tetrahedron_with_rejection_reason(
                [face[0], face[1], face[2], candidate_node_id],
                points,
                options,
            ) {
                Ok(tetrahedron) => tetrahedron,
                Err(RefillTetrahedronError::Rejected("star_tetrahedron_min_volume")) => {
                    diagnostics.tetrahedron_below_minimum_volume += 1;
                    continue;
                }
                Err(RefillTetrahedronError::Rejected("star_tetrahedron_aspect_ratio")) => {
                    diagnostics.tetrahedron_above_maximum_aspect_ratio += 1;
                    continue;
                }
                Err(RefillTetrahedronError::Rejected("star_tetrahedron_scaled_jacobian")) => {
                    diagnostics.tetrahedron_below_minimum_scaled_jacobian += 1;
                    continue;
                }
                Err(RefillTetrahedronError::Rejected(_)) => {
                    unreachable!("raw tetrahedron rejection reasons are closed")
                }
                Err(RefillTetrahedronError::Accounting(error)) => {
                    return Err(ConstrainedCavityRefillError::Accounting(error));
                }
            };
            diagnostics.accepted_before_deduplication += 1;
            scored.push((
                tetrahedron.exact_scaled_jacobian,
                face,
                point,
                candidate.source,
            ));
        }
    }
    scored.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2[0].total_cmp(&right.2[0]))
            .then_with(|| left.2[1].total_cmp(&right.2[1]))
            .then_with(|| left.2[2].total_cmp(&right.2[2]))
            .then_with(|| left.3.cmp(right.3))
    });
    let mut selected = Vec::new();
    for (_, _, point, _) in scored {
        if selected.contains(&point) {
            continue;
        }
        selected.push(point);
        if selected.len() >= maximum_candidates {
            break;
        }
    }
    Ok((selected, diagnostics))
}

fn global_interior_candidates(
    cavity: &ConstrainedCavity,
    boundary_nodes: &std::collections::BTreeMap<u32, [f64; 3]>,
    boundary_triangles: &[[[f64; 3]; 3]],
    tolerance: runmat_meshing_core::quality::tolerance::MeshingTolerance,
    maximum_candidates: usize,
    maximum_evaluations: usize,
) -> Vec<[f64; 3]> {
    if maximum_candidates == 0 || maximum_evaluations == 0 {
        return Vec::new();
    }
    let points = cavity_boundary_node_ids(cavity)
        .into_iter()
        .filter_map(|node_id| boundary_nodes.get(&node_id).copied())
        .collect::<Vec<_>>();
    let mut candidates = Vec::new();
    let mut evaluations = 0usize;
    for first in 0..points.len() {
        for second in (first + 1)..points.len() {
            if evaluations >= maximum_evaluations {
                return candidates;
            }
            evaluations += 1;
            admit_global_candidate(
                &mut candidates,
                maximum_candidates,
                cavity,
                boundary_nodes,
                boundary_triangles,
                tolerance,
                std::array::from_fn(|axis| (points[first][axis] + points[second][axis]) * 0.5),
            );
            for third in (second + 1)..points.len() {
                if candidates.len() >= maximum_candidates || evaluations >= maximum_evaluations {
                    return candidates;
                }
                evaluations += 1;
                admit_global_candidate(
                    &mut candidates,
                    maximum_candidates,
                    cavity,
                    boundary_nodes,
                    boundary_triangles,
                    tolerance,
                    std::array::from_fn(|axis| {
                        (points[first][axis] + points[second][axis] + points[third][axis]) / 3.0
                    }),
                );
            }
        }
    }
    candidates
}

fn admit_global_candidate(
    candidates: &mut Vec<[f64; 3]>,
    maximum_candidates: usize,
    cavity: &ConstrainedCavity,
    boundary_nodes: &std::collections::BTreeMap<u32, [f64; 3]>,
    boundary_triangles: &[[[f64; 3]; 3]],
    tolerance: runmat_meshing_core::quality::tolerance::MeshingTolerance,
    point: [f64; 3],
) {
    if candidates.len() < maximum_candidates
        && classify_cavity_point(cavity, boundary_nodes, boundary_triangles, point, tolerance)
            == PointInClosedSurface::Inside
        && !candidates.contains(&point)
    {
        candidates.push(point);
    }
}

fn validate_budget(
    budget: ConstrainedCavitySteinerCandidateBudget,
) -> Result<(), ConstrainedCavityRefillError> {
    if budget.maximum_candidates == 0
        || budget.maximum_evaluations == 0
        || budget.cancellation_check_interval == 0
    {
        return Err(ConstrainedCavityRefillError::InvalidOptions);
    }
    Ok(())
}

fn unused_node_id(nodes: &[ConstrainedCavityNode]) -> Result<u32, ConstrainedCavityRefillError> {
    let used = nodes
        .iter()
        .map(|node| node.node_id)
        .collect::<BTreeSet<_>>();
    (0..=u32::MAX)
        .find(|node_id| !used.contains(node_id))
        .ok_or_else(|| resource("constrained cavity node identity space exhausted"))
}

fn resource(reason: impl Into<String>) -> ConstrainedCavityRefillError {
    ConstrainedCavityRefillError::ResourceLimit {
        reason: reason.into(),
    }
}
