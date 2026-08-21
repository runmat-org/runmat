use std::collections::BTreeSet;

use runmat_meshing_core::{
    quality::predicate::{
        distance_squared, point_in_closed_triangle_surface, tetrahedron_centroid,
        PointInClosedSurface,
    },
    quality::tolerance::MeshingTolerance,
    MeshingCancellationSignal,
};

use super::{
    boundary_nodes::{
        boundary_node_coordinates, candidate_respects_protected_boundary_distance,
        cavity_boundary_node_centroid, cavity_boundary_triangles,
    },
    caps::local_cap_apex_candidates,
    geometry::face_centroid,
    refill_tetrahedra::raw_refill_tetrahedron_with_rejection_reason,
    topology::sorted_face,
    validation::{validate_constrained_cavity, validate_refill_options},
    ConstrainedCavity, ConstrainedCavityNode, ConstrainedCavityRefillError,
    ConstrainedCavityRefillOptions, ConstrainedCavitySteinerCandidateBudget,
};

/// Builds a canonical quality-ranked inventory of strictly interior points. This function only
/// proposes geometry; the owning CDT mutation path must insert and independently revalidate it.
pub fn generate_constrained_cavity_interior_steiner_candidates(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
    budget: ConstrainedCavitySteinerCandidateBudget,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<[f64; 3]>, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_budget(budget)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_nodes = boundary_node_coordinates(cavity, nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_nodes)?;
    let Some(cavity_centroid) = cavity_boundary_node_centroid(cavity, &boundary_nodes) else {
        return Ok(Vec::new());
    };
    let candidate_node_id = unused_node_id(nodes)?;
    let existing_points = nodes
        .iter()
        .map(|node| node.coordinates_m)
        .collect::<Vec<_>>();
    let mut faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<Vec<_>>();
    faces.sort_unstable();
    faces.dedup();
    let mut evaluated = 0_u64;
    let mut scored = Vec::<(f64, [u32; 3], [f64; 3], &'static str)>::new();

    for face in faces {
        let Some(surface_point) = face_centroid(face, &boundary_nodes) else {
            continue;
        };
        let candidates = std::iter::once(super::caps::LocalCapApexCandidate {
            coordinates_m: cavity_centroid,
            source: "cavity_centroid",
        })
        .chain(local_cap_apex_candidates(
            face,
            surface_point,
            cavity_centroid,
            &boundary_nodes,
        ));
        for candidate in candidates {
            evaluated = evaluated.checked_add(1).ok_or_else(|| {
                resource("constrained cavity Steiner candidate counter overflowed")
            })?;
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
            let point = candidate.coordinates_m;
            if existing_points
                .iter()
                .any(|existing| distance_squared(*existing, point) <= 1.0e-24)
                || !candidate_respects_protected_boundary_distance(
                    cavity,
                    &boundary_nodes,
                    point,
                    options,
                )
                || point_in_closed_triangle_surface(
                    point,
                    &boundary_triangles,
                    MeshingTolerance::default(),
                ) != PointInClosedSurface::Inside
            {
                continue;
            }
            let points = [
                boundary_nodes[&face[0]],
                boundary_nodes[&face[1]],
                boundary_nodes[&face[2]],
                point,
            ];
            if point_in_closed_triangle_surface(
                tetrahedron_centroid(points),
                &boundary_triangles,
                MeshingTolerance::default(),
            ) != PointInClosedSurface::Inside
            {
                continue;
            }
            let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                [face[0], face[1], face[2], candidate_node_id],
                points,
                options,
            ) else {
                continue;
            };
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
    let maximum_candidates = usize::try_from(budget.maximum_candidates).unwrap_or(usize::MAX);
    let mut selected = Vec::new();
    for (_, _, point, _) in scored {
        if selected
            .iter()
            .any(|existing| distance_squared(*existing, point) <= 1.0e-24)
        {
            continue;
        }
        selected.push(point);
        if selected.len() >= maximum_candidates {
            break;
        }
    }
    Ok(selected)
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
