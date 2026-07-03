use super::*;

mod scaled_worst_face;

use scaled_worst_face::scaled_worst_face_star_quality;

#[cfg(test)]
pub(crate) fn diagnostic_interior_star_quality(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    interior_candidates: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<InteriorStarQualityDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let diagnostic_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let mut diagnostic = InteriorStarQualityDiagnostic {
        candidate_count: 0,
        pass_count: 0,
        scaled_worst_face_candidate_count: 0,
        scaled_worst_face_pass_count: 0,
        max_min_scaled_jacobian: 0.0,
        max_scaled_worst_face_min_scaled_jacobian: 0.0,
        min_scaled_jacobian_bins: BTreeMap::new(),
        min_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
        rejected_by_reason: BTreeMap::new(),
    };
    let mut seen_interior_nodes = BTreeSet::<u32>::new();
    let boundary_node_ids = cavity_boundary_node_ids(cavity);
    for node in interior_candidates {
        if !seen_interior_nodes.insert(node.node_id) {
            *diagnostic
                .rejected_by_reason
                .entry("duplicate_interior_node")
                .or_default() += 1;
            continue;
        }
        if boundary_node_ids.contains(&node.node_id) {
            *diagnostic
                .rejected_by_reason
                .entry("interior_node_reuses_boundary_node")
                .or_default() += 1;
            continue;
        }
        if !candidate_respects_protected_boundary_distance(
            cavity,
            &boundary_node_map,
            node.coordinates_m,
            options,
        ) {
            *diagnostic
                .rejected_by_reason
                .entry("protected_boundary_distance")
                .or_default() += 1;
            continue;
        }
        if point_in_closed_triangle_surface(
            node.coordinates_m,
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            *diagnostic
                .rejected_by_reason
                .entry("interior_point_outside_cavity")
                .or_default() += 1;
            continue;
        }
        diagnostic.candidate_count += 1;
        match star_refill_candidate_with_rejection_reason(
            cavity,
            &boundary_node_map,
            node.clone(),
            diagnostic_options,
        ) {
            Ok(Ok(refill)) => {
                let min_quality = refill
                    .tetrahedra
                    .iter()
                    .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
                    .fold(f64::INFINITY, f64::min);
                if min_quality.is_finite() {
                    diagnostic.max_min_scaled_jacobian =
                        diagnostic.max_min_scaled_jacobian.max(min_quality);
                    *diagnostic
                        .min_scaled_jacobian_bins
                        .entry(diagnostic_scaled_jacobian_bin(min_quality))
                        .or_default() += 1;
                    if let Some(worst_tetrahedron) =
                        refill.tetrahedra.iter().min_by(|left, right| {
                            left.exact_scaled_jacobian
                                .total_cmp(&right.exact_scaled_jacobian)
                        })
                    {
                        let points = worst_tetrahedron.node_ids.map(|node_id| {
                            if node_id == node.node_id {
                                node.coordinates_m
                            } else {
                                boundary_node_map[&node_id]
                            }
                        });
                        *diagnostic
                            .min_scaled_jacobian_worst_corner_bins
                            .entry(diagnostic_scaled_jacobian_worst_corner_label(points))
                            .or_default() += 1;
                    }
                    if min_quality >= options.min_scaled_jacobian {
                        diagnostic.pass_count += 1;
                    }
                    if let Some((scaled_count, scaled_quality)) = scaled_worst_face_star_quality(
                        cavity,
                        &boundary_node_map,
                        &boundary_triangles,
                        node,
                        &refill,
                        diagnostic_options,
                    ) {
                        diagnostic.scaled_worst_face_candidate_count += scaled_count;
                        diagnostic.max_scaled_worst_face_min_scaled_jacobian = diagnostic
                            .max_scaled_worst_face_min_scaled_jacobian
                            .max(scaled_quality);
                        diagnostic.scaled_worst_face_pass_count +=
                            usize::from(scaled_quality >= options.min_scaled_jacobian);
                    }
                }
            }
            Ok(Err(reason)) => {
                *diagnostic
                    .rejected_by_reason
                    .entry(boundary_node_refill_rejection_reason(reason))
                    .or_default() += 1;
            }
            Err(err) => {
                *diagnostic
                    .rejected_by_reason
                    .entry(boundary_node_refill_validation_reason(&err))
                    .or_default() += 1;
            }
        }
    }
    Ok(diagnostic)
}
