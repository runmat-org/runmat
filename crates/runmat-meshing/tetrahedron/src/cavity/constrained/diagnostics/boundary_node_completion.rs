use super::*;

#[cfg(test)]
mod face_completion;
#[cfg(test)]
mod split_caps;

#[cfg(test)]
use face_completion::diagnostic_boundary_face_completion;

#[cfg(test)]
pub(crate) fn diagnostic_boundary_node_completion(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryNodeCompletionDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let points = cavity_boundary_node_ids(cavity)
        .into_iter()
        .map(|node_id| ConnectivityPoint {
            node_id,
            coordinates_m: boundary_node_map[&node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut refill_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for tetrahedron in tetrahedralize_points(&points) {
        let node_ids = tetrahedron.vertices.map(|index| points[index].node_id);
        let tetrahedron_points = tetrahedron
            .vertices
            .map(|index| points[index].coordinates_m);
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(tetrahedron_points),
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        if let Ok(tetrahedron) =
            raw_refill_tetrahedron_with_rejection_reason(node_ids, tetrahedron_points, options)
        {
            refill_tetrahedra.push(tetrahedron);
        }
    }
    let mut aggregate = BoundaryNodeCompletionDiagnostic {
        reason: "boundary_node_completion_no_missing_faces",
        missing_face_count: 0,
        cap_candidate_count: 0,
        outside_candidate_count: 0,
        duplicate_candidate_count: 0,
        max_rejected_scaled_jacobian: 0.0,
        rejected_scaled_jacobian_bins: BTreeMap::new(),
        max_rejected_cap_height_ratio: 0.0,
        rejected_cap_height_ratio_bins: BTreeMap::new(),
        rejected_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
        rejected_cap_node_ids: BTreeMap::new(),
        split_cap_candidate_count: 0,
        split_cap_pass_count: 0,
        max_split_cap_scaled_jacobian: 0.0,
        split_cap_scaled_jacobian_bins: BTreeMap::new(),
        split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
        split_cap_apex_limited_node_ids: BTreeMap::new(),
        edge_split_cap_candidate_count: 0,
        edge_split_cap_pass_count: 0,
        max_edge_split_cap_scaled_jacobian: 0.0,
        edge_split_cap_scaled_jacobian_bins: BTreeMap::new(),
        edge_split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
        edge_split_cap_apex_limited_node_ids: BTreeMap::new(),
        three_edge_split_cap_candidate_count: 0,
        three_edge_split_cap_pass_count: 0,
        max_three_edge_split_cap_scaled_jacobian: 0.0,
        three_edge_split_cap_scaled_jacobian_bins: BTreeMap::new(),
        three_edge_split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
        three_edge_split_cap_apex_limited_node_ids: BTreeMap::new(),
        rejected_by_reason: BTreeMap::new(),
    };
    loop {
        let missing_faces = missing_refill_boundary_faces(cavity, &refill_tetrahedra)
            .map_err(ConstrainedCavityRefillError::Validation)?;
        let Some(missing_face) = missing_faces.first().copied() else {
            break;
        };
        aggregate.missing_face_count = missing_faces.len();
        let diagnostic = diagnostic_boundary_face_completion(
            missing_face,
            cavity,
            &boundary_node_map,
            &refill_tetrahedra,
            &boundary_triangles,
            options,
            missing_faces.len(),
        );
        aggregate.cap_candidate_count += diagnostic.cap_candidate_count;
        aggregate.outside_candidate_count += diagnostic.outside_candidate_count;
        aggregate.duplicate_candidate_count += diagnostic.duplicate_candidate_count;
        aggregate.max_rejected_scaled_jacobian = aggregate
            .max_rejected_scaled_jacobian
            .max(diagnostic.max_rejected_scaled_jacobian);
        aggregate.max_rejected_cap_height_ratio = aggregate
            .max_rejected_cap_height_ratio
            .max(diagnostic.max_rejected_cap_height_ratio);
        for (bin, count) in diagnostic.rejected_scaled_jacobian_bins {
            *aggregate
                .rejected_scaled_jacobian_bins
                .entry(bin)
                .or_default() += count;
        }
        for (bin, count) in diagnostic.rejected_cap_height_ratio_bins {
            *aggregate
                .rejected_cap_height_ratio_bins
                .entry(bin)
                .or_default() += count;
        }
        for (bin, count) in diagnostic.rejected_scaled_jacobian_worst_corner_bins {
            *aggregate
                .rejected_scaled_jacobian_worst_corner_bins
                .entry(bin)
                .or_default() += count;
        }
        for (node_id, count) in diagnostic.rejected_cap_node_ids {
            *aggregate.rejected_cap_node_ids.entry(node_id).or_default() += count;
        }
        aggregate.split_cap_candidate_count += diagnostic.split_cap_candidate_count;
        aggregate.split_cap_pass_count += diagnostic.split_cap_pass_count;
        aggregate.max_split_cap_scaled_jacobian = aggregate
            .max_split_cap_scaled_jacobian
            .max(diagnostic.max_split_cap_scaled_jacobian);
        for (bin, count) in diagnostic.split_cap_scaled_jacobian_bins {
            *aggregate
                .split_cap_scaled_jacobian_bins
                .entry(bin)
                .or_default() += count;
        }
        for (bin, count) in diagnostic.split_cap_scaled_jacobian_worst_corner_bins {
            *aggregate
                .split_cap_scaled_jacobian_worst_corner_bins
                .entry(bin)
                .or_default() += count;
        }
        for (node_id, count) in diagnostic.split_cap_apex_limited_node_ids {
            *aggregate
                .split_cap_apex_limited_node_ids
                .entry(node_id)
                .or_default() += count;
        }
        aggregate.edge_split_cap_candidate_count += diagnostic.edge_split_cap_candidate_count;
        aggregate.edge_split_cap_pass_count += diagnostic.edge_split_cap_pass_count;
        aggregate.max_edge_split_cap_scaled_jacobian = aggregate
            .max_edge_split_cap_scaled_jacobian
            .max(diagnostic.max_edge_split_cap_scaled_jacobian);
        for (bin, count) in diagnostic.edge_split_cap_scaled_jacobian_bins {
            *aggregate
                .edge_split_cap_scaled_jacobian_bins
                .entry(bin)
                .or_default() += count;
        }
        for (bin, count) in diagnostic.edge_split_cap_scaled_jacobian_worst_corner_bins {
            *aggregate
                .edge_split_cap_scaled_jacobian_worst_corner_bins
                .entry(bin)
                .or_default() += count;
        }
        for (node_id, count) in diagnostic.edge_split_cap_apex_limited_node_ids {
            *aggregate
                .edge_split_cap_apex_limited_node_ids
                .entry(node_id)
                .or_default() += count;
        }
        aggregate.three_edge_split_cap_candidate_count +=
            diagnostic.three_edge_split_cap_candidate_count;
        aggregate.three_edge_split_cap_pass_count += diagnostic.three_edge_split_cap_pass_count;
        aggregate.max_three_edge_split_cap_scaled_jacobian = aggregate
            .max_three_edge_split_cap_scaled_jacobian
            .max(diagnostic.max_three_edge_split_cap_scaled_jacobian);
        for (bin, count) in diagnostic.three_edge_split_cap_scaled_jacobian_bins {
            *aggregate
                .three_edge_split_cap_scaled_jacobian_bins
                .entry(bin)
                .or_default() += count;
        }
        for (bin, count) in diagnostic.three_edge_split_cap_scaled_jacobian_worst_corner_bins {
            *aggregate
                .three_edge_split_cap_scaled_jacobian_worst_corner_bins
                .entry(bin)
                .or_default() += count;
        }
        for (node_id, count) in diagnostic.three_edge_split_cap_apex_limited_node_ids {
            *aggregate
                .three_edge_split_cap_apex_limited_node_ids
                .entry(node_id)
                .or_default() += count;
        }
        for (reason, count) in diagnostic.rejected_by_reason {
            *aggregate.rejected_by_reason.entry(reason).or_default() += count;
        }
        let Some(tetrahedron) = best_boundary_face_completion_tetrahedron(
            missing_face,
            cavity,
            &boundary_node_map,
            &refill_tetrahedra,
            &boundary_triangles,
            options,
        ) else {
            aggregate.reason = "boundary_node_completion_no_candidate";
            return Ok(aggregate);
        };
        refill_tetrahedra.push(tetrahedron);
    }
    if aggregate.missing_face_count == 0 {
        return Ok(BoundaryNodeCompletionDiagnostic {
            reason: "boundary_node_completion_no_missing_faces",
            missing_face_count: 0,
            cap_candidate_count: 0,
            outside_candidate_count: 0,
            duplicate_candidate_count: 0,
            max_rejected_scaled_jacobian: 0.0,
            rejected_scaled_jacobian_bins: BTreeMap::new(),
            max_rejected_cap_height_ratio: 0.0,
            rejected_cap_height_ratio_bins: BTreeMap::new(),
            rejected_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
            rejected_cap_node_ids: BTreeMap::new(),
            split_cap_candidate_count: 0,
            split_cap_pass_count: 0,
            max_split_cap_scaled_jacobian: 0.0,
            split_cap_scaled_jacobian_bins: BTreeMap::new(),
            split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
            split_cap_apex_limited_node_ids: BTreeMap::new(),
            edge_split_cap_candidate_count: 0,
            edge_split_cap_pass_count: 0,
            max_edge_split_cap_scaled_jacobian: 0.0,
            edge_split_cap_scaled_jacobian_bins: BTreeMap::new(),
            edge_split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
            edge_split_cap_apex_limited_node_ids: BTreeMap::new(),
            three_edge_split_cap_candidate_count: 0,
            three_edge_split_cap_pass_count: 0,
            max_three_edge_split_cap_scaled_jacobian: 0.0,
            three_edge_split_cap_scaled_jacobian_bins: BTreeMap::new(),
            three_edge_split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
            three_edge_split_cap_apex_limited_node_ids: BTreeMap::new(),
            rejected_by_reason: BTreeMap::new(),
        });
    }
    aggregate.reason = "boundary_node_completion_completed";
    Ok(aggregate)
}
