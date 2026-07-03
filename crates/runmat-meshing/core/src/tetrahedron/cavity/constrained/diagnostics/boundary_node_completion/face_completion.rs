use super::*;

#[cfg(test)]
use super::split_caps::{
    diagnostic_edge_split_cap_min_scaled_jacobian, diagnostic_split_cap_min_scaled_jacobian,
    diagnostic_three_edge_split_cap_min_scaled_jacobian,
};

#[cfg(test)]
pub(super) fn diagnostic_boundary_face_completion(
    face: [u32; 3],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    refill_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
    missing_face_count: usize,
) -> BoundaryNodeCompletionDiagnostic {
    let mut cap_candidate_count = 0_usize;
    let mut outside_candidate_count = 0_usize;
    let mut duplicate_candidate_count = 0_usize;
    let mut max_rejected_scaled_jacobian = 0.0_f64;
    let mut rejected_scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    let mut max_rejected_cap_height_ratio = 0.0_f64;
    let mut rejected_cap_height_ratio_bins = BTreeMap::<String, usize>::new();
    let mut rejected_scaled_jacobian_worst_corner_bins = BTreeMap::<&'static str, usize>::new();
    let mut rejected_cap_node_ids = BTreeMap::<u32, usize>::new();
    let mut split_cap_candidate_count = 0_usize;
    let mut split_cap_pass_count = 0_usize;
    let mut max_split_cap_scaled_jacobian = 0.0_f64;
    let mut split_cap_scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    let mut split_cap_scaled_jacobian_worst_corner_bins = BTreeMap::<&'static str, usize>::new();
    let mut split_cap_apex_limited_node_ids = BTreeMap::<u32, usize>::new();
    let mut edge_split_cap_candidate_count = 0_usize;
    let mut edge_split_cap_pass_count = 0_usize;
    let mut max_edge_split_cap_scaled_jacobian = 0.0_f64;
    let mut edge_split_cap_scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    let mut edge_split_cap_scaled_jacobian_worst_corner_bins =
        BTreeMap::<&'static str, usize>::new();
    let mut edge_split_cap_apex_limited_node_ids = BTreeMap::<u32, usize>::new();
    let mut three_edge_split_cap_candidate_count = 0_usize;
    let mut three_edge_split_cap_pass_count = 0_usize;
    let mut max_three_edge_split_cap_scaled_jacobian = 0.0_f64;
    let mut three_edge_split_cap_scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    let mut three_edge_split_cap_scaled_jacobian_worst_corner_bins =
        BTreeMap::<&'static str, usize>::new();
    let mut three_edge_split_cap_apex_limited_node_ids = BTreeMap::<u32, usize>::new();
    let mut rejected_by_reason = BTreeMap::<&'static str, usize>::new();
    let mut saw_non_duplicate = false;
    for node_id in cavity_boundary_node_ids(cavity) {
        if face.contains(&node_id) {
            continue;
        }
        let node_ids = [face[0], face[1], face[2], node_id];
        let points = node_ids.map(|id| boundary_nodes[&id]);
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(points),
            boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            outside_candidate_count += 1;
            continue;
        }
        match raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options) {
            Ok(tetrahedron) => {
                cap_candidate_count += 1;
                if refill_tetrahedra.iter().any(|existing| {
                    sorted_tetrahedron_nodes(existing.node_ids)
                        == sorted_tetrahedron_nodes(tetrahedron.node_ids)
                }) {
                    duplicate_candidate_count += 1;
                } else {
                    saw_non_duplicate = true;
                }
            }
            Err(reason) => {
                *rejected_cap_node_ids.entry(node_id).or_default() += 1;
                let exact_scaled_jacobian = tetrahedron_scaled_jacobian(points);
                if exact_scaled_jacobian.is_finite() {
                    max_rejected_scaled_jacobian =
                        max_rejected_scaled_jacobian.max(exact_scaled_jacobian);
                    *rejected_scaled_jacobian_bins
                        .entry(diagnostic_scaled_jacobian_bin(exact_scaled_jacobian))
                        .or_default() += 1;
                    *rejected_scaled_jacobian_worst_corner_bins
                        .entry(diagnostic_scaled_jacobian_worst_corner_label(points))
                        .or_default() += 1;
                }
                let cap_height_ratio =
                    diagnostic_face_apex_height_ratio(face, node_id, boundary_nodes);
                if cap_height_ratio.is_finite() {
                    max_rejected_cap_height_ratio =
                        max_rejected_cap_height_ratio.max(cap_height_ratio);
                    *rejected_cap_height_ratio_bins
                        .entry(diagnostic_height_ratio_bin(cap_height_ratio))
                        .or_default() += 1;
                }
                if let Some((split_min_quality, split_worst_corner)) =
                    diagnostic_split_cap_min_scaled_jacobian(face, node_id, boundary_nodes, options)
                {
                    split_cap_candidate_count += 1;
                    max_split_cap_scaled_jacobian =
                        max_split_cap_scaled_jacobian.max(split_min_quality);
                    *split_cap_scaled_jacobian_bins
                        .entry(diagnostic_scaled_jacobian_bin(split_min_quality))
                        .or_default() += 1;
                    *split_cap_scaled_jacobian_worst_corner_bins
                        .entry(split_worst_corner)
                        .or_default() += 1;
                    if split_worst_corner == "apex" {
                        *split_cap_apex_limited_node_ids.entry(node_id).or_default() += 1;
                    }
                    if split_min_quality >= options.min_scaled_jacobian {
                        split_cap_pass_count += 1;
                    }
                }
                if let Some((edge_split_min_quality, edge_split_worst_corner)) =
                    diagnostic_edge_split_cap_min_scaled_jacobian(
                        face,
                        node_id,
                        boundary_nodes,
                        options,
                    )
                {
                    edge_split_cap_candidate_count += 1;
                    max_edge_split_cap_scaled_jacobian =
                        max_edge_split_cap_scaled_jacobian.max(edge_split_min_quality);
                    *edge_split_cap_scaled_jacobian_bins
                        .entry(diagnostic_scaled_jacobian_bin(edge_split_min_quality))
                        .or_default() += 1;
                    *edge_split_cap_scaled_jacobian_worst_corner_bins
                        .entry(edge_split_worst_corner)
                        .or_default() += 1;
                    if edge_split_worst_corner == "apex" {
                        *edge_split_cap_apex_limited_node_ids
                            .entry(node_id)
                            .or_default() += 1;
                    }
                    if edge_split_min_quality >= options.min_scaled_jacobian {
                        edge_split_cap_pass_count += 1;
                    }
                }
                if let Some((three_edge_split_min_quality, three_edge_split_worst_corner)) =
                    diagnostic_three_edge_split_cap_min_scaled_jacobian(
                        face,
                        node_id,
                        boundary_nodes,
                        options,
                    )
                {
                    three_edge_split_cap_candidate_count += 1;
                    max_three_edge_split_cap_scaled_jacobian =
                        max_three_edge_split_cap_scaled_jacobian.max(three_edge_split_min_quality);
                    *three_edge_split_cap_scaled_jacobian_bins
                        .entry(diagnostic_scaled_jacobian_bin(three_edge_split_min_quality))
                        .or_default() += 1;
                    *three_edge_split_cap_scaled_jacobian_worst_corner_bins
                        .entry(three_edge_split_worst_corner)
                        .or_default() += 1;
                    if three_edge_split_worst_corner == "apex" {
                        *three_edge_split_cap_apex_limited_node_ids
                            .entry(node_id)
                            .or_default() += 1;
                    }
                    if three_edge_split_min_quality >= options.min_scaled_jacobian {
                        three_edge_split_cap_pass_count += 1;
                    }
                }
                *rejected_by_reason
                    .entry(boundary_node_refill_rejection_reason(reason))
                    .or_default() += 1;
            }
        }
    }
    let reason = if saw_non_duplicate {
        "boundary_node_completion_has_candidate"
    } else if duplicate_candidate_count > 0 {
        "boundary_node_completion_duplicate_tetrahedron"
    } else {
        "boundary_node_completion_no_candidate"
    };
    BoundaryNodeCompletionDiagnostic {
        reason,
        missing_face_count,
        cap_candidate_count,
        outside_candidate_count,
        duplicate_candidate_count,
        max_rejected_scaled_jacobian,
        rejected_scaled_jacobian_bins,
        max_rejected_cap_height_ratio,
        rejected_cap_height_ratio_bins,
        rejected_scaled_jacobian_worst_corner_bins,
        rejected_cap_node_ids,
        split_cap_candidate_count,
        split_cap_pass_count,
        max_split_cap_scaled_jacobian,
        split_cap_scaled_jacobian_bins,
        split_cap_scaled_jacobian_worst_corner_bins,
        split_cap_apex_limited_node_ids,
        edge_split_cap_candidate_count,
        edge_split_cap_pass_count,
        max_edge_split_cap_scaled_jacobian,
        edge_split_cap_scaled_jacobian_bins,
        edge_split_cap_scaled_jacobian_worst_corner_bins,
        edge_split_cap_apex_limited_node_ids,
        three_edge_split_cap_candidate_count,
        three_edge_split_cap_pass_count,
        max_three_edge_split_cap_scaled_jacobian,
        three_edge_split_cap_scaled_jacobian_bins,
        three_edge_split_cap_scaled_jacobian_worst_corner_bins,
        three_edge_split_cap_apex_limited_node_ids,
        rejected_by_reason,
    }
}
