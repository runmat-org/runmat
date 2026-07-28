use std::collections::BTreeMap;

use super::*;

pub(super) fn empty_boundary_node_completion_diagnostic(
    reason: &'static str,
) -> BoundaryNodeCompletionDiagnostic {
    BoundaryNodeCompletionDiagnostic {
        reason,
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
    }
}

pub(super) fn merge_boundary_node_completion_diagnostic(
    aggregate: &mut BoundaryNodeCompletionDiagnostic,
    diagnostic: BoundaryNodeCompletionDiagnostic,
) {
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
}
