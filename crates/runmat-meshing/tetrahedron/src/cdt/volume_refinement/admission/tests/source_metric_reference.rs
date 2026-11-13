//! Frozen former source-metric proxy, used only to identify a counterexample.
use super::*;
use crate::cdt::{insertion::PreparedInsertion, volume_quality::evaluate_tetrahedron_geometry};
use runmat_meshing_core::StableDigest;
pub(super) fn legacy_source_metric_floor(
    topology: &impl crate::cdt::topology::VolumeTopologyRead,
    candidate: &crate::cdt::DelaunayVolumeRefinementCandidate,
    insertion: &PreparedInsertion,
    exact_boundary_nodes: &BTreeSet<StableDigest>,
    options: crate::cdt::DelaunayVolumeQualityOptions,
) -> bool {
    insertion.boundary.iter().all(|face| {
        let points = [
            topology.nodes()[face[0] as usize]
                .coordinates_m
                .canonical_value(),
            topology.nodes()[face[1] as usize]
                .coordinates_m
                .canonical_value(),
            topology.nodes()[face[2] as usize]
                .coordinates_m
                .canonical_value(),
            candidate.node.coordinates_m.canonical_value(),
        ];
        let Some(geometry) =
            evaluate_tetrahedron_geometry(points, candidate.resolved_metric).unwrap()
        else {
            return false;
        };
        let size_violation =
            geometry.maximum_metric_edge_length / options.maximum_metric_edge_length;
        let boundary_node_count = face
            .iter()
            .filter(|index| {
                exact_boundary_nodes.contains(&topology.nodes()[**index as usize].identity)
            })
            .count();
        let violation = if boundary_node_count == 0 {
            size_violation
                .max(geometry.metric_radius_edge_ratio / options.maximum_radius_edge_ratio)
        } else {
            size_violation
        };
        let jacobian_violation =
            options.minimum_metric_scaled_jacobian / geometry.metric_scaled_jacobian;
        violation.is_finite()
            && jacobian_violation.is_finite()
            && crate::cdt::volume_quality::violation_priority_key(jacobian_violation)
                <= crate::cdt::volume_quality::violation_priority_key(1.0)
            && crate::cdt::volume_quality::violation_priority_key(violation)
                <= crate::cdt::volume_quality::violation_priority_key(
                    candidate.source_violation_ratio,
                )
    })
}
