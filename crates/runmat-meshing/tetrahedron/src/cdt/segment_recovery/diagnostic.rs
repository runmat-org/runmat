//! Bounded evidence for recovery failures; never changes the topology.
use super::{
    error, find_node, node_index, DelaunaySegmentRecoveryError, DelaunaySegmentRecoveryErrorKind,
    DyadicNode, SegmentContext, StableDigest,
};
use crate::cdt::insertion::topology_access::TetrahedralTopology;

pub(super) fn trace_rejected_candidate(
    constraint_index: u32,
    failure: &crate::cdt::DelaunayInsertionError,
) {
    tracing::debug!(target: "runmat::meshing::segment_recovery",
        constraint_index,
        failure_kind = ?failure.kind,
        failure = %failure,
        "local edge-recovery candidate failed constrained topology validation"
    );
}

/// One debug event per recovery attempt, before subdivision hides the failed neighborhood.
/// Fixed-size samples avoid retaining or exporting a complete physical topology.
pub(super) fn trace_unrecovered_interval(
    topology: &impl TetrahedralTopology,
    segment: SegmentContext,
    left: DyadicNode,
    right: DyadicNode,
    depth: u8,
) {
    const TARGET: &str = "runmat::meshing::segment_recovery";
    if !tracing::enabled!(target: TARGET, tracing::Level::DEBUG) {
        return;
    }
    let endpoints = [left.identity, right.identity];
    let coordinates = endpoints.map(|identity| {
        find_node(topology, identity).map(|node| node.coordinates_m.canonical_value())
    });
    let mut samples = [None; 16];
    let mut count = 0;
    let mut incident_counts = [0; 2];
    for (slot, identity) in endpoints.into_iter().enumerate() {
        if let Some(index) = node_index(topology, identity) {
            let star = topology.vertex_star(index as u32);
            incident_counts[slot] = star.len();
            for tetrahedron_index in star {
                if count == samples.len() {
                    break;
                }
                if let Some(tetrahedron) = topology.tetrahedron(tetrahedron_index) {
                    samples[count] = Some((
                        tetrahedron_index,
                        tetrahedron.vertex_indices.map(|node| {
                            topology.nodes()[node as usize]
                                .coordinates_m
                                .canonical_value()
                        }),
                    ));
                    count += 1;
                }
            }
        }
    }
    tracing::debug!(target: TARGET,
        constraint_index = segment.constraint_index,
        depth,
        ?endpoints,
        ?coordinates,
        ?incident_counts,
        sampled_tetrahedra = ?&samples[..count],
        "segment has no encroaching witness and local recovery did not produce its edge"
    );
}

pub(super) fn split_limit_failure(
    topology: &impl TetrahedralTopology,
    segment: SegmentContext,
    left: DyadicNode,
    right: DyadicNode,
    witness: Option<(StableDigest, [f64; 3])>,
) -> DelaunaySegmentRecoveryError {
    let nodes = [left.identity, right.identity].map(|identity| find_node(topology, identity));
    let coordinates = nodes.map(|node| node.map(|node| node.coordinates_m.canonical_value()));
    let constructions = nodes.map(|node| node.map(|node| node.construction));
    let incident_tetrahedra = [left.identity, right.identity].map(|identity| {
        node_index(topology, identity).map(|index| topology.vertex_star(index as u32).len())
    });
    let neighbors = [left.identity, right.identity].map(|identity| {
        let mut neighbors = std::collections::BTreeSet::new();
        if let Some(index) = node_index(topology, identity) {
            for tetrahedron in topology.vertex_star(index as u32) {
                if let Some(tetrahedron) = topology.tetrahedron(tetrahedron) {
                    neighbors.extend(tetrahedron.vertex_indices);
                }
            }
        }
        neighbors
    });
    let shared_neighbors = neighbors[0]
        .intersection(&neighbors[1])
        .take(8)
        .map(|index| {
            topology.nodes()[*index as usize]
                .coordinates_m
                .canonical_value()
        })
        .collect::<Vec<_>>();
    error(
        DelaunaySegmentRecoveryErrorKind::UnsatisfiableConstraint,
        Some(segment.constraint_index),
        format!(
            "segment {:?} terminal subinterval {:?} at {:?} -> {:?} with encroaching witness {witness:?} did not recover within the dyadic split-depth limit; original coordinates {:?}; endpoint incident tetrahedra {incident_tetrahedra:?}; endpoint constructions {constructions:?}; shared neighbor coordinates (at most eight) {shared_neighbors:?}",
            [segment.first_identity, segment.last_identity],
            [left.identity, right.identity],
            coordinates[0], coordinates[1],
            [segment.first_coordinates, segment.last_coordinates],
        ),
    )
}
