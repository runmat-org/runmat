#[cfg(test)]
use super::validate_delaunay_volume_topology;
mod batch;
mod schedule;
use batch::recover_delaunay_segments_with_contract;
use std::collections::BTreeSet;

use runmat_meshing_core::quality::predicate::{Point3Construction, PredicateSign};
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::{
    validate_delaunay_constraints, DelaunayConstraintError, DelaunayConstraintErrorKind,
    DelaunayConstraintOptions, DelaunayConstraints, DelaunayInsertionError,
    DelaunayInsertionErrorKind, DelaunayInsertionOptions, DelaunayVolumeNode,
    DelaunayVolumeTopology,
};
use crate::cdt::{
    insertion::{
        insert_delaunay_volume_node_into_mutation,
        insert_delaunay_volume_node_into_mutation_with_barriers,
        legalize_delaunay_volume_topology_around_nodes_with_barriers,
        legalize_delaunay_volume_topology_completely_with_barriers, legalize_inserted_nodes,
        observed_change_violation, topology_access::TetrahedralTopology,
        validate_constrained_delaunay_volume_topology,
    },
    mutation::{TopologyMutation, TopologyMutationError, TopologyMutationErrorKind},
};

mod acute;
mod candidate;
mod diagnostic;
mod encroachment;
mod flip;
mod interval;
mod location;
use interval::{recover_interval, IntervalRecoveryContext};
mod local_reconnection;
mod parameter;
mod validation;
mod work;

use acute::classify_acute_vertices;
use encroachment::{encroaching_point, split_parameter};
use flip::{try_recover_edge_with_face_flip, try_recover_edge_with_local_flip_mutation};
pub(in crate::cdt) use location::coordinates_share_representable_location;
use parameter::{interpolate, steiner_identity, DyadicNode, SegmentContext};

pub use validation::validate_delaunay_segment_recovery;
pub(in crate::cdt) use validation::validate_delaunay_segment_recovery_on_topology;
use validation::{validate_inputs, SegmentRecoveryInputContract};
use work::RecoveryWork;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelaunaySegmentRecoveryOptions {
    pub constraints: DelaunayConstraintOptions,
    pub insertion: DelaunayInsertionOptions,
    pub maximum_steiner_nodes: u64,
    pub maximum_recovery_steps: u64,
    pub maximum_search_steps: u64,
    pub maximum_flip_attempts: u64,
    pub maximum_split_depth: u8,
    pub maximum_recovery_passes: u32,
}

impl Default for DelaunaySegmentRecoveryOptions {
    fn default() -> Self {
        Self {
            constraints: DelaunayConstraintOptions::default(),
            insertion: DelaunayInsertionOptions::default(),
            maximum_steiner_nodes: 10_000_000,
            maximum_recovery_steps: 100_000_000,
            maximum_search_steps: 1_000_000_000,
            maximum_flip_attempts: 100_000_000,
            maximum_split_depth: 48,
            maximum_recovery_passes: 64,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelaunayRecoveredSegmentNode {
    pub identity: StableDigest,
    pub parameter_numerator: u64,
    pub parameter_exponent: u8,
}

impl DelaunayRecoveredSegmentNode {
    pub(super) fn parameter(self) -> Result<f64, DelaunaySegmentRecoveryError> {
        DyadicNode {
            identity: self.identity,
            numerator: self.parameter_numerator,
            exponent: self.parameter_exponent,
        }
        .parameter()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayRecoveredSegment {
    pub constraint_index: u32,
    pub nodes: Vec<DelaunayRecoveredSegmentNode>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DelaunaySegmentRecovery {
    pub topology: DelaunayVolumeTopology,
    pub segments: Vec<DelaunayRecoveredSegment>,
    pub steiner_node_identities: Vec<StableDigest>,
    pub recovery_passes: u32,
}

/// Sorted node-identity keys of edges, for the edges link flips must keep.
pub(in crate::cdt) fn sorted_edge_keys(
    edges: impl IntoIterator<Item = [StableDigest; 2]>,
) -> BTreeSet<[StableDigest; 2]> {
    edges
        .into_iter()
        .map(|mut edge| {
            edge.sort_unstable();
            edge
        })
        .collect()
}

/// Every interval of every recovered segment, as sorted node-identity keys.
pub(in crate::cdt) fn recovered_segment_edge_keys(
    segments: &[DelaunayRecoveredSegment],
) -> BTreeSet<[StableDigest; 2]> {
    sorted_edge_keys(
        segments
            .iter()
            .flat_map(|segment| segment.nodes.windows(2))
            .map(|pair| [pair[0].identity, pair[1].identity]),
    )
}

pub(in crate::cdt) fn recovered_segment_midpoint(
    recovery: &DelaunaySegmentRecovery,
    segment_index: usize,
    interval_index: usize,
) -> Result<(DelaunayVolumeNode, DelaunayRecoveredSegmentNode), DelaunaySegmentRecoveryError> {
    recovered_segment_midpoint_on(
        &recovery.topology,
        &recovery.segments,
        segment_index,
        interval_index,
    )
}

pub(in crate::cdt) fn recovered_segment_midpoint_on(
    topology: &impl TetrahedralTopology,
    segments: &[DelaunayRecoveredSegment],
    segment_index: usize,
    interval_index: usize,
) -> Result<(DelaunayVolumeNode, DelaunayRecoveredSegmentNode), DelaunaySegmentRecoveryError> {
    recovered_segment_point_on(topology, segments, segment_index, interval_index, None)?.ok_or_else(
        || {
            error(
                DelaunaySegmentRecoveryErrorKind::UnsatisfiableConstraint,
                None,
                "recovered segment midpoint is unavailable",
            )
        },
    )
}

/// A split point strictly inside one recovered segment interval: its dyadic midpoint, or, with
/// `at_distance = (endpoint, distance_m)`, the coarsest dyadic parameter nearest the point at
/// that distance from the interval endpoint. `None` when no such parameter lies strictly inside
/// the interval.
pub(in crate::cdt) fn recovered_segment_point_on(
    topology: &impl TetrahedralTopology,
    segments: &[DelaunayRecoveredSegment],
    segment_index: usize,
    interval_index: usize,
    at_distance: Option<(StableDigest, f64)>,
) -> Result<Option<(DelaunayVolumeNode, DelaunayRecoveredSegmentNode)>, DelaunaySegmentRecoveryError>
{
    let segment = segments.get(segment_index).ok_or_else(|| {
        error(
            DelaunaySegmentRecoveryErrorKind::InvalidConstraints,
            Some(segment_index as u32),
            "recovered segment index is outside the evidence inventory",
        )
    })?;
    let [left, right] = segment
        .nodes
        .get(interval_index..=interval_index + 1)
        .and_then(|pair| <&[DelaunayRecoveredSegmentNode; 2]>::try_from(pair).ok())
        .copied()
        .ok_or_else(|| {
            error(
                DelaunaySegmentRecoveryErrorKind::InvalidConstraints,
                Some(segment.constraint_index),
                "recovered segment interval is outside the chain",
            )
        })?;
    let first = segment
        .nodes
        .first()
        .copied()
        .expect("validated segment chain");
    let last = segment
        .nodes
        .last()
        .copied()
        .expect("validated segment chain");
    let first_coordinates = find_node(topology, first.identity)
        .ok_or_else(|| invalid_segment_node(segment.constraint_index))?
        .coordinates_m
        .canonical_value();
    let last_coordinates = find_node(topology, last.identity)
        .ok_or_else(|| invalid_segment_node(segment.constraint_index))?
        .coordinates_m
        .canonical_value();
    let context = SegmentContext {
        constraint_index: segment.constraint_index,
        first_identity: first.identity,
        last_identity: last.identity,
        first_coordinates,
        last_coordinates,
    };
    let left_node = DyadicNode {
        identity: left.identity,
        numerator: left.parameter_numerator,
        exponent: left.parameter_exponent,
    };
    let right_node = DyadicNode {
        identity: right.identity,
        numerator: right.parameter_numerator,
        exponent: right.parameter_exponent,
    };
    let midpoint = match at_distance {
        None => left_node.midpoint(right_node, context)?,
        Some((endpoint, distance_m)) => {
            let length_m = (0..3)
                .map(|axis| (last_coordinates[axis] - first_coordinates[axis]).powi(2))
                .sum::<f64>()
                .sqrt();
            let from = if endpoint == left.identity {
                left_node.parameter()? + distance_m / length_m
            } else if endpoint == right.identity {
                right_node.parameter()? - distance_m / length_m
            } else {
                return Ok(None);
            };
            match left_node.search_near_parameter(right_node, from, false, &mut |_| Ok(true))? {
                Some(parameter) => parameter,
                None => return Ok(None),
            }
        }
    };
    let identity = steiner_identity(context, midpoint);
    let recovered = DelaunayRecoveredSegmentNode {
        identity,
        parameter_numerator: midpoint.numerator,
        parameter_exponent: midpoint.exponent,
    };
    Ok(Some((
        DelaunayVolumeNode {
            identity,
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(interpolate(
                first_coordinates,
                last_coordinates,
                midpoint.parameter()?,
            )),
            construction: Point3Construction::Segment {
                endpoints: [first_coordinates, last_coordinates],
                parameter_numerator: midpoint.numerator,
                parameter_exponent: midpoint.exponent,
            },
        },
        recovered,
    )))
}

pub(in crate::cdt) fn record_recovered_segment_split(
    recovery: &mut DelaunaySegmentRecovery,
    segment_index: usize,
    interval_index: usize,
    node: DelaunayRecoveredSegmentNode,
) -> Result<(), DelaunaySegmentRecoveryError> {
    record_recovered_segment_split_on(
        &mut recovery.segments,
        &mut recovery.steiner_node_identities,
        segment_index,
        interval_index,
        node,
    )
}

pub(in crate::cdt) fn record_recovered_segment_split_on(
    segments: &mut [DelaunayRecoveredSegment],
    steiner_node_identities: &mut Vec<StableDigest>,
    segment_index: usize,
    interval_index: usize,
    node: DelaunayRecoveredSegmentNode,
) -> Result<(), DelaunaySegmentRecoveryError> {
    let segment = segments.get_mut(segment_index).ok_or_else(|| {
        error(
            DelaunaySegmentRecoveryErrorKind::InvalidConstraints,
            Some(segment_index as u32),
            "recovered segment index is outside the evidence inventory",
        )
    })?;
    if interval_index + 1 >= segment.nodes.len() {
        return Err(error(
            DelaunaySegmentRecoveryErrorKind::InvalidConstraints,
            Some(segment.constraint_index),
            "recovered segment interval is outside the chain",
        ));
    }
    segment.nodes.insert(interval_index + 1, node);
    steiner_node_identities.push(node.identity);
    steiner_node_identities.sort_unstable();
    steiner_node_identities.dedup();
    Ok(())
}

fn invalid_segment_node(constraint_index: u32) -> DelaunaySegmentRecoveryError {
    error(
        DelaunaySegmentRecoveryErrorKind::InvalidTopology,
        Some(constraint_index),
        "recovered segment endpoint is missing from topology",
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunaySegmentRecoveryErrorKind {
    InvalidOptions,
    InvalidConstraints,
    InvalidTopology,
    IdentityCollision,
    UnsatisfiableConstraint,
    ResourceLimit,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunaySegmentRecoveryError {
    pub kind: DelaunaySegmentRecoveryErrorKind,
    pub constraint_index: Option<u32>,
    pub reason: String,
}

impl std::fmt::Display for DelaunaySegmentRecoveryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "3D Delaunay segment recovery {:?} at constraint {:?}: {}",
            self.kind, self.constraint_index, self.reason
        )
    }
}

impl std::error::Error for DelaunaySegmentRecoveryError {}

pub fn recover_delaunay_segments(
    topology: DelaunayVolumeTopology,
    constraints: &DelaunayConstraints,
    options: DelaunaySegmentRecoveryOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunaySegmentRecovery, DelaunaySegmentRecoveryError> {
    recover_delaunay_segments_with_contract(
        topology,
        constraints,
        options,
        cancellation,
        SegmentRecoveryInputContract::ExternalUnconstrainedTopology,
    )
}

pub(in crate::cdt) fn recover_delaunay_segments_from_point_set(
    topology: DelaunayVolumeTopology,
    constraints: &DelaunayConstraints,
    options: DelaunaySegmentRecoveryOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunaySegmentRecovery, DelaunaySegmentRecoveryError> {
    recover_delaunay_segments_with_contract(
        topology,
        constraints,
        options,
        cancellation,
        SegmentRecoveryInputContract::CertifiedPointSet,
    )
}

fn local_flip_can_connect_endpoints(
    topology: &impl TetrahedralTopology,
    endpoints: [StableDigest; 2],
) -> bool {
    let [Some(left), Some(right)] = endpoints.map(|identity| topology.node_index(identity)) else {
        return false;
    };
    let right_link = topology
        .vertex_star(right)
        .into_iter()
        .flat_map(|tetrahedron| {
            topology
                .tetrahedron(tetrahedron)
                .expect("live vertex star")
                .vertex_indices
        })
        .filter(|node| *node != right)
        .collect::<BTreeSet<_>>();
    for tetrahedron in topology.vertex_star(left) {
        let vertices = topology
            .tetrahedron(tetrahedron)
            .expect("live vertex star")
            .vertex_indices;
        for first in 0..4 {
            if vertices[first] == left || !right_link.contains(&vertices[first]) {
                continue;
            }
            for second in vertices.iter().skip(first + 1) {
                if *second != left && right_link.contains(second) {
                    return true;
                }
            }
        }
    }
    false
}

pub(in crate::cdt) fn try_recover_delaunay_edges_with_local_flips(
    topology: &DelaunayVolumeTopology,
    endpoints: &[[StableDigest; 2]],
    protected_edges: &[[StableDigest; 2]],
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    options: DelaunaySegmentRecoveryOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<DelaunayVolumeTopology>, DelaunaySegmentRecoveryError> {
    validate_options(options)?;
    let endpoints = endpoints
        .iter()
        .copied()
        .filter(|endpoints| local_flip_can_connect_endpoints(topology, *endpoints))
        .collect::<Vec<_>>();
    if endpoints.is_empty() {
        return Ok(None);
    }
    let mut work = RecoveryWork::new(options, cancellation);
    let mut mutation = TopologyMutation::from_canonical(topology.clone());
    let mut changed = false;
    for endpoints in &endpoints {
        if edge_exists(
            &mutation,
            endpoints[0],
            endpoints[1],
            constraint_index,
            &mut work,
        )? {
            continue;
        }
        changed |= try_recover_edge_with_local_flip_mutation(
            &mut mutation,
            endpoints[0],
            endpoints[1],
            protected_edges,
            protected_faces,
            constraint_index,
            &mut work,
        )?;
    }
    if !changed {
        return Ok(None);
    }
    admit_face_flip_mutation(
        mutation,
        protected_faces,
        constraint_index,
        options,
        cancellation,
    )
}

pub(in crate::cdt) struct TemporarySegmentRecovery {
    pub(in crate::cdt) topology: DelaunayVolumeTopology,
    pub(in crate::cdt) chain: Vec<DelaunayRecoveredSegmentNode>,
    pub(in crate::cdt) inserted_nodes: Vec<DelaunayVolumeNode>,
    pub(in crate::cdt) conflicting_protected_edges: Vec<[StableDigest; 2]>,
}

pub(in crate::cdt) struct TemporarySegmentRequest<'a> {
    pub endpoints: [StableDigest; 2],
    pub protected_edges: &'a [[StableDigest; 2]],
    pub protected_faces: &'a [[StableDigest; 3]],
    pub constraint_index: u32,
    pub options: DelaunaySegmentRecoveryOptions,
    pub cancellation: &'a dyn MeshingCancellationSignal,
}

/// Recovers a temporary facet-support segment with the same bounded interval algorithm used for
/// authored PLC segments. The temporary chain is returned to the facet layer rather than added to
/// authored segment evidence.
pub(in crate::cdt) fn recover_temporary_delaunay_segment(
    topology: &DelaunayVolumeTopology,
    request: TemporarySegmentRequest<'_>,
    mut identity_for_node: impl FnMut(u64, u8, [f64; 3]) -> StableDigest,
) -> Result<TemporarySegmentRecovery, DelaunaySegmentRecoveryError> {
    let TemporarySegmentRequest {
        endpoints,
        protected_edges,
        protected_faces,
        constraint_index,
        options,
        cancellation,
    } = request;
    validate_options(options)?;
    let [Some(first), Some(last)] = endpoints.map(|identity| find_node(topology, identity)) else {
        return Err(error(
            DelaunaySegmentRecoveryErrorKind::InvalidTopology,
            Some(constraint_index),
            "temporary segment endpoint is missing from topology",
        ));
    };
    let context = SegmentContext {
        constraint_index,
        first_identity: first.identity,
        last_identity: last.identity,
        first_coordinates: first.coordinates_m.canonical_value(),
        last_coordinates: last.coordinates_m.canonical_value(),
    };
    let original_identities = topology
        .nodes
        .iter()
        .map(|node| node.identity)
        .collect::<BTreeSet<_>>();
    let mut work = RecoveryWork::new(options, cancellation);
    let mut current = topology.clone();
    for _ in 0..options.maximum_recovery_passes {
        let mut mutation = TopologyMutation::from_canonical(current);
        let mut chain = Vec::new();
        let mut interval_identity = |_, midpoint: DyadicNode, coordinates| {
            identity_for_node(midpoint.numerator, midpoint.exponent, coordinates)
        };
        let mut interval = IntervalRecoveryContext {
            segment: context,
            protected_edges,
            protected_faces,
            split_strong_gabriel: true,
            reported_unrecovered_interval: false,
            completed_intervals: None,
            identity_for_node: &mut interval_identity,
            work: &mut work,
        };
        recover_interval(
            &mut mutation,
            DyadicNode::endpoint(first.identity, false),
            DyadicNode::endpoint(last.identity, true),
            [false, false],
            0,
            &mut chain,
            &mut interval,
        )?;
        let recovered = DelaunayRecoveredSegment {
            constraint_index,
            nodes: chain.clone(),
        };
        let candidate = mutation
            .freeze(options.insertion.topology, cancellation)
            .map_err(mutation_error)?;
        // Only the stars of the chain's nodes can hold illegal faces: Lawson flips there
        // restore the complex in time proportional to the change. The complete repair, whose
        // every step scans the whole complex, is the fallback when local flips stall.
        let focus = chain.iter().map(|node| node.identity).collect::<Vec<_>>();
        let local = legalize_delaunay_volume_topology_around_nodes_with_barriers(
            candidate.clone(),
            protected_faces,
            &focus,
            options.insertion,
            cancellation,
        )
        .ok()
        .filter(|local| {
            validate_constrained_delaunay_volume_topology(
                local,
                protected_faces,
                options.insertion,
                cancellation,
            )
            .is_ok()
        });
        let candidate = match local {
            Some(local) => local,
            None => legalize_delaunay_volume_topology_completely_with_barriers(
                candidate,
                protected_faces,
                options.insertion,
                cancellation,
            )
            .map_err(|failure| {
                temporary_error(failure, constraint_index, context, "legalization")
            })?,
        };
        let conflicting_protected_edges = protected_edges
            .iter()
            .copied()
            .filter(|edge| !canonical_edge_exists(&candidate, *edge))
            .collect::<Vec<_>>();
        if !conflicting_protected_edges.is_empty()
            || recovered_edges_exist(&candidate, &[recovered], &mut work)?
        {
            validate_constrained_delaunay_volume_topology(
                &candidate,
                protected_faces,
                options.insertion,
                cancellation,
            )
            .map_err(|failure| temporary_error(failure, constraint_index, context, "validation"))?;
            let inserted_nodes = candidate
                .nodes
                .iter()
                .copied()
                .filter(|node| !original_identities.contains(&node.identity))
                .collect();
            return Ok(TemporarySegmentRecovery {
                topology: candidate,
                chain,
                inserted_nodes,
                conflicting_protected_edges,
            });
        }
        current = candidate;
    }
    Err(error(
        DelaunaySegmentRecoveryErrorKind::UnsatisfiableConstraint,
        Some(constraint_index),
        "temporary support segment chain did not remain simultaneous within the recovery-pass limit",
    ))
}

/// Recovers missing edges with local flips directly on the working topology. The flips run in a
/// cavity trial and are kept only when the observed change is constrained Delaunay, keeps every
/// protected face and keeps every protected edge; otherwise the trial rolls back. Returns
/// whether the topology changed.
pub(in crate::cdt) fn recover_delaunay_edges_with_local_flips_in_place(
    topology: &mut TopologyMutation,
    endpoints: &[[StableDigest; 2]],
    protected_edges: &[[StableDigest; 2]],
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    options: DelaunaySegmentRecoveryOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    validate_options(options)?;
    let endpoints = endpoints
        .iter()
        .copied()
        .filter(|endpoints| local_flip_can_connect_endpoints(&*topology, *endpoints))
        .collect::<Vec<_>>();
    if endpoints.is_empty() {
        return Ok(false);
    }
    let mut work = RecoveryWork::new(options, cancellation);
    let mut trial = topology.cavity_trial();
    let admitted = {
        let mut observed = trial.observe_edge_changes();
        let mut changed = false;
        for endpoints in &endpoints {
            if edge_exists(
                &*observed,
                endpoints[0],
                endpoints[1],
                constraint_index,
                &mut work,
            )? {
                continue;
            }
            changed |= try_recover_edge_with_local_flip_mutation(
                &mut observed,
                endpoints[0],
                endpoints[1],
                protected_edges,
                protected_faces,
                constraint_index,
                &mut work,
            )?;
        }
        changed
            && local_change_is_admissible(
                &observed,
                protected_edges,
                protected_faces,
                constraint_index,
                options,
                cancellation,
            )?
            .is_none()
    };
    if admitted {
        trial.commit();
    }
    Ok(admitted)
}

/// The outcome of one in-place temporary segment recovery pass.
pub(in crate::cdt) enum InPlaceTemporarySegment {
    /// The chain is recovered and committed to the working topology.
    Recovered {
        chain: Vec<DelaunayRecoveredSegmentNode>,
        inserted_nodes: Vec<DelaunayVolumeNode>,
    },
    /// The chain would remove these protected edges, or leaves an illegal face that only their
    /// removal repairs; the topology is unchanged.
    Conflicts(Vec<[StableDigest; 2]>),
    /// Local flips did not restore a constrained-Delaunay complex, or the chain did not survive
    /// them; the topology is unchanged and the caller uses the complete repair.
    Declined(String),
}

/// One pass of [`recover_temporary_delaunay_segment`] on the working topology. The interval
/// recovery, Lawson flips in the links of the chain's new nodes and an exact check of every
/// changed cell run inside a cavity trial, so the cost follows the size of the change.
pub(in crate::cdt) fn recover_temporary_delaunay_segment_in_place(
    topology: &mut TopologyMutation,
    request: TemporarySegmentRequest<'_>,
    mut identity_for_node: impl FnMut(u64, u8, [f64; 3]) -> StableDigest,
) -> Result<InPlaceTemporarySegment, DelaunaySegmentRecoveryError> {
    let TemporarySegmentRequest {
        endpoints,
        protected_edges,
        protected_faces,
        constraint_index,
        options,
        cancellation,
    } = request;
    validate_options(options)?;
    let [Some(first), Some(last)] =
        endpoints.map(|identity| find_node(&*topology, identity).copied())
    else {
        return Err(error(
            DelaunaySegmentRecoveryErrorKind::InvalidTopology,
            Some(constraint_index),
            "temporary segment endpoint is missing from topology",
        ));
    };
    let context = SegmentContext {
        constraint_index,
        first_identity: first.identity,
        last_identity: last.identity,
        first_coordinates: first.coordinates_m.canonical_value(),
        last_coordinates: last.coordinates_m.canonical_value(),
    };
    let original_node_count = topology.nodes.len();
    let protected_face_set = protected_faces.iter().copied().collect::<BTreeSet<_>>();
    let mut work = RecoveryWork::new(options, cancellation);
    let mut trial = topology.cavity_trial();
    let outcome = {
        let mut observed = trial.observe_edge_changes();
        let mut chain = Vec::new();
        let mut interval_identity = |_, midpoint: DyadicNode, coordinates| {
            identity_for_node(midpoint.numerator, midpoint.exponent, coordinates)
        };
        let mut interval = IntervalRecoveryContext {
            segment: context,
            protected_edges,
            protected_faces,
            split_strong_gabriel: true,
            reported_unrecovered_interval: false,
            completed_intervals: None,
            identity_for_node: &mut interval_identity,
            work: &mut work,
        };
        recover_interval(
            &mut observed,
            DyadicNode::endpoint(first.identity, false),
            DyadicNode::endpoint(last.identity, true),
            [false, false],
            0,
            &mut chain,
            &mut interval,
        )?;
        let inserted = (original_node_count..observed.nodes.len())
            .map(|index| index as u32)
            .collect::<Vec<_>>();
        let kept_edges = sorted_edge_keys(
            protected_edges.iter().copied().chain(
                chain
                    .windows(2)
                    .map(|pair| [pair[0].identity, pair[1].identity]),
            ),
        );
        match legalize_inserted_nodes(
            &mut observed,
            &inserted,
            &protected_face_set,
            &kept_edges,
            options.insertion,
            cancellation,
        ) {
            Ok(()) => {}
            Err(failure)
                if matches!(
                    failure.kind,
                    DelaunayInsertionErrorKind::ResourceLimit
                        | DelaunayInsertionErrorKind::Cancelled
                ) =>
            {
                return Err(insertion_error(failure, constraint_index))
            }
            Err(failure) => {
                return Ok(InPlaceTemporarySegment::Declined(format!(
                    "local legalization of the temporary chain failed: {failure}"
                )))
            }
        }
        let conflicts = removed_protected_edges(&observed, protected_edges, &mut work)?;
        if !conflicts.is_empty() {
            InPlaceTemporarySegment::Conflicts(conflicts)
        } else if !recovered_edges_exist(
            &*observed,
            &[DelaunayRecoveredSegment {
                constraint_index,
                nodes: chain.clone(),
            }],
            &mut work,
        )? {
            InPlaceTemporarySegment::Declined(
                "local legalization removed an edge of the temporary chain".to_owned(),
            )
        } else if let Some(violation) = local_change_is_admissible(
            &observed,
            protected_edges,
            protected_faces,
            constraint_index,
            options,
            cancellation,
        )? {
            // An illegal face whose only repair removes a protected segment edge (the chain node
            // lies in the sphere of a cell across that edge, for example inside the circumcircle
            // of a coplanar recovered triangle of another facet on the far side of the edge):
            // no constrained-Delaunay complex holds the chain while the edge stays, so the
            // caller splits that edge first, as for an edge the chain removed.
            let reflex = crate::cdt::insertion::observed_change_reflex_edges(
                &observed,
                &observed.changed_cells(),
                &protected_face_set,
                options.insertion,
                cancellation,
            )
            .map_err(|failure| insertion_error(failure, constraint_index))?;
            let blocking = protected_edges
                .iter()
                .copied()
                .filter(|edge| {
                    let [Some(first), Some(second)] =
                        edge.map(|identity| observed.node_index(identity))
                    else {
                        return false;
                    };
                    let mut key = [first, second];
                    key.sort_unstable();
                    reflex.contains(&key)
                })
                .collect::<Vec<_>>();
            if blocking.is_empty() {
                InPlaceTemporarySegment::Declined(violation)
            } else {
                InPlaceTemporarySegment::Conflicts(blocking)
            }
        } else {
            InPlaceTemporarySegment::Recovered {
                chain,
                inserted_nodes: observed.nodes[original_node_count..].to_vec(),
            }
        }
    };
    if matches!(outcome, InPlaceTemporarySegment::Recovered { .. }) {
        trial.commit();
    }
    Ok(outcome)
}

/// Protected edges that an observed edit removed: only edges of removed cells can disappear.
fn removed_protected_edges(
    observed: &crate::cdt::mutation::EdgeChangeObserver<'_>,
    protected_edges: &[[StableDigest; 2]],
    work: &mut RecoveryWork<'_>,
) -> Result<Vec<[StableDigest; 2]>, DelaunaySegmentRecoveryError> {
    let mut checkpoint = || Ok::<(), DelaunaySegmentRecoveryError>(());
    let removed = observed.removed_edge_candidates(&mut checkpoint)?;
    if removed.is_empty() {
        return Ok(Vec::new());
    }
    let mut conflicts = Vec::new();
    for edge in protected_edges {
        let [Some(first), Some(second)] = edge.map(|identity| observed.node_index(identity)) else {
            continue;
        };
        let mut key = [first, second];
        key.sort_unstable();
        if !removed.contains(&key) {
            continue;
        }
        work.search_step(0)?;
        if !observed.has_edge(key) {
            conflicts.push(*edge);
        }
    }
    Ok(conflicts)
}

/// The exact local admission of an observed edit: constrained-Delaunay faces around every
/// changed cell, every protected face kept and every protected edge kept.
fn local_change_is_admissible(
    observed: &crate::cdt::mutation::EdgeChangeObserver<'_>,
    protected_edges: &[[StableDigest; 2]],
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    options: DelaunaySegmentRecoveryOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<String>, DelaunaySegmentRecoveryError> {
    let mut work = RecoveryWork::new(options, cancellation);
    if let Some(edge) = removed_protected_edges(observed, protected_edges, &mut work)?.first() {
        return Ok(Some(format!(
            "the edit removed protected edge {:?}",
            edge.map(|identity| find_node(&**observed, identity)
                .map(|node| node.coordinates_m.canonical_value()))
        )));
    }
    let protected_faces = protected_faces.iter().copied().collect::<BTreeSet<_>>();
    observed_change_violation(
        observed,
        &observed.changed_cells(),
        &protected_faces,
        options.insertion,
        cancellation,
    )
    .map_err(|failure| insertion_error(failure, constraint_index))
}

fn canonical_edge_exists(topology: &DelaunayVolumeTopology, endpoints: [StableDigest; 2]) -> bool {
    let [Ok(first), Ok(second)] = endpoints.map(|identity| {
        topology
            .nodes
            .binary_search_by_key(&identity, |node| node.identity)
    }) else {
        return false;
    };
    topology.incidence.vertex_stars[first]
        .iter()
        .any(|tetrahedron| {
            topology.tetrahedra[*tetrahedron as usize]
                .vertex_indices
                .contains(&(second as u32))
        })
}

/// Restores a previously simultaneous set of constrained edges after a local point insertion.
///
/// Point insertion can remove a constrained edge when its entire tetrahedral star belongs to the
/// conflict cavity. Facet recovery uses this transaction before admitting a Steiner point: every
/// missing edge is recovered with local flips while the other segment edges and all recovered
/// facets remain barriers. Returning `None` leaves the caller's topology untouched.
pub(in crate::cdt) fn try_restore_delaunay_edges_with_local_flips(
    topology: &DelaunayVolumeTopology,
    required_edges: &[[StableDigest; 2]],
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    options: DelaunaySegmentRecoveryOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<DelaunayVolumeTopology>, DelaunaySegmentRecoveryError> {
    validate_options(options)?;
    let mut current = topology.clone();
    for _ in 0..options.maximum_recovery_passes {
        let mut work = RecoveryWork::new(options, cancellation);
        let missing = required_edges
            .iter()
            .copied()
            .filter_map(|edge| {
                match edge_exists(&current, edge[0], edge[1], constraint_index, &mut work) {
                    Ok(true) => None,
                    Ok(false) => Some(Ok(edge)),
                    Err(failure) => Some(Err(failure)),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        if missing.is_empty() {
            return Ok(Some(current));
        }
        let Some(updated) = try_recover_delaunay_edges_with_local_flips(
            &current,
            &missing,
            required_edges,
            protected_faces,
            constraint_index,
            options,
            cancellation,
        )?
        else {
            return Ok(None);
        };
        if updated == current {
            return Ok(None);
        }
        current = updated;
    }
    Ok(None)
}

fn admit_face_flip_mutation(
    mutation: TopologyMutation,
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    options: DelaunaySegmentRecoveryOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<DelaunayVolumeTopology>, DelaunaySegmentRecoveryError> {
    let candidate = match mutation.freeze(options.insertion.topology, cancellation) {
        Ok(candidate) => candidate,
        Err(failure) if failure.kind == TopologyMutationErrorKind::InvalidTopology => {
            return Ok(None)
        }
        Err(failure) => return Err(mutation_error(failure)),
    };
    match validate_constrained_delaunay_volume_topology(
        &candidate,
        protected_faces,
        options.insertion,
        cancellation,
    ) {
        Ok(()) => Ok(Some(candidate)),
        Err(failure) if failure.kind == DelaunayInsertionErrorKind::ResourceLimit => Err(error(
            DelaunaySegmentRecoveryErrorKind::ResourceLimit,
            Some(constraint_index),
            failure.to_string(),
        )),
        Err(failure) if failure.kind == DelaunayInsertionErrorKind::Cancelled => Err(error(
            DelaunaySegmentRecoveryErrorKind::Cancelled,
            Some(constraint_index),
            failure.to_string(),
        )),
        Err(_) => Ok(None),
    }
}

fn recovered_edges_exist(
    topology: &impl TetrahedralTopology,
    recovered: &[DelaunayRecoveredSegment],
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    for segment in recovered {
        for pair in segment.nodes.windows(2) {
            if !edge_exists(
                topology,
                pair[0].identity,
                pair[1].identity,
                segment.constraint_index,
                work,
            )? {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn push_chain_node(output: &mut Vec<DelaunayRecoveredSegmentNode>, node: DyadicNode) {
    let recovered = DelaunayRecoveredSegmentNode {
        identity: node.identity,
        parameter_numerator: node.numerator,
        parameter_exponent: node.exponent,
    };
    if output.last() != Some(&recovered) {
        output.push(recovered);
    }
}

fn edge_exists(
    topology: &impl TetrahedralTopology,
    left: StableDigest,
    right: StableDigest,
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    let left_index = node_index(topology, left).ok_or_else(|| {
        error(
            DelaunaySegmentRecoveryErrorKind::InvalidTopology,
            Some(constraint_index),
            "segment endpoint is missing from topology",
        )
    })?;
    let right_index = node_index(topology, right).ok_or_else(|| {
        error(
            DelaunaySegmentRecoveryErrorKind::InvalidTopology,
            Some(constraint_index),
            "segment endpoint is missing from topology",
        )
    })?;
    work.search_step(constraint_index)?;
    Ok(topology.has_edge([left_index as u32, right_index as u32]))
}

fn find_node(
    topology: &impl TetrahedralTopology,
    identity: StableDigest,
) -> Option<&DelaunayVolumeNode> {
    node_index(topology, identity).map(|index| &topology.nodes()[index])
}

fn node_index(topology: &impl TetrahedralTopology, identity: StableDigest) -> Option<usize> {
    topology.node_index(identity).map(|index| index as usize)
}

fn validate_options(
    options: DelaunaySegmentRecoveryOptions,
) -> Result<(), DelaunaySegmentRecoveryError> {
    if options.maximum_steiner_nodes == 0
        || options.maximum_recovery_steps == 0
        || options.maximum_search_steps == 0
        || options.maximum_flip_attempts == 0
        || options.maximum_split_depth == 0
        || options.maximum_split_depth > 60
        || options.maximum_recovery_passes == 0
    {
        return Err(error(
            DelaunaySegmentRecoveryErrorKind::InvalidOptions,
            None,
            "segment recovery limits must be nonzero and split depth at most 60",
        ));
    }
    Ok(())
}

fn constraint_error(error_value: DelaunayConstraintError) -> DelaunaySegmentRecoveryError {
    let kind = match error_value.kind {
        DelaunayConstraintErrorKind::InvalidOptions => {
            DelaunaySegmentRecoveryErrorKind::InvalidOptions
        }
        DelaunayConstraintErrorKind::ResourceLimit => {
            DelaunaySegmentRecoveryErrorKind::ResourceLimit
        }
        DelaunayConstraintErrorKind::Cancelled => DelaunaySegmentRecoveryErrorKind::Cancelled,
        DelaunayConstraintErrorKind::InvalidGeometry
        | DelaunayConstraintErrorKind::InvalidBoundary
        | DelaunayConstraintErrorKind::InvalidIdentity
        | DelaunayConstraintErrorKind::IdentityCollision => {
            DelaunaySegmentRecoveryErrorKind::InvalidConstraints
        }
    };
    error(kind, None, error_value.to_string())
}

fn insertion_error(
    error_value: DelaunayInsertionError,
    constraint_index: u32,
) -> DelaunaySegmentRecoveryError {
    let kind = match error_value.kind {
        DelaunayInsertionErrorKind::Cancelled => DelaunaySegmentRecoveryErrorKind::Cancelled,
        DelaunayInsertionErrorKind::ResourceLimit => {
            DelaunaySegmentRecoveryErrorKind::ResourceLimit
        }
        DelaunayInsertionErrorKind::InvalidOptions => {
            DelaunaySegmentRecoveryErrorKind::InvalidOptions
        }
        DelaunayInsertionErrorKind::InvalidTopology
        | DelaunayInsertionErrorKind::InvalidNode
        | DelaunayInsertionErrorKind::PointOutsideTopology => {
            DelaunaySegmentRecoveryErrorKind::InvalidTopology
        }
    };
    error(kind, Some(constraint_index), error_value.to_string())
}

fn temporary_error(
    error_value: DelaunayInsertionError,
    constraint_index: u32,
    context: SegmentContext,
    stage: &str,
) -> DelaunaySegmentRecoveryError {
    let mut failure = insertion_error(error_value, constraint_index);
    failure.reason = format!(
        "temporary segment {:?}-{:?} {stage}: {}",
        context.first_coordinates, context.last_coordinates, failure.reason
    );
    failure
}

fn mutation_error(error_value: TopologyMutationError) -> DelaunaySegmentRecoveryError {
    let kind = match error_value.kind {
        TopologyMutationErrorKind::ResourceLimit => DelaunaySegmentRecoveryErrorKind::ResourceLimit,
        TopologyMutationErrorKind::Cancelled => DelaunaySegmentRecoveryErrorKind::Cancelled,
        TopologyMutationErrorKind::InvalidTopology => {
            DelaunaySegmentRecoveryErrorKind::InvalidTopology
        }
    };
    error(kind, None, error_value.reason)
}

fn resource(
    constraint_index: Option<u32>,
    reason: impl Into<String>,
) -> DelaunaySegmentRecoveryError {
    error(
        DelaunaySegmentRecoveryErrorKind::ResourceLimit,
        constraint_index,
        reason,
    )
}

fn error(
    kind: DelaunaySegmentRecoveryErrorKind,
    constraint_index: Option<u32>,
    reason: impl Into<String>,
) -> DelaunaySegmentRecoveryError {
    DelaunaySegmentRecoveryError {
        kind,
        constraint_index,
        reason: reason.into(),
    }
}

#[cfg(test)]
#[path = "segment_recovery/tests.rs"]
mod tests;
