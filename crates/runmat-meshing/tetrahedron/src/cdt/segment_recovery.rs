use std::collections::BTreeSet;

use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::{
    validate_delaunay_constraints, validate_delaunay_volume_topology, DelaunayConstraintError,
    DelaunayConstraintErrorKind, DelaunayConstraintOptions, DelaunayConstraints,
    DelaunayInsertionError, DelaunayInsertionErrorKind, DelaunayInsertionOptions,
    DelaunayVolumeNode, DelaunayVolumeTopology,
};
use crate::cdt::{
    insertion::{
        insert_delaunay_volume_node_into_mutation_with_barriers,
        insert_delaunay_volume_node_with_barriers, topology_access::TetrahedralTopology,
        MutableInsertionOutcome,
    },
    mutation::{TopologyMutation, TopologyMutationError, TopologyMutationErrorKind},
};

mod encroachment;
mod flip;
mod location;
mod parameter;
mod validation;
mod work;

use encroachment::{encroaching_point, split_parameter};
use flip::{try_recover_edge_with_face_flip, try_recover_edge_with_face_flip_mutation};
use location::coordinates_share_representable_location;
use parameter::{interpolate, steiner_identity, DyadicNode, SegmentContext};
pub use validation::validate_delaunay_segment_recovery;
pub(in crate::cdt) use validation::validate_delaunay_segment_recovery_on_topology;
use validation::validate_inputs;
use work::RecoveryWork;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    validate_options(options)?;
    validate_inputs(&topology, constraints, options, cancellation, true)?;
    let constraint_identities = constraints
        .nodes
        .iter()
        .map(|node| node.identity)
        .collect::<BTreeSet<_>>();
    let mut topology = TopologyMutation::from_canonical(topology);
    let mut work = RecoveryWork::new(options, cancellation);

    for recovery_pass in 1..=options.maximum_recovery_passes {
        let mut recovered = Vec::with_capacity(constraints.segments.len());
        for (constraint_index, segment) in constraints.segments.iter().enumerate() {
            let first = &constraints.nodes[segment.vertex_indices[0] as usize];
            let last = &constraints.nodes[segment.vertex_indices[1] as usize];
            let context = SegmentContext {
                constraint_index: constraint_index as u32,
                first_identity: first.identity,
                last_identity: last.identity,
                first_coordinates: first.coordinates_m,
                last_coordinates: last.coordinates_m,
            };
            let mut nodes = Vec::<DelaunayRecoveredSegmentNode>::new();
            recover_interval(
                &mut topology,
                context,
                DyadicNode::endpoint(first.identity, false),
                DyadicNode::endpoint(last.identity, true),
                0,
                &mut nodes,
                &[],
                &mut work,
            )?;
            recovered.push(DelaunayRecoveredSegment {
                constraint_index: constraint_index as u32,
                nodes,
            });
        }
        if !recovered_edges_exist(&topology, &recovered, &mut work)? {
            continue;
        }
        let steiner_node_identities = recovered
            .iter()
            .flat_map(|segment| segment.nodes.iter().map(|node| node.identity))
            .filter(|identity| !constraint_identities.contains(identity))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let canonical = topology
            .clone()
            .freeze(options.insertion.topology, cancellation)
            .map_err(mutation_error)?;
        let result = DelaunaySegmentRecovery {
            topology: canonical,
            segments: recovered,
            steiner_node_identities,
            recovery_passes: recovery_pass,
        };
        validate_delaunay_segment_recovery(&result, constraints, options, cancellation)?;
        return Ok(result);
    }
    Err(error(
        DelaunaySegmentRecoveryErrorKind::UnsatisfiableConstraint,
        None,
        "recovered edges did not remain simultaneous within the recovery-pass limit",
    ))
}

pub(in crate::cdt) fn try_recover_delaunay_edge_with_face_flip(
    topology: &DelaunayVolumeTopology,
    endpoints: [StableDigest; 2],
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    options: DelaunaySegmentRecoveryOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<DelaunayVolumeTopology>, DelaunaySegmentRecoveryError> {
    validate_options(options)?;
    let mut work = RecoveryWork::new(options, cancellation);
    try_recover_edge_with_face_flip(
        topology,
        endpoints[0],
        endpoints[1],
        protected_faces,
        constraint_index,
        &mut work,
    )
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

fn recover_interval(
    topology: &mut TopologyMutation,
    context: SegmentContext,
    left: DyadicNode,
    right: DyadicNode,
    depth: u8,
    output: &mut Vec<DelaunayRecoveredSegmentNode>,
    protected_faces: &[[StableDigest; 3]],
    work: &mut RecoveryWork<'_>,
) -> Result<(), DelaunaySegmentRecoveryError> {
    work.recovery_step(context.constraint_index)?;
    if edge_exists(
        topology,
        left.identity,
        right.identity,
        context.constraint_index,
        work,
    )? {
        push_chain_node(output, left);
        push_chain_node(output, right);
        return Ok(());
    }
    if try_recover_edge_with_face_flip_mutation(
        topology,
        left.identity,
        right.identity,
        protected_faces,
        context.constraint_index,
        work,
    )? {
        push_chain_node(output, left);
        push_chain_node(output, right);
        return Ok(());
    }
    if depth >= work.options.maximum_split_depth {
        let left_coordinates = find_node(topology, left.identity).map(|node| node.coordinates_m);
        let right_coordinates = find_node(topology, right.identity).map(|node| node.coordinates_m);
        return Err(error(
            DelaunaySegmentRecoveryErrorKind::UnsatisfiableConstraint,
            Some(context.constraint_index),
            format!(
                "segment {:?} terminal subinterval {:?} at {left_coordinates:?} -> {right_coordinates:?} did not recover within the dyadic split-depth limit",
                [context.first_identity, context.last_identity],
                [left.identity, right.identity],
            ),
        ));
    }
    let witness = encroaching_point(
        topology,
        left.identity,
        right.identity,
        context.constraint_index,
        work,
    )?
    .ok_or_else(|| {
        error(
            DelaunaySegmentRecoveryErrorKind::InvalidTopology,
            Some(context.constraint_index),
            "missing Delaunay segment has no encroaching vertex witness",
        )
    })?;
    let midpoint = split_parameter(topology, context, left, right, witness)?;
    let coordinates = interpolate(
        context.first_coordinates,
        context.last_coordinates,
        midpoint.parameter()?,
    );
    let identity = if let Some(existing) = topology
        .nodes()
        .iter()
        .find(|node| coordinates_share_representable_location(node.coordinates_m, coordinates))
    {
        existing.identity
    } else {
        let identity = steiner_identity(context, midpoint);
        if let Some(existing) = find_node(topology, identity) {
            if existing.coordinates_m != coordinates {
                return Err(error(
                    DelaunaySegmentRecoveryErrorKind::IdentityCollision,
                    Some(context.constraint_index),
                    "segment Steiner identity collides with a different node",
                ));
            }
        } else {
            work.inserted_node(context.constraint_index)?;
            let outcome = insert_delaunay_volume_node_into_mutation_with_barriers(
                topology,
                DelaunayVolumeNode {
                    identity,
                    coordinates_m: coordinates,
                },
                protected_faces,
                true,
                work.options.insertion,
                work.cancellation,
            )
            .map_err(|insertion| insertion_error(insertion, context.constraint_index))?;
            if outcome == MutableInsertionOutcome::RequiresCanonicalFallback {
                let current = topology
                    .clone()
                    .freeze(work.options.insertion.topology, work.cancellation)
                    .map_err(mutation_error)?;
                let updated = insert_delaunay_volume_node_with_barriers(
                    current,
                    DelaunayVolumeNode {
                        identity,
                        coordinates_m: coordinates,
                    },
                    protected_faces,
                    work.options.insertion,
                    work.cancellation,
                )
                .map_err(|insertion| insertion_error(insertion, context.constraint_index))?;
                *topology = TopologyMutation::from_canonical(updated);
            }
        }
        identity
    };
    let midpoint = midpoint.with_identity(identity);
    recover_interval(
        topology,
        context,
        left,
        midpoint,
        depth + 1,
        output,
        protected_faces,
        work,
    )?;
    recover_interval(
        topology,
        context,
        midpoint,
        right,
        depth + 1,
        output,
        protected_faces,
        work,
    )
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
    for tetrahedron_index in topology.vertex_star(left_index as u32) {
        work.search_step(constraint_index)?;
        if topology
            .tetrahedron(tetrahedron_index)
            .is_some_and(|tetrahedron| tetrahedron.vertex_indices.contains(&(right_index as u32)))
        {
            return Ok(true);
        }
    }
    Ok(false)
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
