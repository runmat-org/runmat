use std::collections::BTreeSet;

use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::{
    insert_delaunay_volume_node, validate_delaunay_constraints, validate_delaunay_volume_topology,
    DelaunayConstraintError, DelaunayConstraintErrorKind, DelaunayConstraintOptions,
    DelaunayConstraints, DelaunayInsertionError, DelaunayInsertionErrorKind,
    DelaunayInsertionOptions, DelaunayVolumeNode, DelaunayVolumeTopology,
};

mod flip;
mod parameter;
mod validation;
mod work;

use flip::try_recover_edge_with_face_flip;
use parameter::{interpolate, steiner_identity, DyadicNode, SegmentContext};
pub use validation::validate_delaunay_segment_recovery;
pub(in crate::cdt) use validation::validate_delaunay_segment_recovery_with_protected_faces;
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
            maximum_recovery_passes: 8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelaunayRecoveredSegmentNode {
    pub identity: StableDigest,
    pub parameter_numerator: u64,
    pub parameter_exponent: u8,
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
    let mut topology = Some(topology);
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
            let mut nodes = Vec::new();
            recover_interval(
                &mut topology,
                context,
                DyadicNode::endpoint(first.identity, false),
                DyadicNode::endpoint(last.identity, true),
                0,
                &mut nodes,
                &mut work,
            )?;
            recovered.push(DelaunayRecoveredSegment {
                constraint_index: constraint_index as u32,
                nodes,
            });
        }
        if !recovered_edges_exist(topology_ref(&topology)?, &recovered, &mut work)? {
            continue;
        }
        let steiner_node_identities = recovered
            .iter()
            .flat_map(|segment| segment.nodes.iter().map(|node| node.identity))
            .filter(|identity| !constraint_identities.contains(identity))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let result = DelaunaySegmentRecovery {
            topology: take_topology(&mut topology)?,
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

fn recovered_edges_exist(
    topology: &DelaunayVolumeTopology,
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
    topology: &mut Option<DelaunayVolumeTopology>,
    context: SegmentContext,
    left: DyadicNode,
    right: DyadicNode,
    depth: u8,
    output: &mut Vec<DelaunayRecoveredSegmentNode>,
    work: &mut RecoveryWork<'_>,
) -> Result<(), DelaunaySegmentRecoveryError> {
    work.recovery_step(context.constraint_index)?;
    if edge_exists(
        topology_ref(topology)?,
        left.identity,
        right.identity,
        context.constraint_index,
        work,
    )? {
        push_chain_node(output, left);
        push_chain_node(output, right);
        return Ok(());
    }
    if let Some(updated) = try_recover_edge_with_face_flip(
        topology_ref(topology)?,
        left.identity,
        right.identity,
        context.constraint_index,
        work,
    )? {
        *topology = Some(updated);
        push_chain_node(output, left);
        push_chain_node(output, right);
        return Ok(());
    }
    if depth >= work.options.maximum_split_depth {
        return Err(error(
            DelaunaySegmentRecoveryErrorKind::UnsatisfiableConstraint,
            Some(context.constraint_index),
            "segment did not recover within the dyadic split-depth limit",
        ));
    }
    let midpoint = left.midpoint(right, context)?;
    let coordinates = interpolate(
        context.first_coordinates,
        context.last_coordinates,
        midpoint.parameter()?,
    );
    let identity = if let Some(existing) = topology_ref(topology)?
        .nodes
        .iter()
        .find(|node| node.coordinates_m == coordinates)
    {
        existing.identity
    } else {
        let identity = steiner_identity(context, midpoint);
        if let Some(existing) = find_node(topology_ref(topology)?, identity) {
            if existing.coordinates_m != coordinates {
                return Err(error(
                    DelaunaySegmentRecoveryErrorKind::IdentityCollision,
                    Some(context.constraint_index),
                    "segment Steiner identity collides with a different node",
                ));
            }
        } else {
            work.inserted_node(context.constraint_index)?;
            let current = take_topology(topology)?;
            let updated = insert_delaunay_volume_node(
                current,
                DelaunayVolumeNode {
                    identity,
                    coordinates_m: coordinates,
                },
                work.options.insertion,
                work.cancellation,
            )
            .map_err(|insertion| insertion_error(insertion, context.constraint_index))?;
            *topology = Some(updated);
        }
        identity
    };
    let midpoint = midpoint.with_identity(identity);
    recover_interval(topology, context, left, midpoint, depth + 1, output, work)?;
    recover_interval(topology, context, midpoint, right, depth + 1, output, work)
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
    topology: &DelaunayVolumeTopology,
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
    for tetrahedron_index in &topology.incidence.vertex_stars[left_index] {
        work.search_step(constraint_index)?;
        if topology.tetrahedra[*tetrahedron_index as usize]
            .vertex_indices
            .contains(&(right_index as u32))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn find_node(
    topology: &DelaunayVolumeTopology,
    identity: StableDigest,
) -> Option<&DelaunayVolumeNode> {
    node_index(topology, identity).map(|index| &topology.nodes[index])
}

fn node_index(topology: &DelaunayVolumeTopology, identity: StableDigest) -> Option<usize> {
    topology
        .nodes
        .binary_search_by_key(&identity, |node| node.identity)
        .ok()
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

fn topology_ref(
    topology: &Option<DelaunayVolumeTopology>,
) -> Result<&DelaunayVolumeTopology, DelaunaySegmentRecoveryError> {
    topology.as_ref().ok_or_else(|| {
        error(
            DelaunaySegmentRecoveryErrorKind::InvalidTopology,
            None,
            "segment recovery lost ownership of its topology",
        )
    })
}

fn take_topology(
    topology: &mut Option<DelaunayVolumeTopology>,
) -> Result<DelaunayVolumeTopology, DelaunaySegmentRecoveryError> {
    topology.take().ok_or_else(|| {
        error(
            DelaunaySegmentRecoveryErrorKind::InvalidTopology,
            None,
            "segment recovery lost ownership of its topology",
        )
    })
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
