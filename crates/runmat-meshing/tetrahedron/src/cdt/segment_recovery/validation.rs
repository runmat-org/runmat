use std::collections::BTreeSet;

use runmat_meshing_core::MeshingCancellationSignal;

use super::{
    coordinates_share_representable_location, edge_exists, error, find_node, interpolate,
    validate_delaunay_constraints, validate_options, DelaunayConstraints, DelaunaySegmentRecovery,
    DelaunaySegmentRecoveryError, DelaunaySegmentRecoveryErrorKind, DelaunaySegmentRecoveryOptions,
    DelaunayVolumeTopology, RecoveryWork,
};
use crate::cdt::insertion::validate_constrained_delaunay_volume_topology;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SegmentRecoveryInputContract {
    ExternalUnconstrainedTopology,
    CertifiedPointSet,
}

pub fn validate_delaunay_segment_recovery(
    recovery: &DelaunaySegmentRecovery,
    constraints: &DelaunayConstraints,
    options: DelaunaySegmentRecoveryOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunaySegmentRecoveryError> {
    validate_delaunay_segment_recovery_on_topology(
        recovery,
        &recovery.topology,
        constraints,
        &[],
        options,
        cancellation,
    )
}

pub(in crate::cdt) fn validate_delaunay_segment_recovery_on_topology(
    recovery: &DelaunaySegmentRecovery,
    topology: &DelaunayVolumeTopology,
    constraints: &DelaunayConstraints,
    protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
    options: DelaunaySegmentRecoveryOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunaySegmentRecoveryError> {
    validate_options(options)?;
    validate_delaunay_constraints(constraints, options.constraints, cancellation)
        .map_err(super::constraint_error)?;
    // A recovered segment is already a constraint, even before any facets are protected.
    // Validate the local constrained-Delaunay topology here; the unconstrained prerequisite was
    // validated before recovery and its global empty-sphere property is neither required nor, in
    // general, preserved by constraint recovery.
    validate_constrained_delaunay_volume_topology(
        topology,
        protected_faces,
        options.insertion,
        cancellation,
    )
    .map_err(|validation| super::insertion_error(validation, 0))?;
    validate_constraint_nodes(topology, constraints)?;
    if recovery.recovery_passes == 0
        || recovery.recovery_passes > options.maximum_recovery_passes
        || recovery.segments.len() != constraints.segments.len()
        || recovery
            .steiner_node_identities
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
    {
        return Err(error(
            DelaunaySegmentRecoveryErrorKind::InvalidConstraints,
            None,
            "recovery evidence counts, passes, or inserted-node ordering are invalid",
        ));
    }
    let mut work = RecoveryWork::new(options, cancellation);
    for (expected_index, recovered) in recovery.segments.iter().enumerate() {
        let segment = &constraints.segments[expected_index];
        let first = &constraints.nodes[segment.vertex_indices[0] as usize];
        let last = &constraints.nodes[segment.vertex_indices[1] as usize];
        if recovered.constraint_index != expected_index as u32
            || recovered.nodes.len() < 2
            || recovered.nodes.first().map(|node| node.identity) != Some(first.identity)
            || recovered.nodes.last().map(|node| node.identity) != Some(last.identity)
        {
            return Err(error(
                DelaunaySegmentRecoveryErrorKind::InvalidConstraints,
                Some(expected_index as u32),
                "recovered chain does not bind to its ordered constraint endpoints",
            ));
        }
        let mut previous_parameter = None;
        for node in &recovered.nodes {
            let parameter = node.parameter()?;
            if previous_parameter.is_some_and(|previous| previous >= parameter) {
                return Err(error(
                    DelaunaySegmentRecoveryErrorKind::InvalidConstraints,
                    Some(expected_index as u32),
                    "recovered chain parameters are not strictly increasing",
                ));
            }
            let topology_node = find_node(topology, node.identity).ok_or_else(|| {
                error(
                    DelaunaySegmentRecoveryErrorKind::InvalidTopology,
                    Some(expected_index as u32),
                    "recovered chain references a missing topology node",
                )
            })?;
            let endpoint = node.identity == first.identity || node.identity == last.identity;
            let expected_coordinates = interpolate(
                first.coordinates_m.canonical_value(),
                last.coordinates_m.canonical_value(),
                parameter,
            );
            if !endpoint
                && !coordinates_share_representable_location(
                    topology_node.coordinates_m.canonical_value(),
                    expected_coordinates,
                )
            {
                return Err(error(
                    DelaunaySegmentRecoveryErrorKind::InvalidTopology,
                    Some(expected_index as u32),
                    "recovered chain node is not at the representable location of its declared dyadic segment parameter",
                ));
            }
            previous_parameter = Some(parameter);
        }
        for pair in recovered.nodes.windows(2) {
            if !edge_exists(
                topology,
                pair[0].identity,
                pair[1].identity,
                expected_index as u32,
                &mut work,
            )? {
                return Err(error(
                    DelaunaySegmentRecoveryErrorKind::InvalidTopology,
                    Some(expected_index as u32),
                    "recovered chain contains a missing tetrahedron edge",
                ));
            }
        }
    }
    validate_steiner_evidence(recovery, topology, constraints)
}

pub(super) fn validate_inputs(
    topology: DelaunayVolumeTopology,
    constraints: &DelaunayConstraints,
    options: DelaunaySegmentRecoveryOptions,
    cancellation: &dyn MeshingCancellationSignal,
    input_contract: SegmentRecoveryInputContract,
) -> Result<crate::cdt::insertion::AdmittedPhysicalTopology, DelaunaySegmentRecoveryError> {
    validate_delaunay_constraints(constraints, options.constraints, cancellation)
        .map_err(super::constraint_error)?;
    let admitted = match input_contract {
        SegmentRecoveryInputContract::ExternalUnconstrainedTopology => {
            crate::cdt::insertion::AdmittedPhysicalTopology::verify_global(
                topology,
                options.insertion,
                cancellation,
            )
        }
        SegmentRecoveryInputContract::CertifiedPointSet => {
            crate::cdt::insertion::AdmittedPhysicalTopology::verify(
                topology,
                options.insertion,
                cancellation,
            )
        }
    }
    .map_err(|validation| super::insertion_error(validation, 0))?;
    validate_constraint_nodes(admitted.topology(), constraints)?;
    Ok(admitted)
}

fn validate_constraint_nodes(
    topology: &DelaunayVolumeTopology,
    constraints: &DelaunayConstraints,
) -> Result<(), DelaunaySegmentRecoveryError> {
    for constraint_node in &constraints.nodes {
        let topology_node = find_node(topology, constraint_node.identity).ok_or_else(|| {
            error(
                DelaunaySegmentRecoveryErrorKind::InvalidTopology,
                None,
                "constraint node is missing from topology",
            )
        })?;
        if topology_node.coordinates_m.canonical_value()
            != constraint_node.coordinates_m.canonical_value()
        {
            return Err(error(
                DelaunaySegmentRecoveryErrorKind::InvalidTopology,
                None,
                "constraint node coordinates disagree with topology",
            ));
        }
    }
    Ok(())
}

fn validate_steiner_evidence(
    recovery: &DelaunaySegmentRecovery,
    topology: &DelaunayVolumeTopology,
    constraints: &DelaunayConstraints,
) -> Result<(), DelaunaySegmentRecoveryError> {
    let reported = recovery
        .steiner_node_identities
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let constraint_identities = constraints
        .nodes
        .iter()
        .map(|node| node.identity)
        .collect::<BTreeSet<_>>();
    let expected = recovery
        .segments
        .iter()
        .flat_map(|segment| segment.nodes.iter().map(|node| node.identity))
        .filter(|identity| !constraint_identities.contains(identity))
        .collect::<BTreeSet<_>>();
    if reported != expected
        || reported
            .iter()
            .any(|identity| find_node(topology, *identity).is_none())
    {
        return Err(error(
            DelaunaySegmentRecoveryErrorKind::InvalidConstraints,
            None,
            "Steiner-node evidence does not exactly match the recovered chains",
        ));
    }
    Ok(())
}
