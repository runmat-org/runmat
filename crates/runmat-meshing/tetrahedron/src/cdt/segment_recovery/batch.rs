//! Complete simultaneous segment recovery, topology admission and chain publication.
use super::*;

pub(super) fn recover_delaunay_segments_with_contract(
    topology: DelaunayVolumeTopology,
    constraints: &DelaunayConstraints,
    options: DelaunaySegmentRecoveryOptions,
    cancellation: &dyn MeshingCancellationSignal,
    input_contract: SegmentRecoveryInputContract,
) -> Result<DelaunaySegmentRecovery, DelaunaySegmentRecoveryError> {
    validate_options(options)?;
    let topology = validate_inputs(topology, constraints, options, cancellation, input_contract)?;
    let constraint_identities = constraints
        .nodes
        .iter()
        .map(|node| node.identity)
        .collect::<BTreeSet<_>>();
    let acute_vertices = classify_acute_vertices(constraints, options, cancellation)?;
    let mut topology = TopologyMutation::from_admitted(topology);
    let mut work = RecoveryWork::new(options, cancellation);
    // Retain discovered physical chains between passes. Rebuilding every
    // original interval discards this information and repeats wide queries
    // over an increasingly dense point set, even when most fragments survive.
    let mut recovered = (0..constraints.segments.len())
        .map(|index| DelaunayRecoveredSegment {
            constraint_index: index as u32,
            nodes: Vec::new(),
        })
        .collect::<Vec<_>>();
    let mut retained_intervals =
        vec![Vec::<interval::CompletedInterval>::new(); constraints.segments.len()];

    for recovery_pass in 1..=options.maximum_recovery_passes {
        super::schedule::recover_pass(
            &mut topology,
            constraints,
            &acute_vertices,
            &mut recovered,
            &mut retained_intervals,
            recovery_pass,
            &mut work,
        )?;
        // Insertion legalizes its new-node neighborhood. Recheck the complete
        // batch before admitting simultaneous chains, including faces outside
        // the last insertion's star. The independent edge inventory remains
        // authoritative after repair.
        let canonical = topology
            .validated_snapshot(options.insertion.topology, cancellation)
            .map_err(mutation_error)?;
        let canonical = candidate::restore(canonical, &[], 0, &mut work)?.ok_or_else(|| {
            error(
                DelaunaySegmentRecoveryErrorKind::InvalidTopology,
                None,
                "segment batch could not restore constrained Delaunay admission",
            )
        })?;
        let admitted = crate::cdt::insertion::AdmittedPhysicalTopology::verify(
            canonical,
            options.insertion,
            cancellation,
        )
        .map_err(|failure| insertion_error(failure, 0))?;
        topology.adopt_admitted_with_spatial_index(admitted, &mut || {
            work.search_point_handoff_step(0)
        })?;
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
