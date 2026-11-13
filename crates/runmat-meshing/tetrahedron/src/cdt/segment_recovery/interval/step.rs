//! One admitted recovery operation; scheduling belongs to the caller.
use super::*;

pub(in crate::cdt::segment_recovery) fn recover_once<F>(
    topology: &mut TopologyMutation,
    left: DyadicNode,
    right: DyadicNode,
    acute_lineage: [bool; 2],
    depth: u8,
    output: &mut Vec<DelaunayRecoveredSegmentNode>,
    recovery: &mut IntervalRecoveryContext<'_, '_, F>,
) -> Result<Option<[ChildInterval; 2]>, DelaunaySegmentRecoveryError>
where
    F: FnMut(SegmentContext, DyadicNode, [f64; 3]) -> StableDigest,
{
    let context = recovery.segment;
    let work = &mut *recovery.work;
    work.recovery_step(context.constraint_index)?;
    if edge_exists(
        topology,
        left.identity,
        right.identity,
        context.constraint_index,
        work,
    )? {
        work.outcomes.present += 1;
        push_chain_node(output, left);
        push_chain_node(output, right);
        if let Some(depths) = recovery.completed_intervals.as_deref_mut() {
            depths.push(CompletedInterval {
                depth,
                acute_lineage,
            });
        }
        return Ok(None);
    }
    if local_reconnection::recover(
        topology,
        left.identity,
        right.identity,
        recovery.protected_edges,
        recovery.protected_faces,
        context.constraint_index,
        work,
    )? {
        work.outcomes.local_reconnection += 1;
        push_chain_node(output, left);
        push_chain_node(output, right);
        if let Some(depths) = recovery.completed_intervals.as_deref_mut() {
            depths.push(CompletedInterval {
                depth,
                acute_lineage,
            });
        }
        return Ok(None);
    }
    let witness = encroaching_point(
        topology,
        left.identity,
        right.identity,
        context.constraint_index,
        work,
    )?;
    if witness.is_none_or(|witness| witness.relation == PredicateSign::Zero)
        && try_recover_edge_with_face_flip(
            topology,
            left.identity,
            right.identity,
            recovery.protected_edges,
            recovery.protected_faces,
            context.constraint_index,
            work,
        )?
    {
        work.outcomes.face_flip += 1;
        push_chain_node(output, left);
        push_chain_node(output, right);
        if let Some(depths) = recovery.completed_intervals.as_deref_mut() {
            depths.push(CompletedInterval {
                depth,
                acute_lineage,
            });
        }
        return Ok(None);
    }
    if witness.is_none() && !recovery.split_strong_gabriel {
        return Err(error(
            DelaunaySegmentRecoveryErrorKind::InvalidTopology,
            Some(context.constraint_index),
            format!(
                "Delaunay topology omits strongly Gabriel segment {:?} at recovery depth {depth}",
                [left.identity, right.identity],
            ),
        ));
    }
    if witness.is_none() && !recovery.reported_unrecovered_interval {
        diagnostic::trace_unrecovered_interval(topology, context, left, right, depth);
        recovery.reported_unrecovered_interval = true;
    }
    if depth >= work.options.maximum_split_depth {
        let witness = witness.map(|witness| {
            let witness_node = topology.nodes()[witness.vertex as usize];
            (
                witness_node.identity,
                witness_node.coordinates_m.canonical_value(),
            )
        });
        return Err(diagnostic::split_limit_failure(
            topology, context, left, right, witness,
        ));
    }
    let midpoint = if let Some(witness) = witness {
        if witness.relation == PredicateSign::Negative {
            work.outcomes.interior_split += 1;
        } else {
            work.outcomes.boundary_split += 1;
        }
        work.outcomes.acute_split += u64::from(acute_lineage[0] || acute_lineage[1]);
        split_parameter(
            topology,
            context,
            left,
            right,
            witness.vertex,
            acute_lineage,
            work,
        )?
    } else {
        work.outcomes.absent_witness_split += 1;
        left.midpoint(right, context)?
    };
    let construction = Point3Construction::Segment {
        endpoints: [context.first_coordinates, context.last_coordinates],
        parameter_numerator: midpoint.numerator,
        parameter_exponent: midpoint.exponent,
    };
    let coordinates = interpolate(
        context.first_coordinates,
        context.last_coordinates,
        midpoint.parameter()?,
    );
    let derived_identity = (recovery.identity_for_node)(context, midpoint, coordinates);
    let identity = if let Some(existing) = find_node(topology, derived_identity) {
        if existing.construction != construction {
            return Err(error(
                DelaunaySegmentRecoveryErrorKind::IdentityCollision,
                Some(context.constraint_index),
                "segment Steiner identity collides with a different node",
            ));
        }
        derived_identity
    } else if let Some(identity) =
        location::find_representable_node(topology, coordinates, context.constraint_index, work)?
    {
        identity
    } else {
        work.inserted_node(context.constraint_index)?;
        let node = DelaunayVolumeNode {
            identity: derived_identity,
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(coordinates),
            construction,
        };
        if recovery.protected_faces.is_empty() {
            insert_delaunay_volume_node_into_mutation(
                topology,
                node,
                work.options.insertion,
                work.cancellation,
            )
        } else {
            insert_delaunay_volume_node_into_mutation_with_barriers(
                topology,
                node,
                recovery.protected_faces,
                true,
                work.options.insertion,
                work.cancellation,
            )
        }
        .map_err(|insertion| {
            let mut failure = insertion_error(insertion, context.constraint_index);
            failure.reason = format!(
                "inserting segment point {coordinates:?} on {:?}-{:?} ({} barrier faces): {}",
                context.first_coordinates,
                context.last_coordinates,
                recovery.protected_faces.len(),
                failure.reason
            );
            failure
        })?;
        derived_identity
    };
    let midpoint = midpoint.with_identity(identity);
    let (left_acute_lineage, right_acute_lineage) = if acute_lineage == [true, true] {
        ([true, false], [false, true])
    } else {
        (acute_lineage, acute_lineage)
    };
    Ok(Some([
        (left, midpoint, left_acute_lineage, depth + 1),
        (midpoint, right, right_acute_lineage, depth + 1),
    ]))
}
