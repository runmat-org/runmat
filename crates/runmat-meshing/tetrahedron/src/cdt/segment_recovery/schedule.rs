//! Fair breadth-first constraint recovery. Children wait behind the other
//! physical constraints instead of recursively refining one ray in isolation.
//! Complete simultaneous edge validation remains authoritative after each pass.
use super::*;
use std::collections::VecDeque;

mod queue;
use queue::IntervalQueue;

pub(super) fn recover_pass(
    topology: &mut TopologyMutation,
    constraints: &DelaunayConstraints,
    acute: &[bool],
    recovered: &mut [DelaunayRecoveredSegment],
    retained: &mut [Vec<interval::CompletedInterval>],
    recovery_pass: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<(), DelaunaySegmentRecoveryError> {
    let mut pending = IntervalQueue::default();
    for (constraint_index, segment) in constraints.segments.iter().enumerate() {
        work.search_step(constraint_index as u32)?;
        let previous = std::mem::take(&mut recovered[constraint_index].nodes);
        let lineage = std::mem::take(&mut retained[constraint_index]);
        if previous.is_empty() {
            let [first, last] = segment.vertex_indices;
            pending.insert(
                constraint_index,
                (
                    DyadicNode::endpoint(constraints.nodes[first as usize].identity, false),
                    DyadicNode::endpoint(constraints.nodes[last as usize].identity, true),
                    [acute[first as usize], acute[last as usize]],
                    0,
                ),
                topology,
            )?;
        } else {
            if lineage.len() + 1 != previous.len() {
                return Err(error(
                    DelaunaySegmentRecoveryErrorKind::InvalidTopology,
                    Some(constraint_index as u32),
                    "retained segment chain has inconsistent split lineage",
                ));
            }
            for (pair, lineage) in previous.windows(2).zip(lineage) {
                let [left, right] = [pair[0], pair[1]].map(|node| DyadicNode {
                    identity: node.identity,
                    numerator: node.parameter_numerator,
                    exponent: node.parameter_exponent,
                });
                pending.insert(
                    constraint_index,
                    (left, right, lineage.acute_lineage, lineage.depth),
                    topology,
                )?;
            }
        }
    }
    let mut operations = 0_u64;
    let mut scratch = Vec::with_capacity(2);
    while let Some((index, constraint_index, (left, right, lineage, depth))) = pending.pop() {
        operations += 1;
        if operations.is_multiple_of(4096) {
            tracing::info!(target: "runmat::meshing::segment_recovery", recovery_pass,
                constraint_index, operations, pending_intervals = pending.pending_len(),
                node_count = topology.nodes().len(), tetrahedron_count = topology.active_tetrahedron_count(),
                "segment queue progress");
        }
        let segment = &constraints.segments[constraint_index];
        let [first, last] = segment
            .vertex_indices
            .map(|index| &constraints.nodes[index as usize]);
        let context = SegmentContext {
            constraint_index: constraint_index as u32,
            first_identity: first.identity,
            last_identity: last.identity,
            first_coordinates: first.coordinates_m.canonical_value(),
            last_coordinates: last.coordinates_m.canonical_value(),
        };
        let mut identity_for_node = |context, midpoint, _| steiner_identity(context, midpoint);
        scratch.clear();
        let mut recovery = IntervalRecoveryContext {
            segment: context,
            protected_edges: &[],
            protected_faces: &[],
            split_strong_gabriel: true,
            reported_unrecovered_interval: false,
            completed_intervals: None,
            identity_for_node: &mut identity_for_node,
            work,
        };
        let mut trial = topology.observe_edge_changes();
        let children = interval::recover_once(
            &mut trial,
            left,
            right,
            lineage,
            depth,
            &mut scratch,
            &mut recovery,
        )?;
        let changed = trial
            .removed_edge_candidates(&mut || recovery.work.search_step(constraint_index as u32))?;
        drop(trial);
        if let Some(children) = children {
            pending.retire(index);
            for child in children {
                pending.insert(constraint_index, child, topology)?;
            }
        }
        pending.requeue_lost(changed, topology, work)?;
    }
    let mut completed = pending.into_completed(constraints.segments.len());
    for (constraint_index, leaves) in completed.iter_mut().enumerate() {
        // Exact dyadic order, including parameters too close for f64 ordering.
        leaves.sort_unstable_by_key(|(left, _, _, _)| {
            u128::from(left.numerator) << (62 - left.exponent)
        });
        for &(left, right, acute_lineage, depth) in leaves.iter() {
            work.search_step(constraint_index as u32)?;
            push_chain_node(&mut recovered[constraint_index].nodes, left);
            push_chain_node(&mut recovered[constraint_index].nodes, right);
            retained[constraint_index].push(interval::CompletedInterval {
                depth,
                acute_lineage,
            });
        }
    }
    Ok(())
}
