//! Recursive physical interval recovery, including retained split lineage.
use super::*;

#[derive(Clone, Copy)]
pub(super) struct CompletedInterval {
    pub(super) depth: u8,
    pub(super) acute_lineage: [bool; 2],
}

pub(super) struct IntervalRecoveryContext<'a, 'w, F> {
    pub(super) segment: SegmentContext,
    pub(super) protected_edges: &'a [[StableDigest; 2]],
    pub(super) protected_faces: &'a [[StableDigest; 3]],
    pub(super) split_strong_gabriel: bool,
    pub(super) reported_unrecovered_interval: bool,
    pub(super) identity_for_node: &'a mut F,
    pub(super) completed_intervals: Option<&'a mut Vec<CompletedInterval>>,
    pub(super) work: &'a mut RecoveryWork<'w>,
}

pub(super) type ChildInterval = (DyadicNode, DyadicNode, [bool; 2], u8);
mod step;
pub(super) use step::recover_once;

pub(super) fn recover_interval<F>(
    topology: &mut TopologyMutation,
    left: DyadicNode,
    right: DyadicNode,
    acute_lineage: [bool; 2],
    depth: u8,
    output: &mut Vec<DelaunayRecoveredSegmentNode>,
    recovery: &mut IntervalRecoveryContext<'_, '_, F>,
) -> Result<(), DelaunaySegmentRecoveryError>
where
    F: FnMut(SegmentContext, DyadicNode, [f64; 3]) -> StableDigest,
{
    if let Some(children) = recover_once(
        topology,
        left,
        right,
        acute_lineage,
        depth,
        output,
        recovery,
    )? {
        for (left, right, lineage, depth) in children {
            recover_interval(topology, left, right, lineage, depth, output, recovery)?;
        }
    }
    Ok(())
}
