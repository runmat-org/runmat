//! Bound/check original owned input closures before joining/decode allocations.
use super::*;

pub(super) fn require_source_decode_budget(
    prepared: &PreparedMaxwellHarmonicTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<()> {
    crate::require_maxwell_source_decode_budget(&prepared.inputs, &prepared.resource_plan, limits)
}
