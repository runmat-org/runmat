//! Admission estimate for serial metadata-only task reconstruction, not measured allocator peak.
use super::*;
use crate::results::field::RecoveredFieldPartitionDescriptor;
use runmat_canonical_codec::profile_serialized;

impl ObservationPublicationCatalog {
    pub(crate) fn execution_workspace_bytes_estimate(&self) -> AnalysisExecutionResult<u64> {
        let mut records = self.execution.iter();
        let mut peak = 0;
        self.visit_results(|source, observation, result, evidence| {
            let record = records
                .next()
                .ok_or_else(|| invalid("execution omits workspace source"))?;
            if record.result != result {
                return Err(invalid("execution workspace source order differs"));
            }
            let plan = self.execution_plan(source, observation, evidence)?;
            let field = source_field(source, &plan, evidence)?;
            // These shared profiles cover retained owned request/plan metadata and their
            // canonical value trees, plus source-set and contribution-request encoding.
            // They are structural estimates; no numerical field payload is reconstructed.
            // Request identity validation rederives one expected plan while this plan and
            // every source selection remain live. Charge both owners explicitly.
            let plan_bytes = profile_serialized(&plan, PROFILE_LIMITS)
                .map_err(invalid)?
                .materialization_resident_bytes_estimate;
            let mut bytes = add(plan_bytes, plan_bytes)?;
            for profile in [
                profile_serialized(source, PROFILE_LIMITS),
                profile_serialized(field, PROFILE_LIMITS),
                profile_serialized(evidence, PROFILE_LIMITS),
            ] {
                bytes = add(
                    bytes,
                    profile
                        .map_err(invalid)?
                        .materialization_resident_bytes_estimate,
                )?;
            }
            let group_ids = record
                .source_partitions
                .iter()
                .try_fold(0usize, |sum, group| {
                    sum.checked_add(group.len())
                        .ok_or_else(|| invalid("execution group count overflow"))
                })?;
            let largest_group = record
                .source_partitions
                .iter()
                .map(Vec::len)
                .max()
                .unwrap_or(0);
            bytes = add(
                bytes,
                scratch_bytes(
                    group_ids,
                    largest_group,
                    record.source_partitions.len(),
                    field.partitions.len(),
                    evidence.contributions.len(),
                )?,
            )?;
            peak = peak.max(bytes);
            Ok(())
        })?;
        if records.next().is_some() {
            return Err(invalid("execution has extra workspace sources"));
        }
        Ok(peak)
    }
}

// Capacities reserved explicitly by visit_execution and the shared storage selector. They
// deliberately sum construction and callback buffers even where a temporary drops earlier.
fn scratch_bytes(
    groups: usize,
    largest: usize,
    partitions: usize,
    descriptors: usize,
    contributions: usize,
) -> AnalysisExecutionResult<u64> {
    let mut total = 0;
    for (count, width) in [
        (
            groups,
            std::mem::size_of::<RecoveredFieldPartitionIdentity>(),
        ),
        (
            groups,
            std::mem::size_of::<&RecoveredFieldPartitionDescriptor>(),
        ),
        (largest, std::mem::size_of::<usize>()),
        (
            partitions,
            std::mem::size_of::<RecoveredFieldPartitionSelection<'_>>(),
        ),
        (
            descriptors,
            std::mem::size_of::<&RecoveredFieldPartitionDescriptor>(),
        ),
        (
            contributions,
            std::mem::size_of::<ObservationPartitionContributionIdentity>(),
        ),
    ] {
        let bytes = count
            .checked_mul(width)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| invalid("execution workspace size overflow"))?;
        total = add(total, bytes)?;
    }
    Ok(total)
}

fn add(left: u64, right: u64) -> AnalysisExecutionResult<u64> {
    left.checked_add(right)
        .ok_or_else(|| invalid("execution workspace sum overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_counts_all_live_groups_not_just_unique_union() {
        let small = scratch_bytes(2, 1, 2, 3, 2).unwrap();
        let shared_blocks = scratch_bytes(4, 2, 2, 3, 2).unwrap();
        let expected_growth = 2
            * (std::mem::size_of::<RecoveredFieldPartitionIdentity>()
                + std::mem::size_of::<&RecoveredFieldPartitionDescriptor>())
            + std::mem::size_of::<usize>();
        assert_eq!(shared_blocks - small, expected_growth as u64);
        assert!(scratch_bytes(4, 2, 2, 100, 2).unwrap() > shared_blocks);
        assert_eq!(
            scratch_bytes(4, 2, 2, 3, 3).unwrap() - shared_blocks,
            std::mem::size_of::<ObservationPartitionContributionIdentity>() as u64,
        );
    }

    #[test]
    fn scratch_rejects_overflow() {
        assert!(scratch_bytes(usize::MAX, 0, 0, 0, 0).is_err());
        assert!(scratch_bytes(0, 0, 0, 0, usize::MAX).is_err());
        assert!(add(u64::MAX, 1).is_err());
    }
}
