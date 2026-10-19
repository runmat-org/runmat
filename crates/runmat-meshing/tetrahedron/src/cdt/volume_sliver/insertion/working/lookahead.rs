//! Original insertion depth and width, with checked branch-relative seed replay.
use super::*;
pub(super) fn search<'ctx>(
    state: &mut impl SearchState<'ctx>,
    frontier: Vec<WorkingSeed>,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut OptimizationWork<'_>,
    policy: SliverSearchPolicy,
    remaining_depth: u8,
) -> Result<WorkingInsertionSearch, DelaunayVolumeSliverError> {
    let mut result = WorkingInsertionSearch {
        candidate: None,
        rejection_counts: BTreeMap::new(),
        first_rejection_details: BTreeMap::new(),
    };
    for seed in frontier {
        let mut trial = state.trial();
        seed.replay(&mut trial, policy.options, cancellation)?;
        let Some(source) = trial
            .quality()
            .worst(policy.current_objective(
                trial.quality().worst(QualityObjective::Shape),
                trial.quality().worst(QualityObjective::Refinement),
            ))
            .cloned()
        else {
            continue;
        };
        let relocation = crate::cdt::volume_sliver::relocation::working::candidates(
            &mut trial,
            &source,
            cancellation,
            work,
            policy,
            &seed.mutations,
        )?;
        tracing::trace!(rejections = ?relocation.rejection_counts, "compound fixed-star relocation completed");
        if let Some(candidate) = relocation.candidate {
            result.candidate = Some(candidate);
            return Ok(result);
        }
        let next = candidates(
            &mut trial,
            &source,
            cancellation,
            work,
            policy,
            remaining_depth - 1,
            &seed.mutations,
        )?;
        if let Some(candidate) = next.candidate {
            result.candidate = Some(candidate);
            return Ok(result);
        }
        for (reason, count) in next.rejection_counts {
            *result.rejection_counts.entry(reason).or_default() += count;
        }
        for (reason, detail) in next.first_rejection_details {
            result
                .first_rejection_details
                .entry(reason)
                .or_insert(detail);
        }
    }
    Ok(result)
}
