//! Expensive graph and cluster work runs once per ranked local geometry seed,
//! outside line-search backtracks. Every seed retains the original quality owner.
use super::*;
pub(in crate::cdt::volume_sliver::relocation::cavity) fn search<'ctx>(
    state: &mut impl SearchState<'ctx>,
    seeds: Vec<WorkingSeed>,
    policy: SliverSearchPolicy,
    work: &mut OptimizationWork<'_>,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<WorkingSeed>, DelaunayVolumeSliverError> {
    let mut best = None;
    for seed in seeds {
        let mut trial = state.trial();
        seed.replay(&mut trial, policy.options, cancellation)?;
        let objective = policy.current_objective(
            trial
                .quality()
                .worst(crate::cdt::volume_quality::QualityObjective::Shape),
            trial
                .quality()
                .worst(crate::cdt::volume_quality::QualityObjective::Refinement),
        );
        let Some(source) = trial.quality().worst(objective).cloned() else {
            continue;
        };
        let connected = crate::cdt::volume_sliver::reconnection::working::connectivity_candidates(
            &mut trial,
            &source,
            cancellation,
            work,
            policy,
            &seed.mutations,
        )?;
        if let Some(candidate) = connected.candidate {
            WorkingSeed::retain_better(&mut best, candidate, policy);
        }
        if let Some(candidate) = crate::cdt::volume_sliver::relocation::block::candidate(
            &mut trial,
            &source,
            cancellation,
            work,
            policy,
            &seed.mutations,
        )? {
            WorkingSeed::retain_better(&mut best, candidate, policy);
        }
    }
    Ok(best)
}
