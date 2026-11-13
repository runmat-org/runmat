//! Original depth/width and candidate order; seed replay touches local edits.
use super::*;
use crate::cdt::volume_quality::QualityObjective;
use crate::cdt::volume_sliver::working::CaptureSeed;
pub(super) fn search<'ctx>(
    state: &mut impl SearchState<'ctx>,
    mut frontier: Vec<WorkingSeed>,
    refinement_baseline: &SpectrumDelta,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut OptimizationWork<'_>,
    search_policy: SearchPolicy,
    rejection_counts: &mut BTreeMap<&'static str, u64>,
) -> Result<Option<WorkingSeed>, DelaunayVolumeSliverError> {
    let SearchPolicy {
        sliver: policy,
        continuation,
        deferred: _,
    } = search_policy;
    const MAXIMUM_LOOKAHEAD_DEPTH: usize = 4;
    const MAXIMUM_LOOKAHEAD_WIDTH: usize = 8;
    if matches!(
        policy.refinement,
        crate::cdt::volume_sliver::admission::CandidateRefinementPolicy::AllowForCoupledConvergence
    ) {
        // The caller already ranked first-step deltas against the original
        // owner. Apply the declared beam width before visiting those seeds too.
        frontier.truncate(MAXIMUM_LOOKAHEAD_WIDTH);
    }
    for _depth in 2..=MAXIMUM_LOOKAHEAD_DEPTH {
        let mut next_frontier =
            crate::cdt::volume_sliver::frontier::RankedFrontier::new(MAXIMUM_LOOKAHEAD_WIDTH);
        let mut seen = BTreeSet::new();
        for seed in frontier {
            let mut current = state.trial();
            seed.replay(&mut current, policy.options, cancellation)?;
            let Some(source) = current
                .quality()
                .worst(policy.current_objective(
                    current.quality().worst(QualityObjective::Shape),
                    current.quality().worst(QualityObjective::Refinement),
                ))
                .cloned()
            else {
                continue;
            };
            let index = source_index(current.topology(), &source)?;
            for candidate in
                local_candidates(current.topology(), index as usize, source.resolved_metric)?
            {
                work.evaluate()?;
                let mutation =
                    reconnection_mutation(current.topology(), &candidate, source.node_identities);
                let mut trial = current.trial();
                if !apply(&mut trial, &candidate, policy.options, cancellation)? {
                    continue;
                }
                let mut mutations = seed.mutations.clone();
                mutations.push(mutation);
                if crate::cdt::volume_sliver::admission::working_quality_is_admissible(
                    trial.quality(),
                    refinement_baseline,
                    policy.refinement,
                    policy.objective,
                    rejection_counts,
                ) {
                    return Ok(Some(trial.capture(mutations)));
                }
                let retained = trial.capture(mutations);
                let key = retained.connectivity_key();
                if seen.insert((key.0.clone(), key.1.clone())) {
                    next_frontier.retain(
                        retained,
                        |candidate: &WorkingSeed, entry: &WorkingSeed| {
                            candidate.compare_policy(entry, policy).then_with(|| {
                                candidate.compare_connectivity(entry, trial.quality())
                            })
                        },
                    );
                }
            }
            if matches!(continuation, Continuation::ConnectivityOnly) {
                continue;
            }
            let relocation = crate::cdt::volume_sliver::relocation::working::candidates(
                &mut current,
                &source,
                cancellation,
                work,
                policy,
                &seed.mutations,
            )?;
            tracing::trace!(rejections = ?relocation.rejection_counts, "compound fixed-star relocation completed");
            if let Some(candidate) = relocation.candidate {
                return Ok(Some(candidate));
            }
        }
        frontier = next_frontier.into_entries();
        if frontier.is_empty() {
            break;
        }
    }
    Ok(None)
}
