//! Exploration uses the already admitted, bounded partition frontier. It does
//! not manufacture candidates or replace the placement cost/resource policy.

use std::collections::BTreeMap;

use runmat_execution::{
    CandidateCostUnavailable, CandidateObservationEvidence, Digest, ExecutionCandidateKind,
    PlacementPlanRequest,
};

use super::{
    alternate_is_transactional, evidence, feedback_candidate_key, within_exploration_bound,
    FeedbackCandidateKey, PlacementFeedbackSeries, PlacementSessionConfig,
};
use crate::placement::partition::PartitionPlan;

#[cfg(test)]
mod tests;

pub(super) fn choose_alternate<'a>(
    request: &PlacementPlanRequest,
    partition: &'a PartitionPlan,
    config: PlacementSessionConfig,
    signature: Digest,
    feedback: &BTreeMap<
        (Digest, FeedbackCandidateKey, ExecutionCandidateKind),
        PlacementFeedbackSeries,
    >,
    managed: Option<&super::managed_feedback::ManagedFeedbackPool>,
    tick: u64,
) -> Option<(&'a [usize], &'a u64)> {
    partition
        .alternates
        .iter()
        .enumerate()
        .filter(|(_, (selections, cost))| {
            alternate_is_transactional(request, &partition.selections, selections)
                && within_exploration_bound(
                    partition.total_ns,
                    *cost,
                    config.exploration_max_overhead_basis_points,
                )
        })
        .filter_map(|(order, (selections, cost))| {
            let need = least_observed_changed_candidate(
                request,
                &partition.selections,
                selections,
                signature,
                feedback,
                managed,
                tick,
            )?;
            Some((order, selections, cost, need))
        })
        .min_by_key(|(order, _, _, need)| (*need, *order))
        .map(|(_, selections, cost, _)| (selections.as_slice(), cost))
}

fn least_observed_changed_candidate(
    request: &PlacementPlanRequest,
    best: &[usize],
    alternate: &[usize],
    signature: Digest,
    feedback: &BTreeMap<
        (Digest, FeedbackCandidateKey, ExecutionCandidateKind),
        PlacementFeedbackSeries,
    >,
    managed: Option<&super::managed_feedback::ManagedFeedbackPool>,
    tick: u64,
) -> Option<u64> {
    let mut least = None;
    for (node, candidate) in alternate.iter().enumerate() {
        if *candidate != best[node] {
            let descriptor = &request.graph.nodes[node].candidates[*candidate].descriptor;
            let key = feedback_candidate_key(&descriptor.identity);
            let observed = match key {
                FeedbackCandidateKey::Canonical(candidate) => managed
                    .and_then(|pool| pool.observed(signature, candidate, descriptor.kind, tick)),
                FeedbackCandidateKey::Text(_) => None,
            }
            .or_else(|| {
                feedback
                    .get(&(signature, key, descriptor.kind))
                    .map(|series| evidence::from_series(series, tick))
            });
            let need = match observed {
                Some(CandidateObservationEvidence::Observed(_)) => continue,
                Some(CandidateObservationEvidence::Unavailable(
                    CandidateCostUnavailable::InsufficientObservations { successful, .. },
                )) => successful.min(2),
                None
                | Some(CandidateObservationEvidence::Unavailable(
                    CandidateCostUnavailable::NoObservations
                    | CandidateCostUnavailable::StaleObservations,
                )) => 0,
                Some(CandidateObservationEvidence::Unavailable(
                    CandidateCostUnavailable::RepeatedFailures { .. }
                    | CandidateCostUnavailable::NoEstimate
                    | CandidateCostUnavailable::UnrepresentableEstimate,
                )) => return None,
            };
            least = Some(least.map_or(need, |prior: u64| prior.min(need)));
        }
    }
    least
}
