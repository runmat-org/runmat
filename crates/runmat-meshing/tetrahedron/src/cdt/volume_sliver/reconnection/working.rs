//! Reconnection search on one working owner. Each candidate is a nested local
//! transaction; retained beam entries contain checked edits and ranking deltas.
use super::*;
use crate::cdt::volume_quality::working::SpectrumDelta;
use crate::cdt::volume_sliver::working::CaptureSeed;
use crate::cdt::volume_sliver::working::{SearchState, WorkingSeed, WorkingTrial};
mod deferred;
mod lookahead;

pub(in crate::cdt::volume_sliver) struct WorkingReconnectionSearch {
    pub(in crate::cdt::volume_sliver) candidate: Option<WorkingSeed>,
    pub(in crate::cdt::volume_sliver) rejection_counts: BTreeMap<&'static str, u64>,
    pending: Option<deferred::DeferredSearch>,
}

#[derive(Clone, Copy)]
enum Continuation {
    Relocation,
    ConnectivityOnly,
}

#[derive(Clone, Copy)]
struct SearchPolicy {
    sliver: SliverSearchPolicy,
    continuation: Continuation,
    deferred: bool,
}

pub(in crate::cdt::volume_sliver) fn connectivity_candidates<'ctx>(
    state: &mut impl SearchState<'ctx>,
    source: &DelaunayTetrahedronQuality,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut OptimizationWork<'_>,
    policy: SliverSearchPolicy,
    prefix: &[DelaunayVolumeSliverMutation],
) -> Result<WorkingReconnectionSearch, DelaunayVolumeSliverError> {
    search(
        state,
        source,
        cancellation,
        work,
        SearchPolicy {
            sliver: policy,
            continuation: Continuation::ConnectivityOnly,
            deferred: false,
        },
        ReconnectionMode {
            allow_lookahead: true,
            maximum_candidates: Some(policy.options.maximum_compound_reconnections_per_insertion),
        },
        prefix,
    )
}

pub(in crate::cdt::volume_sliver) fn candidates<'ctx>(
    state: &mut impl SearchState<'ctx>,
    source: &DelaunayTetrahedronQuality,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut OptimizationWork<'_>,
    policy: SliverSearchPolicy,
    mode: ReconnectionMode,
    prefix: &[DelaunayVolumeSliverMutation],
) -> Result<WorkingReconnectionSearch, DelaunayVolumeSliverError> {
    search(
        state,
        source,
        cancellation,
        work,
        SearchPolicy {
            sliver: policy,
            continuation: Continuation::Relocation,
            deferred: false,
        },
        mode,
        prefix,
    )
}

pub(in crate::cdt::volume_sliver) fn deferred_candidates<'ctx>(
    state: &mut impl SearchState<'ctx>,
    source: &DelaunayTetrahedronQuality,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut OptimizationWork<'_>,
    policy: SliverSearchPolicy,
    prefix: &[DelaunayVolumeSliverMutation],
) -> Result<WorkingReconnectionSearch, DelaunayVolumeSliverError> {
    search(
        state,
        source,
        cancellation,
        work,
        SearchPolicy {
            sliver: policy,
            continuation: Continuation::Relocation,
            deferred: true,
        },
        ReconnectionMode {
            allow_lookahead: true,
            maximum_candidates: None,
        },
        prefix,
    )
}

fn search<'ctx>(
    state: &mut impl SearchState<'ctx>,
    source: &DelaunayTetrahedronQuality,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut OptimizationWork<'_>,
    search_policy: SearchPolicy,
    mode: ReconnectionMode,
    prefix: &[DelaunayVolumeSliverMutation],
) -> Result<WorkingReconnectionSearch, DelaunayVolumeSliverError> {
    let SearchPolicy {
        sliver: policy,
        continuation: _,
        deferred: _,
    } = search_policy;
    let index = source_index(state.topology(), source)?;
    let choices = local_candidates(state.topology(), index as usize, source.resolved_metric)?;
    let refinement_baseline = state.quality().capture_spectrum();
    let mut rejection_counts = BTreeMap::new();
    let mut seeds = Vec::new();
    let mut best = None;
    for (position, candidate) in choices.into_iter().enumerate() {
        if mode
            .maximum_candidates
            .is_some_and(|limit| position as u64 >= limit)
        {
            break;
        }
        work.evaluate()?;
        let mutation = reconnection_mutation(state.topology(), &candidate, source.node_identities);
        let mut trial = state.trial();
        if !apply(&mut trial, &candidate, policy.options, cancellation)? {
            reject(
                &mut rejection_counts,
                "reconnection_geometry_not_admissible",
            );
            continue;
        }
        let mut mutations = prefix.to_vec();
        mutations.push(mutation);
        let admitted = crate::cdt::volume_sliver::admission::working_quality_is_admissible(
            trial.quality(),
            &refinement_baseline,
            policy.refinement,
            policy.objective,
            &mut rejection_counts,
        );
        if admitted {
            let candidate = trial.capture(mutations);
            if matches!(
                policy.refinement,
                super::super::admission::CandidateRefinementPolicy::Preserve
            ) {
                return Ok(WorkingReconnectionSearch {
                    candidate: Some(candidate),
                    rejection_counts,
                    pending: None,
                });
            }
            // Proposal order uses the source tensor. Changed cells can resolve
            // different scopes, so only actual trial quality can rank them.
            WorkingSeed::retain_better(&mut best, candidate, policy);
            continue;
        }
        if mode.allow_lookahead {
            seeds.push(trial.capture(mutations));
        }
    }
    if matches!(
        policy.refinement,
        super::super::admission::CandidateRefinementPolicy::AllowForCoupledConvergence
    ) {
        seeds.sort_by(|a, b| {
            a.compare_policy(b, policy)
                .then_with(|| a.compare_connectivity(b, state.quality()))
        });
    }
    if search_policy.deferred {
        return Ok(WorkingReconnectionSearch {
            candidate: best,
            rejection_counts,
            pending: Some(deferred::DeferredSearch {
                seeds,
                baseline: refinement_baseline,
                policy: search_policy,
            }),
        });
    }
    if best.is_some() {
        return Ok(WorkingReconnectionSearch {
            candidate: best,
            rejection_counts,
            pending: None,
        });
    }
    let candidate = if mode.allow_lookahead {
        lookahead::search(
            state,
            seeds,
            &refinement_baseline,
            cancellation,
            work,
            search_policy,
            &mut rejection_counts,
        )?
    } else {
        None
    };
    Ok(WorkingReconnectionSearch {
        candidate,
        rejection_counts,
        pending: None,
    })
}

fn source_index(
    topology: &(impl VolumeTopologyRead + ?Sized),
    source: &DelaunayTetrahedronQuality,
) -> Result<u32, DelaunayVolumeSliverError> {
    let nodes = source
        .node_identities
        .into_iter()
        .map(|identity| {
            topology.node_index(identity).ok_or_else(|| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    "working quality corner is absent",
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    topology
        .tetrahedron_index(nodes.try_into().expect("four corners"))
        .ok_or_else(|| {
            error(
                DelaunayVolumeSliverErrorKind::InvalidTopology,
                "working quality source is absent",
            )
        })
}

fn apply(
    state: &mut WorkingTrial<'_, '_>,
    candidate: &LocalTetrahedronFlipCandidate,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<bool, DelaunayVolumeSliverError> {
    let region = state
        .topology()
        .cell(candidate.removed_tetrahedron_ids[0])
        .region_id
        .clone();
    let removed = candidate.removed_tetrahedron_ids.iter().copied().collect();
    let created = candidate
        .created_tetrahedra
        .iter()
        .copied()
        .map(|vertices| (vertices, region.clone()))
        .collect();
    match state
        .replace(&removed, created, options, cancellation)
        .map_err(DelaunayVolumeSliverError::from)
    {
        Ok(()) => Ok(true),
        Err(failure) if failure.kind == DelaunayVolumeSliverErrorKind::InvalidTopology => Ok(false),
        Err(failure) => Err(failure),
    }
}

#[cfg(test)]
mod tests;
