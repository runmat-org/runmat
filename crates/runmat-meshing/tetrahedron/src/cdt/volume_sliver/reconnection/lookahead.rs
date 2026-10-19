//! Compact-seed beam search; the active topology is materialized one state at a time.
use super::*;
use crate::cdt::volume_sliver::admission::CandidateRefinementPolicy;

pub(super) fn bounded_reconnection_lookahead(
    input: DelaunayVolumeRefinementInput<'_>,
    current_spectrum: &[f64],
    cancellation: &dyn MeshingCancellationSignal,
    frontier: LookaheadFrontier<'_>,
    rejection_counts: &mut BTreeMap<&'static str, u64>,
    work: &mut OptimizationWork<'_>,
    policy: SliverSearchPolicy,
) -> Result<Option<ReconnectionCandidate>, DelaunayVolumeSliverError> {
    let LookaheadFrontier {
        protected_faces,
        seeds: lookahead_seeds,
    } = frontier;
    let SliverSearchPolicy {
        options,
        refinement: refinement_policy,
        objective,
    } = policy;
    const MAXIMUM_LOOKAHEAD_DEPTH: usize = 4;
    const MAXIMUM_LOOKAHEAD_WIDTH: usize = 8;
    let mut frontier = lookahead_seeds;
    if matches!(
        refinement_policy,
        CandidateRefinementPolicy::AllowForCoupledConvergence
    ) {
        frontier.sort_by(|a, b| {
            a.quality
                .compare_policy(&b.quality, input.quality, policy)
                .then_with(|| a.patch.compare(&b.patch, input.topology))
        });
    }
    for _depth in 2..=MAXIMUM_LOOKAHEAD_DEPTH {
        let mut next_frontier =
            crate::cdt::volume_sliver::frontier::RankedFrontier::new(MAXIMUM_LOOKAHEAD_WIDTH);
        let mut seen = BTreeSet::new();
        for state in frontier {
            let topology = state
                .patch
                .materialize(input.topology, options, cancellation)?;
            let quality = state.quality.materialize(input.quality);
            let Some(next_source) = crate::cdt::volume_sliver::admission::source_for_policy(
                &quality, &topology, policy,
            ) else {
                continue;
            };
            let Some(next_source_index) = topology.tetrahedra.iter().position(|tetrahedron| {
                identities(&topology, tetrahedron.vertex_indices) == next_source.node_identities
            }) else {
                return Err(error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    "lookahead quality source is absent from its candidate topology",
                ));
            };
            for next in local_candidates(&topology, next_source_index, next_source.resolved_metric)?
            {
                work.evaluate()?;
                let Some(next_topology) = apply_candidate(&topology, &next, options, cancellation)?
                else {
                    continue;
                };
                let Some(next_quality) = evaluate_canonical_candidate_unranked(
                    &next_topology,
                    input,
                    protected_faces,
                    options,
                    cancellation,
                    rejection_counts,
                )?
                else {
                    continue;
                };
                let mut mutations = state.mutations.clone();
                mutations.push(reconnection_mutation(
                    &topology,
                    &next,
                    next_source.node_identities,
                ));
                if candidate_quality_is_admissible(
                    &next_quality,
                    input,
                    current_spectrum,
                    refinement_policy,
                    objective,
                    rejection_counts,
                ) {
                    return Ok(Some(ReconnectionCandidate {
                        topology: next_topology.into_inner(),
                        quality: next_quality,
                        mutations,
                    }));
                }
                let key = connectivity_delta(input.topology, &next_topology);
                if seen.insert(key) {
                    let patch = TopologyPatch::between(input.topology, &next_topology)?;
                    next_frontier.retain(
                        LookaheadSeed {
                            patch,
                            quality: QualityPatch::between(input.quality, &next_quality)?,
                            mutations,
                        },
                        |candidate: &LookaheadSeed, entry: &LookaheadSeed| {
                            candidate
                                .quality
                                .compare_policy(&entry.quality, input.quality, policy)
                                .then_with(|| candidate.patch.compare(&entry.patch, input.topology))
                        },
                    );
                }
            }
            let state_input = DelaunayVolumeRefinementInput {
                topology: &topology,
                metric_request: input.metric_request,
                provenance: input.provenance,
                quality: &quality,
                quality_options: input.quality_options,
                fixed_node_identities: input.fixed_node_identities,
            };
            let relocation = relocation_candidates(
                state_input,
                next_source,
                current_spectrum,
                cancellation,
                work,
                SliverSearchPolicy {
                    options,
                    refinement: refinement_policy,
                    objective,
                },
                crate::cdt::volume_sliver::relocation::RelocationSearchScope::FixedStarOnly,
            )?;
            if let Some(relocation) = relocation.candidate {
                let mut mutations = state.mutations;
                mutations.push(relocation.mutation);
                return Ok(Some(ReconnectionCandidate {
                    topology: relocation.topology,
                    quality: relocation.quality,
                    mutations,
                }));
            }
        }
        let next_frontier = next_frontier.into_entries();
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }
    Ok(None)
}
