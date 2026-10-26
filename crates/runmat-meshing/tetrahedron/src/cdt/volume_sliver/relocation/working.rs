//! Local geometry and cavity relocation share direct and bounded compound searches. Proposals
//! and numerical scoring are shared with the original independently checked path.
use super::*;
use crate::cdt::insertion::topology_access::TetrahedralTopology;
use crate::cdt::volume_sliver::working::CaptureSeed;
use crate::cdt::volume_sliver::working::{SearchState, WorkingSeed};

pub(in crate::cdt::volume_sliver) struct WorkingRelocationSearch {
    pub(in crate::cdt::volume_sliver) candidate: Option<WorkingSeed>,
    pub(in crate::cdt::volume_sliver) rejection_counts: BTreeMap<&'static str, u64>,
}

pub(in crate::cdt::volume_sliver) fn candidates<'ctx>(
    state: &mut impl SearchState<'ctx>,
    source: &DelaunayTetrahedronQuality,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut OptimizationWork<'_>,
    policy: SliverSearchPolicy,
    prefix: &[DelaunayVolumeSliverMutation],
) -> Result<WorkingRelocationSearch, DelaunayVolumeSliverError> {
    let mut rejection_counts = BTreeMap::new();
    let mut best = None;
    let refinement_baseline = state.quality().capture_spectrum();
    let mut source_nodes = source.node_identities;
    source_nodes.sort_unstable();
    for identity in source_nodes {
        if state.protected_node(identity) {
            continue;
        }
        let index = state.topology().node_index(identity).ok_or_else(|| {
            error(
                DelaunayVolumeSliverErrorKind::InvalidTopology,
                "working relocation source is absent",
            )
        })?;
        if state.topology().vertex_is_boundary(index) {
            continue;
        }
        let start = state.topology().node(index).coordinates_m.canonical_value();
        let mut coordinates = smooth_vertex_star_points_with_metrics(
            state.topology(),
            index as usize,
            &|cell| {
                let identities = state
                    .topology()
                    .cell(cell)
                    .vertex_indices
                    .map(|vertex| state.topology().node(vertex).identity);
                state
                    .quality()
                    .quality_for_cell(identities)
                    .expect("current star quality")
                    .resolved_metric
            },
            geometry::SmoothingObjective::for_policy(policy, state.quality_options()),
            cancellation,
        )?;
        let targets = [
            sliver_height_target(
                state.topology(),
                source,
                identity,
                state.quality_options().minimum_metric_scaled_jacobian,
            )?,
            Some(volume_weighted_star_centroid(
                state.topology(),
                index as usize,
            )?),
            Some(neighbor_centroid(state.topology(), index as usize)?),
        ];
        for target in targets.into_iter().flatten() {
            for fraction in [1.0, 0.5, 0.25, 0.125] {
                let point = std::array::from_fn(|axis| {
                    start[axis] + fraction * (target[axis] - start[axis])
                });
                if !coordinates.contains(&point) {
                    coordinates.push(point);
                }
            }
        }
        for point in coordinates {
            work.evaluate()?;
            let replacement = DelaunayVolumeNode::explicit(replacement_identity(identity), point);
            let mut trial = state.trial();
            if let Err(failure) = trial
                .replace_node(index, replacement, policy.options, cancellation)
                .map_err(DelaunayVolumeSliverError::from)
            {
                if failure.kind == DelaunayVolumeSliverErrorKind::InvalidTopology {
                    reject(
                        &mut rejection_counts,
                        "fixed_star_relocation_not_admissible",
                    );
                    continue;
                }
                return Err(failure);
            }
            if !crate::cdt::volume_sliver::admission::working_quality_is_admissible(
                trial.quality(),
                &refinement_baseline,
                policy.refinement,
                policy.objective,
                &mut rejection_counts,
            ) {
                continue;
            }
            let mut mutations = prefix.to_vec();
            mutations.push(relocation_mutation(
                identity,
                replacement,
                source.node_identities,
            ));
            let candidate = trial.capture(mutations);
            if matches!(
                policy.refinement,
                super::super::admission::CandidateRefinementPolicy::Preserve
            ) {
                return Ok(WorkingRelocationSearch {
                    candidate: Some(candidate),
                    rejection_counts,
                });
            }
            WorkingSeed::retain_better(&mut best, candidate, policy);
        }
    }
    if matches!(
        policy.refinement,
        super::super::admission::CandidateRefinementPolicy::AllowForCoupledConvergence
    ) {
        // A tentative reconnection can join previously isolated free-point
        // stars. The same bounded cluster operator must be available there;
        // accepting only independent point moves strands that compound branch.
        if let Some(candidate) =
            super::block::candidate(state, source, cancellation, work, policy, prefix)?
        {
            WorkingSeed::retain_better(&mut best, candidate, policy);
        }
        if let Some(candidate) =
            super::cavity::candidate(state, source, cancellation, work, policy, prefix)?
        {
            WorkingSeed::retain_better(&mut best, candidate, policy);
        }
    }
    Ok(WorkingRelocationSearch {
        candidate: best,
        rejection_counts,
    })
}
