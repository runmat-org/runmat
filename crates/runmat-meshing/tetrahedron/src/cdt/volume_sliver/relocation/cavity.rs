//! Generate and rank checked cavity geometry once per source. Only distinct
//! retained connectivity branches receive graph search and joint relaxation.
use super::super::working::WorkingSeed;
use super::*;
use crate::cdt::insertion::topology_access::TetrahedralTopology;
use crate::cdt::volume_state::SearchState;
mod attempt;
use crate::cdt::volume_sliver::frontier::RankedFrontier;
use attempt::{attempt, CavityAttempt, CavityGrowth};

pub(super) fn candidate<'ctx>(
    state: &mut impl SearchState<'ctx>,
    source: &DelaunayTetrahedronQuality,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut OptimizationWork<'_>,
    policy: SliverSearchPolicy,
    prefix: &[DelaunayVolumeSliverMutation],
) -> Result<Option<WorkingSeed>, DelaunayVolumeSliverError> {
    let vertices = super::neighborhood::vertices(state, source);
    if vertices.is_empty() {
        return Ok(None);
    }
    let baseline = state.quality().capture_spectrum();
    let mut best = None;
    let mut frontier = RankedFrontier::new(2);
    let mut identities = source.node_identities;
    identities.sort_unstable();
    for identity in identities {
        let primary = state
            .topology()
            .node_index(identity)
            .expect("source corner");
        if !vertices.contains(&primary) {
            continue;
        }
        let start = state
            .topology()
            .node(primary)
            .coordinates_m
            .canonical_value();
        let mut points = Vec::new();
        let mut face = source
            .node_identities
            .into_iter()
            .filter(|id| *id != identity)
            .collect::<Vec<_>>();
        face.sort_unstable();
        let face: [StableDigest; 3] = face.try_into().expect("opposite face");
        if state.preparation().protected_facets().contains(&face) {
            let positions = face.map(|id| {
                state
                    .topology()
                    .node(state.topology().node_index(id).expect("face point"))
                    .coordinates_m
                    .canonical_value()
            });
            points.extend(
                super::super::offcenter::points_for_face(
                    positions,
                    start,
                    source.resolved_metric,
                    state.quality_options().minimum_metric_scaled_jacobian,
                    cancellation,
                )?
                .into_iter()
                .map(|(point, _)| point),
            );
        }
        if let Some(target) =
            geometry::apex::target(state.topology(), source, identity, cancellation)?
        {
            points.extend((0..24).map(|backtrack| {
                let fraction = 2_f64.powi(-backtrack);
                std::array::from_fn(|axis| start[axis] + fraction * (target[axis] - start[axis]))
            }));
        }
        for point in points {
            for growth in [CavityGrowth::Visibility, CavityGrowth::Quality] {
                work.evaluate()?;
                let request = CavityAttempt {
                    growth,
                    source,
                    primary,
                    point,
                    baseline: &baseline,
                    policy,
                    prefix,
                };
                let Some(candidate) = attempt(state, request, work, cancellation)? else {
                    continue;
                };
                if candidate.admitted {
                    WorkingSeed::retain_better(&mut best, candidate.seed, policy);
                } else {
                    frontier.retain_distinct_by(
                        candidate.seed,
                        |candidate, current| candidate.compare_policy(current, policy),
                        |candidate, current| {
                            candidate.connectivity_key() == current.connectivity_key()
                        },
                    );
                }
            }
        }
    }
    if let Some(candidate) =
        attempt::continuation::search(state, frontier.into_entries(), policy, work, cancellation)?
    {
        WorkingSeed::retain_better(&mut best, candidate, policy);
    }
    Ok(best)
}
