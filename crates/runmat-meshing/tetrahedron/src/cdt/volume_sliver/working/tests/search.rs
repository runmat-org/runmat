//! Differential search evidence against complete canonical candidate admission.
use super::*;
use crate::cdt::volume_sliver::working::CaptureSeed;
use crate::cdt::volume_sliver::{
    admission, insertion, reconnection, relocation, work::OptimizationWork,
};

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: One interior-sliver fixture compares reconnection, fixed-star relocation and
//   insertion selection against complete canonical search, including mutation sequence, topology
//   and every quality record. Uniform and scoped face sizing are both exercised. This does not
//   establish arbitrary mesh equivalence, deletion/retriangulation locality, measured native demand
//   or accepted study results.
#[test]
fn working_searches_match_canonical_candidate_selection_and_full_quality() {
    check_searches(false);
    check_searches(true);
}
fn check_searches(scoped: bool) {
    let (base, provenance) = interior_sliver_fixture();
    let mut request = metric();
    if scoped {
        request
            .contributions
            .push(runmat_meshing_size::metric::MetricContribution {
                source: runmat_meshing_size::metric::MetricSourceKind::Face,
                scope: runmat_meshing_size::metric::MetricContributionScope::Entity {
                    entity_id: provenance.facets[0].entity_ids[0].clone(),
                },
                metric: runmat_meshing_size::metric::MetricTensor3::isotropic_length_m(0.2)
                    .unwrap(),
            });
    }
    let reference = crate::cdt::evaluate_delaunay_volume_quality(
        &base,
        &request,
        &provenance,
        quality_options(),
        &NeverCancelled,
    )
    .unwrap();
    let options = DelaunayVolumeSliverOptions::default();
    let prepared = PreparedLocalQuality::prepare(&request, &provenance, quality_options()).unwrap();
    let input = DelaunayVolumeRefinementInput {
        topology: &base,
        quality: &reference,
        metric_request: &request,
        provenance: &provenance,
        quality_options: quality_options(),
        fixed_node_identities: &[],
    };
    let source =
        admission::worst_quality_source(&reference, &base, QualityObjective::Shape).unwrap();
    let spectrum = admission::quality_spectrum(&reference, QualityObjective::Shape);
    let policy = SliverSearchPolicy {
        options,
        refinement: admission::CandidateRefinementPolicy::Preserve,
        objective: QualityObjective::Shape,
    };
    let work = || {
        OptimizationWork::new(
            options.maximum_candidate_evaluations_per_pass,
            options.cancellation_check_interval,
            &NeverCancelled,
        )
    };
    for strategy in 0..3 {
        let mut state = WorkingState::new(
            base.clone(),
            reference.clone(),
            quality_options(),
            &prepared,
            &[],
            &NeverCancelled,
        )
        .unwrap();
        let (expected, actual) = match strategy {
            0 => {
                let mode = reconnection::ReconnectionMode {
                    allow_lookahead: true,
                    maximum_candidates: None,
                };
                let legacy = reconnection::reconnection_candidates(
                    input,
                    source,
                    &spectrum,
                    &NeverCancelled,
                    &mut work(),
                    policy,
                    mode,
                )
                .unwrap();
                let local = reconnection::working::candidates(
                    &mut state,
                    source,
                    &NeverCancelled,
                    &mut work(),
                    policy,
                    mode,
                    &[],
                )
                .unwrap();
                (
                    legacy
                        .candidate
                        .map(|c| (c.topology, c.quality, c.mutations)),
                    local.candidate,
                )
            }
            1 => {
                let legacy = relocation::relocation_candidates(
                    input,
                    source,
                    &spectrum,
                    &NeverCancelled,
                    &mut work(),
                    policy,
                    relocation::RelocationSearchScope::FixedStarOnly,
                )
                .unwrap();
                let local = relocation::working::candidates(
                    &mut state,
                    source,
                    &NeverCancelled,
                    &mut work(),
                    policy,
                    &[],
                )
                .unwrap();
                (
                    legacy
                        .candidate
                        .map(|c| (c.topology, c.quality, vec![c.mutation])),
                    local.candidate,
                )
            }
            _ => {
                let legacy = insertion::insertion_candidates(
                    input,
                    source,
                    &spectrum,
                    &NeverCancelled,
                    &mut work(),
                    policy,
                    1,
                )
                .unwrap();
                let local = insertion::working::candidates(
                    &mut state,
                    source,
                    &NeverCancelled,
                    &mut work(),
                    policy,
                    1,
                    &[],
                )
                .unwrap();
                (
                    legacy
                        .candidate
                        .map(|c| (c.topology, c.quality, c.mutations)),
                    local.candidate,
                )
            }
        };
        assert_eq!(actual.is_some(), expected.is_some(), "strategy {strategy}");
        if let (Some(candidate), Some((expected_topology, expected_quality, mutations))) =
            (actual, expected)
        {
            {
                let mut trial = state.trial();
                candidate
                    .replay(&mut trial, options, &NeverCancelled)
                    .unwrap();
                trial.commit();
            }
            let (actual_topology, actual_quality) = state.finish(options, &NeverCancelled).unwrap();
            assert_eq!(
                candidate.mutations, mutations,
                "strategy {strategy} mutations"
            );
            assert_eq!(
                actual_topology, expected_topology,
                "strategy {strategy} topology"
            );
            assert_eq!(
                actual_quality, expected_quality,
                "strategy {strategy} quality"
            );
            assert_eq!(
                actual_quality,
                crate::cdt::evaluate_delaunay_volume_quality(
                    &actual_topology,
                    &request,
                    &provenance,
                    quality_options(),
                    &NeverCancelled
                )
                .unwrap(),
                "strategy {strategy} independent full evaluation"
            );
        }
    }
}

#[test]
fn constrained_working_insertion_uses_exact_changed_quality_and_rolls_back() {
    let (base, provenance) = interior_sliver_fixture();
    let reference = quality(&base, &provenance);
    let request = metric();
    let options = DelaunayVolumeSliverOptions::default();
    let prepared = PreparedLocalQuality::prepare(&request, &provenance, quality_options()).unwrap();
    let mut state = WorkingState::new(
        base.clone(),
        reference.clone(),
        quality_options(),
        &prepared,
        &[],
        &NeverCancelled,
    )
    .unwrap();
    let node = DelaunayVolumeNode::explicit(StableDigest::from_bytes([6; 32]), [0.25, 0.25, 0.25]);
    let seed = {
        let mut trial = state.trial();
        trial
            .insert_constrained(node, options, &NeverCancelled)
            .unwrap();
        trial.capture(Vec::new())
    };
    assert_eq!(state.quality().cell_count(), reference.tetrahedra.len());
    {
        let mut trial = state.trial();
        seed.replay(&mut trial, options, &NeverCancelled).unwrap();
        trial.commit();
    }
    let (actual, actual_quality) = state.finish(options, &NeverCancelled).unwrap();
    assert_eq!(actual_quality, quality(&actual, &provenance));
    let expected = crate::cdt::insertion::insert_delaunay_volume_node_locally_with_barriers(
        base,
        node,
        &provenance
            .facets
            .iter()
            .map(|facet| facet.node_identities)
            .collect::<Vec<_>>(),
        options.insertion,
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(actual, expected);
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: One scoped interior-sliver treatment compares final topology, all quality records and
//   the mutation sequence with the complete canonical orchestration oracle. Complete
//   deletion/retriangulation locality, native resource peaks and guide-study qualification require
//   separate evidence.
#[test]
fn scoped_treatment_matches_complete_canonical_oracle() {
    let (topology, provenance) = interior_sliver_fixture();
    let mut request = metric();
    request
        .contributions
        .push(runmat_meshing_size::metric::MetricContribution {
            source: runmat_meshing_size::metric::MetricSourceKind::Region,
            scope: runmat_meshing_size::metric::MetricContributionScope::Region {
                region_id: topology.tetrahedra[0].region_id.clone().unwrap(),
            },
            metric: runmat_meshing_size::metric::MetricTensor3::isotropic_length_m(0.5).unwrap(),
        });
    let quality = crate::cdt::evaluate_delaunay_volume_quality(
        &topology,
        &request,
        &provenance,
        quality_options(),
        &NeverCancelled,
    )
    .unwrap();
    let context = crate::cdt::volume_refinement::context::RefinementContext {
        metric_request: &request,
        provenance: &provenance,
        quality_options: quality_options(),
        fixed_node_identities: &[],
    };
    let expected = crate::cdt::volume_sliver::reference::run(
        topology.clone(),
        quality.clone(),
        context,
        DelaunayVolumeSliverOptions::default(),
        &NeverCancelled,
        QualityConvergenceMode::ShapeOnly,
    )
    .unwrap();
    let actual = crate::cdt::volume_sliver::run_treatment(
        topology,
        quality,
        context,
        DelaunayVolumeSliverOptions::default(),
        &NeverCancelled,
        QualityConvergenceMode::ShapeOnly,
    )
    .unwrap();
    assert_eq!(actual, expected);
}
