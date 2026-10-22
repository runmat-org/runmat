//! Exhaustive complete-topology arithmetic checks the local candidate ordering.
use super::*;
use crate::cdt::volume_sliver::{admission, working::WorkingState};
use crate::cdt::{test_fixtures::coupled_quality, volume_quality::working::PreparedLocalQuality};
use runmat_meshing_core::NeverCancelled;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Coupled direct reconnection is compared with complete-topology quality arithmetic
//   across scoped-metric candidate cells and identity permutations. Actual changed-cell spectra
//   decide selection; source-tensor proposal order supplies no final ranking authority. Standalone
//   first-admissible behavior remains a separate contract.
#[test]
fn coupled_direct_reconnection_selects_the_best_actual_cell_spectrum() {
    let mut checked = 0;
    for permuted in [false, true] {
        let (topology, provenance) = fixture(permuted);
        let request = coupled_quality::metric(&provenance, true);
        let quality_options = crate::cdt::DelaunayVolumeQualityOptions {
            minimum_metric_scaled_jacobian: 0.5,
            ..crate::cdt::volume_sliver::tests::quality_options()
        };
        let quality = crate::cdt::evaluate_delaunay_volume_quality(
            &topology,
            &request,
            &provenance,
            quality_options,
            &NeverCancelled,
        )
        .unwrap();
        let prepared =
            PreparedLocalQuality::prepare(&request, &provenance, quality_options).unwrap();
        let options = DelaunayVolumeSliverOptions::default();
        let policy = SliverSearchPolicy {
            options,
            refinement: admission::CandidateRefinementPolicy::AllowForCoupledConvergence,
            objective: admission::QualityObjective::Shape,
        };
        let baseline = admission::candidate_spectrum(&quality, policy.refinement, policy.objective);
        for (source_index, source) in quality.tetrahedra.iter().enumerate() {
            let mut expected = None;
            for candidate in
                local_candidates(&topology, source_index, source.resolved_metric).unwrap()
            {
                let Some(candidate) =
                    super::super::apply_candidate(&topology, &candidate, options, &NeverCancelled)
                        .unwrap()
                else {
                    continue;
                };
                let candidate_quality = crate::cdt::evaluate_delaunay_volume_quality(
                    &candidate,
                    &request,
                    &provenance,
                    quality_options,
                    &NeverCancelled,
                )
                .unwrap();
                let spectrum = admission::candidate_spectrum(
                    &candidate_quality,
                    policy.refinement,
                    policy.objective,
                );
                if admission::compare_spectra(&spectrum, &baseline).is_lt()
                    && expected.as_ref().is_none_or(|previous: &Vec<f64>| {
                        admission::compare_spectra(&spectrum, previous).is_lt()
                    })
                {
                    expected = Some(spectrum);
                }
            }
            let mut state = WorkingState::new(
                topology.clone(),
                quality.clone(),
                quality_options,
                &prepared,
                &[],
                &NeverCancelled,
            )
            .unwrap();
            let mut work = OptimizationWork::new(
                options.maximum_candidate_evaluations_per_pass,
                options.cancellation_check_interval,
                &NeverCancelled,
            );
            let actual = candidates(
                &mut state,
                source,
                &NeverCancelled,
                &mut work,
                policy,
                ReconnectionMode {
                    allow_lookahead: false,
                    maximum_candidates: None,
                },
                &[],
            )
            .unwrap()
            .candidate;
            assert_eq!(actual.is_some(), expected.is_some());
            if let Some(actual) = actual {
                let mut trial = state.trial();
                actual.replay(&mut trial, options, &NeverCancelled).unwrap();
                trial.commit();
                let (_, actual) = state.finish(options, &NeverCancelled).unwrap();
                assert_eq!(
                    admission::compare_spectra(
                        &admission::candidate_spectrum(
                            &actual,
                            policy.refinement,
                            policy.objective
                        ),
                        &expected.unwrap()
                    ),
                    std::cmp::Ordering::Equal
                );
                checked += 1;
            }
        }
    }
    assert!(
        checked > 0,
        "the oracle must exercise admitted reconnections"
    );
}

fn fixture(
    permuted: bool,
) -> (
    crate::cdt::DelaunayVolumeTopology,
    crate::cdt::DelaunayVolumeProvenance,
) {
    let (_, template) = coupled_quality::fixture(false, 0);
    let region = template.facets[0].region_ids[0].clone();
    let mut nodes = [
        [0., 0., 0.],
        [1., 0., 0.],
        [0.5, 0.866, 0.],
        [0.5, 0.3, 0.1],
        [0.5, 0.3, -0.1],
    ]
    .into_iter()
    .enumerate()
    .map(|(i, p)| {
        crate::cdt::DelaunayVolumeNode::explicit(
            StableDigest::from_bytes([if permuted { 100 - i as u8 } else { i as u8 + 1 }; 32]),
            p,
        )
    })
    .collect::<Vec<_>>();
    if permuted {
        nodes.reverse();
    }
    let cells = [[0, 1, 2, 3], [0, 1, 2, 4]]
        .map(|cell| cell.map(|vertex| if permuted { 4 - vertex } else { vertex }))
        .to_vec();
    // Input IDs are intentionally permuted independently of geometry.
    let topology = crate::cdt::build_delaunay_volume_topology(
        nodes,
        cells,
        Default::default(),
        &NeverCancelled,
    )
    .unwrap();
    let count = topology.tetrahedra.len();
    let topology = crate::cdt::assign_delaunay_volume_regions(
        topology,
        vec![region.clone(); count],
        Default::default(),
        &NeverCancelled,
    )
    .unwrap();
    let mut facets = topology
        .incidence
        .boundary_facets
        .iter()
        .enumerate()
        .map(|(i, face)| {
            let mut facet = template.facets[0].clone();
            facet.node_identities = face
                .vertex_indices
                .map(|vertex| topology.nodes[vertex as usize].identity);
            facet.node_identities.sort_unstable();
            facet.entity_ids[0].source_topology_id = format!("bipyramid-face-{i}");
            facet
        })
        .collect::<Vec<_>>();
    facets.sort_by_key(|face| face.node_identities);
    (
        topology,
        crate::cdt::DelaunayVolumeProvenance {
            nodes: vec![],
            segments: vec![],
            facets,
        },
    )
}
