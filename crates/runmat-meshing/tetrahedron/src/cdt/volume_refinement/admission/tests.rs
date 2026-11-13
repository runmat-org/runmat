//! Independent complete evaluation is the oracle for field-aware local trials.
use super::*;
use crate::cdt::{
    insertion::{
        prepare_delaunay_volume_node_insertion_with_prepared_barriers_from_hint,
        topology_access::TetrahedralTopology,
    },
    test_fixtures::coupled_quality::{fixture_scaled, metric},
    volume_quality::working::PreparedLocalQuality,
    volume_refinement::*,
    volume_state::{SearchState, WorkingState},
};
use runmat_meshing_core::NeverCancelled;
use std::collections::BTreeSet;
mod atomic_batch;
mod source_metric_reference;
use source_metric_reference::legacy_source_metric_floor;

// witness: FEA-SHARED-RESOURCES-001/governed_path_equivalence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: An actual cavity trial rejected by the source-metric proxy improves complete
//   resolved-field quality and matches independent recomputation. This finite counterexample
//   establishes neither arbitrary geometry convergence nor complete study acceptance.
#[test]
fn complete_metric_trial_retains_independent_improvement_rejected_by_source_metric_proxy() {
    let mut witnessed = 0;
    for permuted in [false, true] {
        for scoped in [false, true] {
            for scale in [[1.0; 3], [1.7, 0.5, 1.0], [0.4, 3.0, 1.3], [1.0, 0.1, 4.0]] {
                witnessed += usize::from(check_case(permuted, scoped, scale));
            }
        }
    }
    assert!(
        witnessed > 0,
        "must exercise a real source-metric proxy mismatch and independently improving trial"
    );
}

fn check_case(permuted: bool, scoped: bool, scale: [f64; 3]) -> bool {
    let (topology, provenance) = fixture_scaled(permuted, 0, scale);
    let request = metric(&provenance, scoped);
    let options = DelaunayVolumeQualityOptions {
        minimum_metric_scaled_jacobian: 0.001,
        maximum_radius_edge_ratio: 10.0,
        ..DelaunayVolumeQualityOptions::default()
    };
    let original = crate::cdt::evaluate_delaunay_volume_quality(
        &topology,
        &request,
        &provenance,
        options,
        &NeverCancelled,
    )
    .unwrap();
    let prepared = PreparedLocalQuality::prepare(&request, &provenance, options).unwrap();
    let mut state = WorkingState::new(
        topology,
        original.clone(),
        options,
        &prepared,
        &[],
        &NeverCancelled,
    )
    .unwrap();
    let controls = DelaunayVolumeRefinementStepOptions::default();
    for source in &original.tetrahedra {
        let vertices = source
            .node_identities
            .map(|id| state.topology().node_index(id).unwrap());
        let index = state.topology().tetrahedron_index(vertices).unwrap();
        let mut excluded = BTreeSet::new();
        let mut work = CandidateWork::new(controls.candidate, &NeverCancelled);
        for _ in 0..16 {
            let Ok(candidate) = candidate::construct_candidate_on_topology(
                state.topology(),
                source,
                options,
                &[],
                &excluded,
                &mut work,
            ) else {
                break;
            };
            excluded.insert(candidate.node.identity);
            let Ok(insertion) =
                prepare_delaunay_volume_node_insertion_with_prepared_barriers_from_hint(
                    state.topology(),
                    candidate.node,
                    state.preparation().protected_facets(),
                    true,
                    Some(index),
                    controls.insertion,
                    &NeverCancelled,
                )
            else {
                continue;
            };
            if legacy_source_metric_floor(
                state.topology(),
                &candidate,
                &insertion,
                state.preparation().boundary_nodes(),
                options,
            ) {
                continue;
            }
            let parent = state.quality().capture_spectrum();
            let mut trial = state.trial();
            if trial
                .insert_prepared(insertion, controls, &NeverCancelled)
                .is_err()
                || RefinementProgressPolicy::PreserveShape
                    .rejection(trial.quality(), &parent)
                    .is_some()
            {
                continue;
            }
            let change = trial.commit_prepared(&NeverCancelled).unwrap();
            state.rebase_committed(change);
            let (topology, quality) = state.finish(controls, &NeverCancelled).unwrap();
            let independent = crate::cdt::evaluate_delaunay_volume_quality(
                &topology,
                &request,
                &provenance,
                options,
                &NeverCancelled,
            )
            .unwrap();
            assert_eq!(quality, independent);
            assert!(canonical_progress(&original, &independent, &NeverCancelled).unwrap());
            return true;
        }
    }
    false
}

#[test]
fn intermediate_single_split_preserves_local_ceiling_without_unrelated_masking() {
    // Policy-only ratios: these values do not constitute geometric admission.
    let (topology, provenance) = fixture_scaled(false, 0, [1.0; 3]);
    let request = metric(&provenance, false);
    let mut prior = crate::cdt::evaluate_delaunay_volume_quality(
        &topology,
        &request,
        &provenance,
        DelaunayVolumeQualityOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let make = |ids: [u8; 4], ratio| {
        let mut cell = prior.tetrahedra[0].clone();
        cell.node_identities = ids.map(|id| StableDigest::from_bytes([id; 32]));
        cell.refinement_violation_ratio = ratio;
        cell
    };
    // Canonical order uses sorted identities, independent of cell orientation.
    let source = make([1, 2, 5, 3], 2.0);
    let unrelated = make([20, 21, 22, 23], 100.0);
    let first = make([1, 2, 4, 3], 2.0);
    let second = make([1, 2, 5, 4], 2.0);
    prior.tetrahedra = vec![source, unrelated.clone()];
    let mut result = prior.clone();
    result.tetrahedra = vec![first, second, unrelated];
    assert!(canonical_single_step_nonregression(&prior, &result, &NeverCancelled).unwrap());
    assert_eq!(
        compare(
            &prior,
            &result,
            QualityObjective::Refinement,
            &NeverCancelled
        )
        .unwrap(),
        Ordering::Greater
    );
    result.tetrahedra[0].refinement_violation_ratio = 3.0;
    assert!(!canonical_single_step_nonregression(&prior, &result, &NeverCancelled).unwrap());
}
#[test]
fn single_step_comparison_polls_through_long_unchanged_prefixes() {
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    struct Stop(AtomicUsize);
    impl runmat_meshing_core::MeshingCancellationSignal for Stop {
        fn is_cancelled(&self) -> bool {
            self.0.fetch_add(1, AtomicOrdering::Relaxed) >= 1
        }
    }
    let (topology, provenance) = fixture_scaled(false, 0, [1.0; 3]);
    let mut prior = crate::cdt::evaluate_delaunay_volume_quality(
        &topology,
        &metric(&provenance, false),
        &provenance,
        DelaunayVolumeQualityOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let template = prior.tetrahedra[0].clone();
    prior.tetrahedra = (0..400u64)
        .map(|index| {
            let mut cell = template.clone();
            cell.node_identities = std::array::from_fn(|corner| {
                let mut bytes = [0; 32];
                bytes[..8].copy_from_slice(&(index * 4 + corner as u64).to_be_bytes());
                StableDigest::from_bytes(bytes)
            });
            cell
        })
        .collect();
    assert_eq!(
        canonical_single_step_nonregression(&prior, &prior, &Stop(AtomicUsize::new(0)))
            .unwrap_err()
            .kind,
        DelaunayVolumeRefinementStepErrorKind::Cancelled
    );
}
