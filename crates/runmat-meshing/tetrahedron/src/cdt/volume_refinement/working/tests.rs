//! The production refinement step must not evaluate untouched background cells.
use super::*;
use crate::cdt::test_fixtures::coupled_quality::{fixture, metric};
use crate::cdt::volume_quality::working::PreparedLocalQuality;
use runmat_meshing_core::NeverCancelled;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: The actual production refinement step admits identical coordinates and evaluates
//   exactly eight cells with 0/64/4096 unrelated cells under uniform/scoped metrics. Its last
//   scoped grading update remains 16 invalidated cells and 152 edge reads. Complete preparation and
//   terminal independent admission are outside the local measurement; this does not bound all
//   search operations, total memory, complete-relocation work or original-model runtime.
#[test]
fn production_local_refinement_work_is_independent_of_unrelated_mesh_inventory() {
    for scoped in [false, true] {
        let mut baseline = None;
        for background in [0, 64, 4096] {
            let (topology, provenance) = fixture(false, background);
            let request = metric(&provenance, scoped);
            let options = crate::cdt::DelaunayVolumeQualityOptions::default();
            let quality = crate::cdt::evaluate_delaunay_volume_quality(
                &topology,
                &request,
                &provenance,
                options,
                &NeverCancelled,
            )
            .unwrap();
            let prepared = PreparedLocalQuality::prepare(&request, &provenance, options).unwrap();
            let mut state =
                WorkingState::new(topology, quality, options, &prepared, &[], &NeverCancelled)
                    .unwrap();
            let before = prepared.cell_evaluations();
            let progress = WorkingRefinement::default()
                .step(
                    &mut state,
                    DelaunayVolumeRefinementStepOptions::default(),
                    64,
                    &NeverCancelled,
                )
                .unwrap();
            let WorkingRefinementProgress::Accepted(inserted) = progress else {
                panic!("genuine cavity was not admitted");
            };
            assert!(!inserted.is_empty());
            let evaluations = prepared.cell_evaluations() - before;
            assert!(
                evaluations > 0 && evaluations < 128,
                "evaluated {evaluations} cells with {background} unrelated cells"
            );
            let scoped_work = scoped.then(|| state.quality().last_scoped_update_work().unwrap());
            if let Some((invalidated, reads)) = scoped_work {
                assert!(
                    invalidated > 0 && invalidated < 128 && reads > 0,
                    "unexpected grading support: {invalidated} cells, {reads} edge reads"
                );
            }
            // Grading reads edges in multiple propagation phases. Compare its
            // actual counts across inventory sizes rather than treating an edge
            // visit as a distinct invalidated cell.
            eprintln!(
                "local refinement: scoped={scoped}, background={background}, evaluated_cells={evaluations}, grading_work={scoped_work:?}"
            );
            let observed = (
                evaluations,
                scoped_work,
                inserted
                    .iter()
                    .map(|node| node.coordinates_m.canonical_value())
                    .collect::<Vec<_>>(),
            );
            if let Some(expected) = &baseline {
                assert_eq!(&observed, expected, "work grew with unrelated inventory");
            } else {
                baseline = Some(observed);
            }
            // This terminal check is deliberately complete and outside the local
            // work measurement. Quality may still require a later shape step.
            let _validated = state
                .finish(
                    DelaunayVolumeRefinementStepOptions::default(),
                    &NeverCancelled,
                )
                .unwrap();
        }
    }
}
