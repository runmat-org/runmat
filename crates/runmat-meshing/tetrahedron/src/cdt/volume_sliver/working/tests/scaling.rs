//! A fixed local edit must not retain the disconnected background inventory.
//! Full independent handoff checks remain proportional to the complete mesh.
use super::*;
use crate::cdt::topology::build_delaunay_volume_topology_with_regions;
use crate::cdt::volume_sliver::working::CaptureSeed;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: One centroid split retains two checked edits and bounded quality undo entries with
//   zero, 32 and256 disconnected background cells under uniform and scoped face sizing; rollback
//   and independent complete handoff match the original state. The counter bounds retained local
//   records, not total preparation work, allocator bytes, connected dependency propagation or
//   native peak memory.
#[test]
fn retained_local_edits_and_quality_undo_are_independent_of_background_mesh_size() {
    check_scaling(false);
    check_scaling(true);
}
fn check_scaling(scoped: bool) {
    for background_count in [0, 32, 256] {
        let (base, provenance) = interior_sliver_fixture();
        let mut nodes = base.nodes;
        let mut cells = base
            .tetrahedra
            .into_iter()
            .map(|cell| (cell.vertex_indices, cell.region_id))
            .collect::<Vec<_>>();
        let region = cells[0].1.clone();
        for background in 0..background_count {
            let first = nodes.len() as u32;
            let x = 10.0 + background as f64 * 3.0;
            for (corner, point) in [
                [x, 0.0, 0.0],
                [x + 1.0, 0.0, 0.0],
                [x, 1.0, 0.0],
                [x, 0.0, 1.0],
            ]
            .into_iter()
            .enumerate()
            {
                let mut identity = [128; 32];
                identity[28..].copy_from_slice(&((background * 4 + corner) as u32).to_be_bytes());
                nodes.push(DelaunayVolumeNode::explicit(
                    StableDigest::from_bytes(identity),
                    point,
                ));
            }
            cells.push(([first, first + 1, first + 2, first + 3], region.clone()));
        }
        let base = build_delaunay_volume_topology_with_regions(
            nodes,
            cells,
            DelaunayVolumeSliverOptions::default().insertion.topology,
            &NeverCancelled,
        )
        .unwrap();
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
        let prepared =
            PreparedLocalQuality::prepare(&request, &provenance, quality_options()).unwrap();
        let mut state = WorkingState::new(
            base.clone(),
            reference.clone(),
            quality_options(),
            &prepared,
            &[],
            &NeverCancelled,
        )
        .unwrap();
        {
            let mut trial = state.trial();
            split(&mut trial, StableDigest::from_bytes([6; 32]));
            let seed = trial.capture(Vec::new());
            assert_eq!(seed.edits.len(), 2, "background cells {background_count}");
            let retained = trial.quality().retained_trial_entry_count();
            if scoped {
                let (invalidated, reads) = trial.quality().last_scoped_update_work().unwrap();
                assert!(
                    invalidated <= 8 && reads < 256,
                    "background {background_count}, invalidated={invalidated}, reads={reads}"
                );
                assert!(
                    retained <= 8,
                    "background cells {background_count}: {retained}"
                );
            } else {
                assert_eq!(retained, 5);
            }
            assert_eq!(trial.quality().cell_count(), reference.tetrahedra.len() + 3);
        }
        let (actual, actual_quality) = state
            .finish(DelaunayVolumeSliverOptions::default(), &NeverCancelled)
            .unwrap();
        assert_eq!(actual, base);
        assert_eq!(actual_quality, reference);
    }
}
