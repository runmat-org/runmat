//! Genuine coupled geometry: compliant protected surface edges enclose a long
//! interior edge and narrow boundary collars. No quality evidence is fabricated.
use super::*;
use crate::cdt::test_fixtures::coupled_quality::{fixture, metric};
use crate::cdt::*;
use runmat_meshing_core::NeverCancelled;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A genuine protected octahedral collar with simultaneous size/shape defects converges
//   under default quality criteria for uniform/scoped metrics and identity permutations. Each case
//   observes one shared-state preparation and one terminal handoff, preserves boundary coordinates
//   and analytic volume, and independently validates topology/quality. This finite fixture does not
//   certify complete relocation costs, original CAD convergence or whole-task resource demand.
#[test]
fn compliant_surface_with_long_interior_edge_and_thin_collars_converges() {
    for permuted in [false, true] {
        for scoped in [false, true] {
            let (topology, provenance) = fixture(permuted, 0);
            let fixed = provenance
                .facets
                .iter()
                .flat_map(|face| face.node_identities)
                .collect::<std::collections::BTreeSet<_>>();
            let boundary = topology
                .nodes
                .iter()
                .filter(|node| fixed.contains(&node.identity))
                .copied()
                .collect::<Vec<_>>();
            let request = metric(&provenance, scoped);
            let mut options = DelaunayVolumeMeshOptions::default();
            options.refinement.maximum_insertions = 64;
            options.sliver.maximum_passes = 64;
            let original = evaluate_delaunay_volume_quality(
                &topology,
                &request,
                &provenance,
                options.quality,
                &NeverCancelled,
            )
            .unwrap();
            assert!(original.worst_refinement_tetrahedron.is_some());
            assert!(original.worst_optimization_tetrahedron.is_some());
            let result = converge_volume_quality(
                topology,
                original,
                &request,
                provenance,
                &[],
                options,
                crate::cdt::volume_mesh::quality_convergence::QualityExecution::without_observer(
                    &NeverCancelled,
                ),
            )
            .unwrap();
            assert_eq!(
                result.lifecycle_counts,
                (1, 1),
                "a local collar repair repeatedly reconstructed complete state"
            );
            assert!(result.quality.worst_refinement_tetrahedron.is_none());
            assert!(result.quality.worst_optimization_tetrahedron.is_none());
            assert!(result.quality_mutations.iter().any(|step| matches!(
                step,
                super::super::DelaunayVolumeQualityMutation::Optimized(_)
            )));
            for node in boundary {
                let current = result
                    .topology
                    .nodes
                    .iter()
                    .find(|n| n.identity == node.identity)
                    .unwrap();
                assert_eq!(*current, node, "published surface moved");
            }
            let faces = result
                .provenance
                .facets
                .iter()
                .map(|face| face.node_identities)
                .collect::<Vec<_>>();
            crate::cdt::insertion::validate_conforming_volume_topology(
                &result.topology,
                &faces,
                options.refinement.step.insertion,
                &NeverCancelled,
            )
            .unwrap();
            validate_delaunay_volume_quality(
                &result.topology,
                &request,
                &result.provenance,
                &result.quality,
                options.quality,
                &NeverCancelled,
            )
            .unwrap();
            let volume = result
                .topology
                .tetrahedra
                .iter()
                .map(|cell| {
                    let points = cell.vertex_indices.map(|i| {
                        result.topology.nodes[i as usize]
                            .coordinates_m
                            .canonical_value()
                    });
                    runmat_meshing_core::predicate::tetrahedron_volume(points).abs()
                })
                .sum::<f64>();
            assert!(
                (volume - 4. / 3.).abs() < 1e-12,
                "changed physical domain: {volume}"
            );
        }
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: The coupled protected-collar fixture rejects pre-cancelled execution and an
//   insufficient original candidate-evaluation cap without returning a mesh. This covers these
//   failure boundaries, not every interruption point or native process quiescence.
#[test]
fn cancellation_and_original_candidate_budget_fail_without_a_mesh() {
    struct Cancelled;
    impl MeshingCancellationSignal for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }
    for cancelled in [false, true] {
        let (topology, provenance) = fixture(false, 0);
        let request = metric(&provenance, false);
        let mut options = DelaunayVolumeMeshOptions::default();
        if !cancelled {
            options
                .refinement
                .step
                .candidate
                .maximum_candidate_evaluations = 1;
        }
        let quality = evaluate_delaunay_volume_quality(
            &topology,
            &request,
            &provenance,
            options.quality,
            &NeverCancelled,
        )
        .unwrap();
        let signal: &dyn MeshingCancellationSignal = if cancelled {
            &Cancelled
        } else {
            &NeverCancelled
        };
        let Err(failure) = converge_volume_quality(
            topology,
            quality,
            &request,
            provenance,
            &[],
            options,
            crate::cdt::volume_mesh::quality_convergence::QualityExecution::without_observer(
                signal,
            ),
        ) else {
            panic!("unadmitted work returned a mesh");
        };
        assert_eq!(
            failure.kind,
            if cancelled {
                DelaunayVolumeMeshErrorKind::Cancelled
            } else {
                DelaunayVolumeMeshErrorKind::ResourceLimit
            }
        );
    }
}

// witness: FEA-MESH-DISCRETIZATION-001/volume_quality_convergence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Exhausted insertion or repair pass budgets fail without publishing a mesh;
//   convergence on the bracket, heat sink and fork within the authored time budget is not
//   demonstrated here.
#[test]
fn exhausted_insertion_or_repair_pass_budget_fails_without_a_mesh() {
    for exhausted_refinement in [true, false] {
        let (topology, provenance) = fixture(false, 0);
        let request = metric(&provenance, false);
        let mut options = DelaunayVolumeMeshOptions::default();
        if exhausted_refinement {
            options.refinement.maximum_insertions = 0;
        } else {
            options.sliver.maximum_passes = 0;
        }
        let quality = evaluate_delaunay_volume_quality(
            &topology,
            &request,
            &provenance,
            options.quality,
            &NeverCancelled,
        )
        .unwrap();
        let Err(failure) = converge_volume_quality(
            topology,
            quality,
            &request,
            provenance,
            &[],
            options,
            crate::cdt::volume_mesh::quality_convergence::QualityExecution::without_observer(
                &NeverCancelled,
            ),
        ) else {
            panic!("a declared budget of zero returned a mesh");
        };
        assert_eq!(
            failure.kind,
            DelaunayVolumeMeshErrorKind::ResourceLimit,
            "refinement budget exhausted: {exhausted_refinement}: {failure:?}"
        );
    }
}
