use super::*;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: An independently evaluated admitted tetrahedron retains the same node, cell and
//   quality allocations through owned refinement and optimization handoffs, with no mutations and
//   independently recomputed terminal quality. This narrow ownership witness does not measure
//   whole-task memory or establish general quality convergence.
#[test]
fn admitted_state_moves_through_refinement_and_optimization_without_mesh_copies() {
    let topology = topology([
        [0.0, 0.0, 0.0],
        [0.25, 0.0, 0.0],
        [0.0, 0.25, 0.0],
        [0.0, 0.0, 0.25],
    ]);
    let provenance = context(&topology);
    let metric = metric();
    let options = DelaunayVolumeQualityOptions::default();
    let quality =
        evaluate_delaunay_volume_quality(&topology, &metric, &provenance, options, &NeverCancelled)
            .unwrap();
    assert!(quality.worst_refinement_tetrahedron.is_none());
    assert!(quality.worst_optimization_tetrahedron.is_none());
    let nodes = topology.nodes.as_ptr();
    let tetrahedra = topology.tetrahedra.as_ptr();
    let qualities = quality.tetrahedra.as_ptr();
    let authority = super::super::context::RefinementContext {
        metric_request: &metric,
        provenance: &provenance,
        quality_options: options,
        fixed_node_identities: &[],
    };
    let refined = super::super::iteration::refine_owned_until_stalled(
        topology,
        quality,
        authority,
        DelaunayVolumeRefinementOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert!(!refined.stalled);
    assert!(refined.refinement.insertions.is_empty());
    let optimized = crate::cdt::volume_sliver::run_treatment(
        refined.refinement.topology,
        refined.refinement.quality,
        authority,
        crate::cdt::DelaunayVolumeSliverOptions::default(),
        &NeverCancelled,
        crate::cdt::volume_sliver::QualityConvergenceMode::Coupled,
    )
    .unwrap();
    assert!(optimized.mutations.is_empty());
    assert_eq!(optimized.topology.nodes.as_ptr(), nodes);
    assert_eq!(optimized.topology.tetrahedra.as_ptr(), tetrahedra);
    assert_eq!(optimized.quality.tetrahedra.as_ptr(), qualities);
    // Allocation preservation is the handoff claim; independently recompute the
    // physical quality so an unchanged pointer cannot stand in for admission.
    assert_eq!(
        optimized.quality,
        evaluate_delaunay_volume_quality(
            &optimized.topology,
            &metric,
            &provenance,
            options,
            &NeverCancelled,
        )
        .unwrap()
    );
}
