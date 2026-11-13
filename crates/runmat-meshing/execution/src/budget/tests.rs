use super::*;
use runmat_geometry_core::GeometryEvaluationControl;
use runmat_meshing_core::NeverCancelled;

// witness: FEA-SHARED-RESOURCES-001/governed_path_equivalence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Terminal stage evidence retains cumulative shared search and iteration work and peak
//   allocation after the last algorithm checkpoint, without adding duplicate absolute counters or
//   resetting usage. This is accounting evidence, not measured RSS or whole study acceptance.
#[test]
fn terminal_evidence_retains_shared_usage_after_the_last_algorithm_checkpoint() {
    let ceilings = crate::serial_tests::ceilings();
    let fixture = crate::task_tests::Fixture::new(MeshingStageKind::SurfaceMesh);
    let mut progress = NoopMeshingProgress;
    let mut control = MeshingStageControl::new(
        MeshingStageKind::SurfaceMesh,
        0,
        &fixture.request,
        ceilings,
        &NeverCancelled,
        &mut progress,
    )
    .unwrap();
    control
        .checkpoint(MeshingStageCheckpoint {
            peak_memory_bytes: 100,
            search_work: 2,
            iterations: 5,
            ..MeshingStageCheckpoint::default()
        })
        .unwrap();
    let geometry = control.geometry_evaluation_control();
    geometry.consume_allocation_bytes(200).unwrap();
    geometry.consume_search_work(7).unwrap();
    geometry.consume_iterations(3).unwrap();
    geometry.release_allocation_bytes(200);
    let evidence = control.stage_evidence(
        fixture.workload.partition.clone(),
        StableDigest::from_bytes([7; 32]),
    );
    assert_eq!(evidence.peak_memory_bytes, 200);
    assert_eq!(evidence.search_work, 7);
    assert_eq!(evidence.iterations, 5);
    assert_eq!(geometry.usage().live_allocation_bytes, 0);
    assert_eq!(geometry.usage().search_work, 7);
}
