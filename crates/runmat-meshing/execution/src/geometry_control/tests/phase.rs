//! Validate lifetime transitions against the same real stage ceiling, including
//! imported input ownership and a still-live independent kernel view.
use super::*;
use crate::{MeshingStageControl, NoopMeshingProgress};
use runmat_meshing_core::{MeshingChunkMediaType, MeshingChunkStream, MeshingStageKind};
fn records(bytes: usize) -> Vec<MeshingChunkStream> {
    vec![MeshingChunkStream {
        media_type: MeshingChunkMediaType::SurfaceMesh,
        schema_version: 1,
        records: vec![vec![1; bytes]],
    }]
}
// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: All kernel views share the stage ceiling. Imported inputs remain charged while
//   obsolete kernel storage releases and admitted returned records retain their original capacity
//   charge through their last use. This finite ownership witness does not establish model capacity
//   or measured RSS.
#[test]
fn publication_releases_kernel_scratch_and_drops_record_charge_at_last_use() {
    let mut ceilings = crate::serial_tests::ceilings();
    let request = crate::serial_tests::request();
    ceilings.maximum_memory_bytes = 200;
    let mut progress = NoopMeshingProgress;
    let mut stage = MeshingStageControl::new(
        MeshingStageKind::SurfaceMesh,
        0,
        &request,
        ceilings,
        &NeverCancelled,
        &mut progress,
    )
    .unwrap();
    let input = stage.geometry_evaluation_control();
    let input_storage =
        runmat_geometry_core::GeometryAllocationReservation::new(&input, 100).unwrap();
    stage.begin_kernel_allocations().unwrap();
    {
        let first = stage.geometry_evaluation_control();
        let second = stage.geometry_evaluation_control();
        first.consume_allocation_bytes(60).unwrap(); // Retained scratch, freed at phase end.
        second.consume_allocation_bytes(40).unwrap();
        assert_eq!(input.usage().live_allocation_bytes, 200);
        assert!(second.consume_allocation_bytes(1).is_err());
    }
    let streams = records(40);
    let storage = stage.finish_kernel_allocations(&streams).unwrap();
    assert_eq!(input.usage().live_allocation_bytes, 140);
    assert_eq!(input.usage().allocation_bytes, 200);
    drop(streams);
    drop(storage);
    assert_eq!(input.usage().live_allocation_bytes, 100);
    drop(input_storage);
    assert_eq!(input.usage().live_allocation_bytes, 0);
}
// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A returned record inventory exceeding retained kernel admission rejects without
//   discarding owned charges before error cleanup. This checks the stage ownership boundary, not
//   every concrete encoder.
#[test]
fn unadmitted_record_transfer_rejects_and_keeps_ownership_until_error_cleanup() {
    let ceilings = crate::serial_tests::ceilings();
    let request = crate::serial_tests::request();
    let mut progress = NoopMeshingProgress;
    let mut stage = MeshingStageControl::new(
        MeshingStageKind::SurfaceMesh,
        0,
        &request,
        ceilings,
        &NeverCancelled,
        &mut progress,
    )
    .unwrap();
    let observer = stage.geometry_evaluation_control();
    stage.begin_kernel_allocations().unwrap();
    {
        let kernel = stage.geometry_evaluation_control();
        kernel.consume_allocation_bytes(10).unwrap();
    }
    let streams = records(20);
    let failure = stage.finish_kernel_allocations(&streams).err().unwrap();
    assert_eq!(
        failure.category,
        runmat_meshing_core::MeshingFailureCategory::InternalInvariantViolation
    );
    assert_eq!(observer.usage().live_allocation_bytes, 10);
    drop(streams);
    drop(stage);
    assert_eq!(observer.usage().live_allocation_bytes, 0);
}
#[test]
fn surviving_kernel_view_preserves_its_own_storage_but_new_views_use_publication_owner() {
    let ceilings = crate::serial_tests::ceilings();
    let request = crate::serial_tests::request();
    let mut progress = NoopMeshingProgress;
    let mut stage = MeshingStageControl::new(
        MeshingStageKind::SurfaceMesh,
        0,
        &request,
        ceilings,
        &NeverCancelled,
        &mut progress,
    )
    .unwrap();
    stage.begin_kernel_allocations().unwrap();
    let kernel = stage.geometry_evaluation_control();
    kernel.consume_allocation_bytes(40).unwrap();
    let streams = records(20);
    let storage = stage.finish_kernel_allocations(&streams).unwrap();
    let publication = stage.geometry_evaluation_control();
    assert_eq!(publication.usage().live_allocation_bytes, 40);
    let publication_scratch =
        runmat_geometry_core::GeometryAllocationReservation::new(&publication, 30).unwrap();
    drop(kernel);
    assert_eq!(publication.usage().live_allocation_bytes, 50);
    drop(streams);
    drop(storage);
    assert_eq!(publication.usage().live_allocation_bytes, 30);
    drop(publication_scratch);
    assert_eq!(publication.usage().live_allocation_bytes, 0);
}
