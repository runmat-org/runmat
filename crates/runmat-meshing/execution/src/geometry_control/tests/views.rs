use super::*;
use crate::{MeshingStageControl, NoopMeshingProgress};
use runmat_geometry_core::GeometryAllocationReservation;
use runmat_meshing_core::MeshingStageKind;

#[test]
fn stage_views_share_retention_work_and_rejection_without_reset() {
    let mut ceilings = crate::serial_tests::ceilings();
    let request = crate::serial_tests::request();
    ceilings.maximum_memory_bytes = 100;
    ceilings.maximum_search_work = 2;
    ceilings.maximum_iterations = 2;
    let mut progress = NoopMeshingProgress;
    let stage = MeshingStageControl::new(
        MeshingStageKind::OrderElevation,
        0,
        &request,
        ceilings,
        &NeverCancelled,
        &mut progress,
    )
    .unwrap();
    let first = stage.geometry_evaluation_control();
    let second = stage.geometry_evaluation_control();
    let retained = GeometryAllocationReservation::new(&first, 60).unwrap();
    {
        let scratch = GeometryAllocationReservation::new(&second, 40).unwrap();
        assert_eq!(first.usage().live_allocation_bytes, 100);
        assert!(GeometryAllocationReservation::new(&first, 1).is_err());
        drop(scratch);
    }
    assert_eq!(second.usage().live_allocation_bytes, 60);
    for view in [&first, &second] {
        view.consume_search_work(1).unwrap();
        view.consume_iterations(1).unwrap();
    }
    assert!(first.consume_search_work(1).is_err());
    assert!(second.consume_iterations(1).is_err());
    drop(retained);
    drop(first);
    drop(second);
    let later = stage.geometry_evaluation_control();
    assert_eq!(later.usage().live_allocation_bytes, 0);
    assert_eq!(later.usage().allocation_bytes, 100);
    assert_eq!(later.usage().search_work, 2);
    assert_eq!(later.usage().iterations, 2);
}
