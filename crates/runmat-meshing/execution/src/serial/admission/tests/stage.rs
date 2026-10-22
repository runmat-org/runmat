//! Current immutable closures must survive cold import without retaining the
//! scratch used to validate them. Denials retain the original source category.
use super::controlled::RecordingControl;
use super::*;
use runmat_execution::value::ValuePayload;

fn exercise(
    fail_allocation: Option<usize>,
    fail_checkpoint: Option<usize>,
    kind: runmat_geometry_core::GeometryEvaluationErrorKind,
) -> (usize, usize) {
    let objects = crate::tests::fixture(vec![vec![3; 1024], vec![7; 2048]], 1024);
    let access = crate::ArtifactAccess {
        authorization_scope: "retained-stage-owner".into(),
        encryption_context: Digest::sha256(b"retained-stage-owner"),
    };
    let limits = runmat_execution_artifact::object::ObjectInventoryLimits::default();
    let expected = crate::prepare_result_publication(objects, access.clone(), limits).unwrap();
    let ValuePayload::Object(root) = expected.root_output() else {
        unreachable!()
    };
    let mut cache = crate::tests::MemoryCache::default();
    cache.insert_all(&expected.stage_objects().objects);
    let control = RecordingControl::new(fail_allocation, fail_checkpoint, kind);
    {
        let scope = GeometryAllocationScope::new(&control);
        let reader = InputReader::new(&cache, &scope);
        let imported = super::super::stage::import(&reader, root, &access, limits);
        if fail_allocation.is_some() || fail_checkpoint.is_some() {
            assert!(imported.is_err());
            let error = reader
                .take_failure()
                .expect("stage import must preserve original control failure");
            assert_eq!(error.kind, kind);
            assert_eq!(error.reason, "injected original input control failure");
        } else {
            let actual = imported.as_ref().unwrap();
            assert_eq!(actual, &expected);
            let retained = actual.retained_storage_bytes(&control).unwrap()
                - std::mem::size_of::<crate::PreparedMeshingResultPublication>() as u64;
            assert_eq!(control.inner.usage().live_allocation_bytes, retained);
            assert!(
                control.inner.usage().allocation_bytes > retained,
                "verification scratch must stop being charged before the next input"
            );
        }
        drop(imported);
    }
    assert_eq!(control.inner.usage().live_allocation_bytes, 0);
    let allocations = control.allocations.lock().unwrap().len();
    (allocations, control.checkpoints.load(Ordering::Relaxed))
}
// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Complete immutable stage closures preserve their original objects and reference
//   authority while verification scratch releases before the next input. Every tested allocation
//   denial and sampled time/cancellation boundaries preserve their original cause and release owned
//   charges. This finite fixture does not establish arbitrary closure capacity or whole-study
//   performance.
#[test]
fn immutable_stage_import_transfers_only_original_retention_and_preserves_denials() {
    use runmat_geometry_core::GeometryEvaluationErrorKind::*;
    let (allocations, checkpoints) = exercise(None, None, AllocationBudgetExceeded);
    assert!(allocations > 5 && checkpoints > 5);
    for index in 0..allocations {
        exercise(Some(index), None, AllocationBudgetExceeded);
    }
    for kind in [Cancelled, TimeBudgetExceeded] {
        for index in [0, checkpoints / 2] {
            exercise(None, Some(index), kind);
        }
    }
}
