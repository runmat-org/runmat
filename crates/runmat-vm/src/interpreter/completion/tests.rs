use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
};
use runmat_value::Value;

#[test]
fn projection_drop_preserves_output_alias_and_retires_unpublished_values() {
    runmat_gc::gc_test_context(|| {
        let owner = TestAllocationOwner::new();
        let shared = owner
            .allocate_zeroed(vec![1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let old = owner
            .allocate_zeroed(vec![1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let projection = CompletedFrameProjection::new(vec![
            Value::GpuTensor(shared.handle),
            Value::GpuTensor(old.handle),
        ])
        .unwrap();
        let output = projection.values().get(0).unwrap().clone();
        drop(projection);
        assert_eq!(shared.probe.retirements(), 0);
        assert_eq!(old.probe.retirements(), 1);
        drop(output);
        assert_eq!(shared.probe.retirements(), 1);
    });
}

#[test]
fn projection_roots_managed_targets_until_projection_is_dropped() {
    runmat_gc::gc_test_context(|| {
        let target = runmat_gc::gc_allocate_rooted(Value::Num(7.0)).unwrap();
        let identity = target.handle();
        let projection =
            CompletedFrameProjection::new(vec![Value::HandleObject(runmat_value::HandleRef {
                class_name: "ProjectionTarget".into(),
                target: identity,
                valid: true,
            })])
            .unwrap();
        target.unroot().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_with_value(&identity, |_| ()).is_ok());
        drop(projection);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_with_value(&identity, |_| ()).is_err());
    });
}
