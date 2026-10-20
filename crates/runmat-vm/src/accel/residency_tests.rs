//! Value/container lifetime regressions: no global provider or manual free.
use crate::interpreter::CompletedFrameProjection;
use runmat_accelerate_api::{
    test_support::{TestAllocation, TestAllocationOwner},
    GpuTensorHandle, GpuTensorStorage, NumericElementType,
};
use runmat_value::{CellArray, Closure, HandleRef, Value};

fn allocation(owner: &TestAllocationOwner) -> TestAllocation {
    owner
        .allocate_zeroed(vec![1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap()
}

fn nested_cell(handle: GpuTensorHandle) -> Value {
    Value::Cell(CellArray::new(vec![Value::GpuTensor(handle)], 1, 1).unwrap())
}

fn captured(handle: GpuTensorHandle) -> Value {
    Value::Closure(Closure::new(
        "worker".into(),
        None,
        runmat_types::FunctionSourceRole::Unknown,
        None,
        vec![Value::GpuTensor(handle)],
        None,
    ))
}

#[test]
fn cells_and_closure_captures_retain_their_own_allocation_references() {
    for wrap in [nested_cell as fn(GpuTensorHandle) -> Value, captured] {
        let owner = TestAllocationOwner::new();
        let owned = allocation(&owner);
        let weak = owned.handle.downgrade();
        let nested = wrap(owned.handle.clone());
        drop(owned.handle);
        assert_eq!(owned.probe.retirements(), 0);
        let alias = weak.acquire().unwrap();
        owned
            .probe
            .with_data(&alias, |data| assert_eq!(data.len(), 1))
            .unwrap();
        drop(alias);
        drop(nested);
        assert_eq!(owned.probe.retirements(), 1);
        assert!(weak.acquire().is_err());
    }
}

#[test]
fn direct_values_retire_only_after_the_last_alias_is_dropped() {
    let owner = TestAllocationOwner::new();
    let owned = allocation(&owner);
    let value = Value::GpuTensor(owned.handle);
    let alias = value.clone();
    drop(value);
    assert_eq!(owned.probe.retirements(), 0);
    drop(alias);
    assert_eq!(owned.probe.retirements(), 1);
}

#[test]
fn managed_target_retains_nested_allocation_until_collection() {
    runmat_gc::gc_test_context(|| {
        let owner = TestAllocationOwner::new();
        let owned = allocation(&owner);
        let mut payload = runmat_value::UnboundedStructValueBuilder::with_capacity(1);
        payload.insert("nested", Value::GpuTensor(owned.handle));
        let target = runmat_gc::gc_allocate_rooted(Value::Struct(payload.finish())).unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert_eq!(owned.probe.retirements(), 0);
        target.unroot().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert_eq!(owned.probe.retirements(), 1);
    });
}

#[test]
fn completed_projection_preserves_alias_in_a_managed_output_target() {
    runmat_gc::gc_test_context(|| {
        let owner = TestAllocationOwner::new();
        let shared = allocation(&owner);
        let old = allocation(&owner);
        let mut payload = runmat_value::UnboundedStructValueBuilder::with_capacity(1);
        payload.insert("nested", Value::GpuTensor(shared.handle.clone()));
        let target = runmat_gc::gc_allocate_rooted(Value::Struct(payload.finish())).unwrap();
        let output = Value::HandleObject(HandleRef {
            class_name: "Payload".into(),
            target: target.handle(),
            valid: true,
        });
        let projection = CompletedFrameProjection::new(vec![
            Value::GpuTensor(shared.handle),
            Value::GpuTensor(old.handle),
        ])
        .unwrap();
        drop(projection);
        assert_eq!(old.probe.retirements(), 1);
        assert_eq!(shared.probe.retirements(), 0);
        drop(output);
        target.unroot().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert_eq!(shared.probe.retirements(), 1);
    });
}
