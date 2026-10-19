use super::*;
use crate::simple_provider::allocation_state::{NumericAllocationState, NumericOwner};
use runmat_accelerate_api::allocation::AllocationIdentity;
use runmat_accelerate_api::GpuTensorStorage;

fn owned(logical: bool) -> (NumericOwner, GpuTensorHandle) {
    let state = NumericOwner::new(NumericAllocationState::default());
    let id = AllocationIdentity::new(runmat_accelerate_api::next_device_id(), 1, 1);
    let handle = state
        .stage(
            id,
            InProcessNumericData::F64(vec![0.0, 1.0]),
            vec![1, 2],
            GpuTensorStorage::Real,
            logical,
        )
        .unwrap()
        .publish()
        .unwrap();
    (state, handle)
}

#[test]
fn borrowed_guard_keeps_payload_alive_and_unlocks_before_final_lease_drop() {
    let (state, handle) = owned(false);
    let wire = handle.wire_descriptor();
    let guard = F64RegistryGuard(state.lock().unwrap());
    let data = guard.get(&handle).unwrap();
    let pointer = data.as_ptr();
    drop(handle);
    assert_eq!(data, &[0.0, 1.0]);
    assert_eq!(data.as_ptr(), pointer);
    drop(guard);
    assert!(state.allocations.try_lock().unwrap().is_empty());
    assert!(state.acquire(&wire).is_err());
}

#[test]
fn repeated_reads_retain_one_lease_and_foreign_handle_never_aliases_by_number() {
    let (state, handle) = owned(false);
    let (_other, foreign) = owned(false);
    assert_eq!(handle.buffer_id(), foreign.buffer_id());
    let guard = state.lock().unwrap();
    assert!(guard.get(&handle).is_some());
    assert!(guard.get(&handle).is_some());
    assert_eq!(guard.retained.borrow().len(), 1);
    assert!(guard.get(&foreign).is_none());
    assert_eq!(guard.retained.borrow().len(), 1);
}

#[test]
fn logical_allocation_cannot_be_mutated_arbitrarily_through_numeric_view() {
    let (state, logical) = owned(true);
    let numeric = logical
        .with_view(logical.view().clone().with_logical(false))
        .unwrap();
    let mut guard = F64RegistryGuard(state.lock().unwrap());
    assert!(guard.get_mut_for_write(&numeric, &[2.0]).is_none());
    assert_eq!(guard.get(&logical).unwrap(), &[0.0, 1.0]);
    guard.get_mut_for_write(&numeric, &[1.0]).unwrap()[0] = 1.0;
    assert_eq!(guard.get(&logical).unwrap(), &[1.0, 1.0]);
}

#[test]
fn reentrant_retention_borrow_fails_closed_without_dropping_existing_owners() {
    let (state, handle) = owned(false);
    let guard = state.lock().unwrap();
    assert!(guard.get(&handle).is_some());
    let retained = guard.retained.borrow_mut();
    assert!(guard.get(&handle).is_none());
    assert_eq!(retained.len(), 1);
    drop(retained);
    assert!(guard.get(&handle).is_some());
}
