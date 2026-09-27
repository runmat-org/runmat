use super::*;
use runmat_accelerate_api::{GpuTensorStorage, NumericElementType};
use std::sync::atomic::AtomicBool;

mod driver;
mod quarantine;
mod initialization;
mod ownership;
use driver::FakeDriver;

fn owner() -> (Arc<AllocationState>, Arc<FakeDriver>) {
    let driver = Arc::new(FakeDriver::default());
    let owner = Arc::new(AllocationState::with_driver(
        driver.clone(),
        7,
        11,
        runmat_accelerate_api::next_device_id(),
    ));
    (owner, driver)
}

fn allocate(owner: &Arc<AllocationState>, pointer: u64) -> GpuTensorHandle {
    owner
        .allocate(
            vec![1, 2],
            NativeDeviceClass::Numeric(NumericElementType::U64),
            GpuTensorStorage::Real,
            16,
            || Ok(pointer),
            |_| Ok(()),
        )
        .unwrap()
}

#[test]
fn cuda_state_retains_context_until_last_owner_and_native_guard() {
    let (state, driver) = owner();
    let handle = allocate(&state, 101);
    assert_eq!(state.allocation(&handle).unwrap().byte_length, 16);
    assert_eq!(
        state.allocation(&handle).unwrap().descriptor,
        handle.descriptor()
    );
    let guard = state.enter_native().unwrap();
    let weak = Arc::downgrade(&state);
    drop(state);
    assert!(weak.upgrade().is_some());
    assert_eq!(driver.release_count(), 0);
    drop(guard);
    assert!(
        weak.upgrade().is_some(),
        "tensor lease retains the primary context"
    );
    drop(handle);
    assert!(weak.upgrade().is_none());
    assert_eq!(driver.frees(), vec![101]);
    assert_eq!(driver.release_count(), 1);
    assert_eq!(
        driver.events.lock().unwrap().last().copied(),
        Some("release")
    );
}

#[test]
fn cuda_free_runs_outside_record_lock_and_validates_exact_owner() {
    let (state, driver) = owner();
    let handle = allocate(&state, 202);
    let id = handle.buffer_id();
    let weak = Arc::downgrade(&state);
    *driver.on_free.lock().unwrap() = Some(Box::new(move || {
        let state = weak.upgrade().unwrap();
        let records = state
            .allocations
            .try_lock()
            .expect("no driver call under allocation lock");
        assert!(matches!(
            records.get(&id).unwrap().phase,
            Phase::Uncertain(_)
        ));
    }));
    let (other_owner, _) = owner();
    let foreign = allocate(&other_owner, 203);
    assert!(state.allocation(&foreign).is_err());
    assert!(driver.frees().is_empty());
    state.retire_record(handle.buffer_id()).unwrap();
    assert!(state.retire_record(handle.buffer_id()).is_err());
    assert_eq!(driver.frees(), vec![202]);
    assert!(state.records().is_empty());
}

#[test]
fn cuda_entry_failure_retains_record_and_allows_only_unattempted_free_retry() {
    let (state, driver) = owner();
    let handle = allocate(&state, 303);
    driver.fail_push.store(true, Ordering::SeqCst);
    assert!(state.retire_record(handle.buffer_id()).is_err());
    assert!(matches!(
        state.records()[&handle.buffer_id()].phase,
        Phase::EntryFailed(_)
    ));
    assert!(driver.frees().is_empty());
    state.retire_record(handle.buffer_id()).unwrap();
    assert_eq!(driver.frees(), vec![303]);
}

#[test]
fn cuda_uncertain_free_retains_payload_and_is_never_retried_even_at_teardown() {
    let (state, driver) = owner();
    let handle = allocate(&state, 404);
    driver.fail_free.store(true, Ordering::SeqCst);
    assert!(state.retire_record(handle.buffer_id()).is_err());
    assert!(matches!(
        state.records()[&handle.buffer_id()].phase,
        Phase::Uncertain(_)
    ));
    assert_eq!(
        state.records()[&handle.buffer_id()]
            .allocation
            .unwrap()
            .pointer,
        404
    );
    assert!(state.allocation(&handle).is_err());
    assert!(state.retire_record(handle.buffer_id()).is_err());
    drop(handle);
    drop(state);
    assert_eq!(driver.frees(), vec![404]);
    assert_eq!(driver.release_count(), 1);
}

#[test]
fn cuda_pop_failure_preserves_freed_outcome_and_quarantines_subsequent_operations() {
    let (state, driver) = owner();
    let handle = allocate(&state, 505);
    let other = allocate(&state, 506);
    driver.fail_pop.store(true, Ordering::SeqCst);
    let error = state.retire_record(handle.buffer_id()).unwrap_err();
    assert!(error
        .downcast_ref::<context::ContextUnavailable>()
        .is_some());
    assert!(matches!(
        state.records()[&handle.buffer_id()].phase,
        Phase::FreedExitFailed(_)
    ));
    let events = driver.events.lock().unwrap().len();
    assert!(state
        .with_context::<()>(|| panic!("quarantined operation executed"))
        .is_err());
    assert!(state.enter_native().is_err());
    assert!(state.context_identity().is_err());
    assert!(state.allocation(&other).is_err());
    assert!(state.retire_record(handle.buffer_id()).is_err());
    assert_eq!(driver.events.lock().unwrap().len(), events);
    drop(handle);
    drop(other);
    drop(state);
    assert_eq!(
        driver.frees(),
        vec![505],
        "no second free or unsafe teardown of other allocation"
    );
}

#[test]
fn cuda_unpublished_allocation_survives_context_exit_failure_in_same_map() {
    let (state, driver) = owner();
    driver.fail_pop.store(true, Ordering::SeqCst);
    assert!(state
        .allocate(
            vec![1],
            NativeDeviceClass::Numeric(NumericElementType::F32),
            GpuTensorStorage::Real,
            4,
            || Ok(606),
            |_| Ok(()),
        )
        .is_err());
    assert_eq!(state.records()[&1].allocation.unwrap().pointer, 606);
    assert!(matches!(state.records()[&1].phase, Phase::Live));
    drop(state);
    assert!(
        driver.frees().is_empty(),
        "unhealthy context must not execute cleanup"
    );
}

#[test]
fn cuda_native_guard_pop_failure_marks_retained_owner_unavailable() {
    let (state, driver) = owner();
    let guard = state.enter_native().unwrap();
    driver.fail_pop.store(true, Ordering::SeqCst);
    drop(guard);
    assert!(state
        .with_context(|| Ok(()))
        .unwrap_err()
        .downcast_ref::<context::ContextUnavailable>()
        .is_some());
}

#[cfg(panic = "unwind")]
#[test]
fn cuda_context_scope_pops_during_operation_unwind() {
    let (state, driver) = owner();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _: Result<()> = state.with_context(|| panic!("operation unwind"));
    }));
    assert!(result.is_err());
    assert_eq!(*driver.events.lock().unwrap(), vec!["push", "pop"]);
    state.with_context(|| Ok(())).unwrap();
}

#[test]
fn cuda_failed_allocation_leaves_no_phantom_record() {
    let (state, driver) = owner();
    assert!(state
        .allocate(
            vec![1],
            NativeDeviceClass::Numeric(NumericElementType::F64),
            GpuTensorStorage::Real,
            8,
            || Err(anyhow!("allocation failed")),
            |_| Ok(()),
        )
        .is_err());
    assert!(state.records().is_empty());
    assert!(driver.frees().is_empty());
}

#[test]
fn cuda_free_and_exit_failure_preserve_uncertainty_and_context_failure_separately() {
    let (state, driver) = owner();
    let handle = allocate(&state, 707);
    driver.fail_free.store(true, Ordering::SeqCst);
    driver.fail_pop.store(true, Ordering::SeqCst);
    assert!(state
        .retire_record(handle.buffer_id())
        .unwrap_err()
        .to_string()
        .contains("free uncertain"));
    assert!(matches!(
        state.records()[&handle.buffer_id()].phase,
        Phase::Uncertain(_)
    ));
    assert!(state
        .context_identity()
        .unwrap_err()
        .downcast_ref::<context::ContextUnavailable>()
        .is_some());
    drop(handle);
    drop(state);
    assert_eq!(driver.frees(), vec![707]);
}

#[cfg(panic = "unwind")]
#[test]
fn cuda_poisoned_records_remain_accounted_without_driver_cleanup_or_false_success() {
    let (state, driver) = owner();
    let handle = allocate(&state, 808);
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _records = state.records();
        panic!("poison owner bookkeeping");
    }));
    assert!(state
        .retire_record(handle.buffer_id())
        .unwrap_err()
        .downcast_ref::<context::ContextUnavailable>()
        .is_some());
    assert_eq!(
        state.records()[&handle.buffer_id()]
            .allocation
            .unwrap()
            .pointer,
        808
    );
    drop(handle);
    drop(state);
    assert!(driver.frees().is_empty());
    assert_eq!(driver.release_count(), 1);
}

#[test]
fn cuda_initialization_failure_uses_recorded_owner_cleanup() {
    let (state, driver) = owner();
    assert!(state
        .allocate(
            vec![1],
            NativeDeviceClass::Numeric(NumericElementType::F64),
            GpuTensorStorage::Real,
            8,
            || Ok(909),
            |_| Err(anyhow!("initialization failed"))
        )
        .is_err());
    assert!(state.records().is_empty());
    assert_eq!(driver.frees(), vec![909]);
}

#[cfg(panic = "unwind")]
#[test]
fn cuda_unreturned_pointer_reservation_is_removed_on_allocator_unwind() {
    let (state, driver) = owner();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = state.allocate(
            vec![1],
            NativeDeviceClass::Numeric(NumericElementType::F64),
            GpuTensorStorage::Real,
            8,
            || panic!("before pointer returned"),
            |_| Ok(()),
        );
    }))
    .is_err());
    assert!(state.records().is_empty());
    assert!(driver.frees().is_empty());
    state.with_context(|| Ok(())).unwrap();
}

#[cfg(panic = "unwind")]
#[test]
fn cuda_cleanup_unwinds_are_contained_during_existing_unwind() {
    let (state, driver) = owner();
    allocate(&state, 1001);
    driver.panic_free.store(true, Ordering::SeqCst);
    driver.panic_release.store(true, Ordering::SeqCst);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _state = state;
            panic!("outer unwind");
        }))
        .is_err()
    );
    assert_eq!(driver.frees(), vec![1001]);
    assert_eq!(driver.release_count(), 1);
}

#[cfg(panic = "unwind")]
#[test]
fn cuda_context_pop_unwind_is_contained_and_marks_owner_unavailable() {
    let (state, driver) = owner();
    driver.panic_pop.store(true, Ordering::SeqCst);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _: Result<()> = state.with_context(|| panic!("outer operation unwind"));
    }))
    .is_err());
    assert!(state
        .context_identity()
        .unwrap_err()
        .downcast_ref::<context::ContextUnavailable>()
        .is_some());
}

#[cfg(panic = "unwind")]
#[test]
fn cuda_initializer_unwind_retires_unpublished_pointer_exactly_once() {
    let (state, driver) = owner();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = state.allocate(
            vec![1],
            NativeDeviceClass::Numeric(NumericElementType::F64),
            GpuTensorStorage::Real,
            8,
            || Ok(1101),
            |_| panic!("initializer unwind"),
        );
    }))
    .is_err());
    assert!(state.records().is_empty());
    assert_eq!(driver.frees(), vec![1101]);
    assert_eq!(
        *driver.events.lock().unwrap(),
        vec!["push", "pop", "push", "free", "pop"]
    );
    drop(state);
    assert_eq!(driver.frees(), vec![1101]);
}

#[cfg(panic = "unwind")]
#[test]
fn cuda_initializer_unwind_cleanup_failure_remains_uncertain_without_retry() {
    let (state, driver) = owner();
    driver.fail_free.store(true, Ordering::SeqCst);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = state.allocate(
            vec![1],
            NativeDeviceClass::Numeric(NumericElementType::F64),
            GpuTensorStorage::Real,
            8,
            || Ok(1102),
            |_| panic!("initializer unwind"),
        );
    }))
    .is_err());
    assert!(matches!(state.records()[&1].phase, Phase::Uncertain(_)));
    assert_eq!(state.records()[&1].allocation.unwrap().pointer, 1102);
    drop(state);
    assert_eq!(driver.frees(), vec![1102]);
}

#[cfg(panic = "unwind")]
#[test]
fn cuda_initializer_unwind_pop_failure_preserves_unpublished_pointer_without_unsafe_cleanup() {
    let (state, driver) = owner();
    driver.fail_pop.store(true, Ordering::SeqCst);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = state.allocate(
            vec![1],
            NativeDeviceClass::Numeric(NumericElementType::F64),
            GpuTensorStorage::Real,
            8,
            || Ok(1103),
            |_| panic!("initializer unwind"),
        );
    }))
    .is_err());
    assert_eq!(state.records()[&1].allocation.unwrap().pointer, 1103);
    assert!(state
        .context_identity()
        .unwrap_err()
        .downcast_ref::<context::ContextUnavailable>()
        .is_some());
    drop(state);
    assert!(driver.frees().is_empty());
}
