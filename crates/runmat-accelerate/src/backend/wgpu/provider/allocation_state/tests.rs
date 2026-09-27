use super::*;
use runmat_accelerate_api::allocation::{
    AllocationAcquisitionError, AllocationIdentity, AllocationOwner, GpuPhysicalDescriptor,
};

fn empty_state() -> AllocationState {
    AllocationState::new(
        runmat_accelerate_api::next_device_id(),
        1,
        1024,
        Arc::new(BindGroupCache::new()),
        Weak::new(),
    )
}

#[test]
fn allocation_state_does_not_strongly_retain_handle_provenance_owner() {
    let cache = Arc::new(BindGroupCache::new());
    let resources = Arc::new(KernelResourceRegistry::new());
    let state = Arc::new(AllocationState::new(
        runmat_accelerate_api::next_device_id(),
        1,
        1024,
        cache.clone(),
        Arc::downgrade(&resources),
    ));
    assert_eq!(Arc::strong_count(&resources), 1);
    assert_eq!(Arc::strong_count(&cache), 2);
    drop(resources);
    assert!(state.kernel_resources.upgrade().is_none());
    drop(state);
    assert_eq!(Arc::strong_count(&cache), 1);
}

#[test]
fn identifiers_do_not_wrap_or_reuse_after_exhaustion() {
    let state = empty_state();
    assert_eq!(state.next_identifier().unwrap(), 1);
    state.next_id.store(u64::MAX - 1, Ordering::Relaxed);
    assert_eq!(state.next_identifier().unwrap(), u64::MAX - 1);
    for _ in 0..2 {
        assert_eq!(
            state
                .next_identifier()
                .unwrap_err()
                .downcast_ref::<AllocationStateError>(),
            Some(&AllocationStateError::IdentifierExhausted)
        );
    }
    assert_eq!(state.next_id.load(Ordering::Relaxed), u64::MAX);
}

#[test]
fn nonowning_identifiers_cannot_acquire_or_validate_unpublished_records() {
    let state = empty_state();
    let physical = GpuPhysicalDescriptor::new(
        runmat_accelerate_api::NumericElementType::F32,
        runmat_accelerate_api::GpuTensorStorage::Real,
        0,
        0,
        4,
    )
    .unwrap();
    for (index, phase) in [
        Phase::Registering(Stage::Registration),
        Phase::Retiring(Stage::Cache),
        Phase::Failed {
            stage: Stage::Cache,
            failure: Failure::Cache,
            disposition: Disposition::Retained,
        },
    ]
    .into_iter()
    .enumerate()
    {
        let id = index as u64 + 1;
        state.buffers.lock().unwrap().insert(
            id,
            Record {
                phase,
                entry: None,
                retained_bytes: 11,
                physical,
                lease: None,
                logical_producer: false,
            },
        );
        assert_eq!(
            state.validate_live(AllocationIdentity::new(state.device_id, id, 1)),
            Err(AllocationAcquisitionError::NotLive)
        );
        assert!(state.retire_physical(id).is_err());
    }
    assert_eq!(state.retained_bytes(), 33);
    assert_eq!(
        state.validate_live(AllocationIdentity::new(state.device_id, 1, 2)),
        Err(AllocationAcquisitionError::StaleGeneration)
    );
}

#[cfg(panic = "unwind")]
#[test]
fn poisoned_map_rejects_acquisition_without_panicking() {
    let state = empty_state();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = state.buffers.lock().unwrap();
        panic!("injected map poison");
    }))
    .is_err());
    assert_eq!(
        state.validate_live(AllocationIdentity::new(state.device_id, 1, 1)),
        Err(AllocationAcquisitionError::NotLive)
    );
    assert!(state.retire_physical(1).is_err());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn device_loss_is_not_observation_worker_stop() {
    let state = empty_state();
    let observation_stopped = AtomicBool::new(false);
    observation_stopped.store(true, Ordering::Release);
    assert!(!state.is_lost());
    state.mark_device_lost();
    assert!(state.is_lost());
    assert_eq!(
        state.validate_live(AllocationIdentity::new(state.device_id, 1, 1)),
        Err(AllocationAcquisitionError::NotLive)
    );
}
