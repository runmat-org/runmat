use super::super::owner_shared::OwnerShared;
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
        OwnerWeak::new(),
    )
}

#[test]
fn reset_retry_resumes_only_proven_pre_destroy_admission_failure() {
    let retry = Phase::Failed {
        stage: Stage::Resource,
        failure: Failure::DestroyAdmission,
        disposition: Disposition::Retained,
    };
    assert_eq!(retry.retirement_start(true), Some(Stage::Resource));
    assert_eq!(retry.retirement_start(false), None);
    assert_eq!(Phase::Live.retirement_start(true), Some(Stage::Cache));
    for stage in [
        Stage::Cache,
        Stage::Provenance,
        Stage::Resource,
        Stage::Complete,
    ] {
        for failure in [
            Failure::MapPoisoned,
            Failure::Cache,
            Failure::Provenance,
            Failure::Unwind,
            Failure::DestroyAdmission,
        ] {
            for disposition in [
                Disposition::Retained,
                Disposition::Uncertain,
                Disposition::Released,
            ] {
                let phase = Phase::Failed {
                    stage,
                    failure,
                    disposition,
                };
                if phase != retry {
                    assert_eq!(phase.retirement_start(true), None);
                }
            }
        }
        assert_eq!(Phase::Retiring(stage).retirement_start(true), None);
        assert_eq!(Phase::Registering(stage).retirement_start(true), None);
    }
}

#[cfg(all(not(target_arch = "wasm32"), target_pointer_width = "64"))]
#[test]
#[ignore = "requires a WGPU adapter and RUNMAT_WGPU_FORCE_PRECISION=f32"]
fn overflowing_pool_acquisition_rejects_before_buffer_or_pool_state_changes() {
    use crate::backend::wgpu::provider::{WgpuProvider, WgpuProviderOptions};
    use runmat_accelerate_api::{GpuTensorStorage, HostNumericDataView, HostNumericTensorView};

    let provider = WgpuProvider::new(WgpuProviderOptions::default())
        .expect("qualification requires a real WGPU adapter");
    assert_eq!(
        provider.precision,
        crate::backend::wgpu::types::NumericPrecision::F32
    );
    let input = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 2.0]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .expect("borrowed source witness");
    let before_records = provider.allocations.record_count().unwrap();
    let before_pool_bytes = provider.allocations.pooled_bytes(provider.element_size);
    let weak = input.downgrade();
    let rejected = provider
        .with_emission(|emission, _| {
            provider.allocations.acquire(
                emission,
                BufferUsageClass::FusionOut,
                usize::MAX,
                8,
                "overflowing-pool-witness",
            )
        })
        .expect("shared emission access remains valid");
    assert!(rejected.is_err());
    let rejected_public_owner = provider
        .with_emission(|emission, _| {
            provider.create_storage_buffer_for_usage(
                emission,
                BufferUsageClass::FusionOut,
                usize::MAX,
                "overflowing-provider-witness",
            )
        })
        .expect("shared emission access remains valid");
    assert!(rejected_public_owner.is_err());
    assert_eq!(provider.allocations.record_count().unwrap(), before_records);
    assert_eq!(
        provider.allocations.pooled_bytes(provider.element_size),
        before_pool_bytes
    );
    input.allocation_lease().validate_live().unwrap();
    assert!(weak.acquire().is_ok());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn allocation_state_does_not_strongly_retain_handle_provenance_owner() {
    let cache = Arc::new(BindGroupCache::new());
    let resources = OwnerShared::new(KernelResourceRegistry::new());
    let state = OwnerShared::new(AllocationState::new(
        runmat_accelerate_api::next_device_id(),
        1,
        1024,
        cache.clone(),
        OwnerShared::downgrade(&resources),
    ));
    assert_eq!(OwnerShared::strong_count(&resources), 1);
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
                scope: None,
                registration_slot: None,
                phase,
                entry: None,
                retained_bytes: 11,
                physical,
                lease: None,
                resource_ticket: None,
                logical_producer: false,
                host_metadata_claim: None,
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
