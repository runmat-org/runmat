use super::*;

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
fn identifiers_do_not_wrap_or_reuse_after_exhaustion() {
    let state = empty_state();
    assert_eq!(state.next_identifier().unwrap(), 1);
    state.next_id.store(u64::MAX - 1, Ordering::Relaxed);
    assert_eq!(state.next_identifier().unwrap(), u64::MAX - 1);
    for _ in 0..2 {
        let error = state.next_identifier().unwrap_err();
        assert_eq!(
            error.downcast_ref::<AllocationStateError>(),
            Some(&AllocationStateError::IdentifierExhausted)
        );
    }
    assert_eq!(state.next_id.load(Ordering::Relaxed), u64::MAX);
}

#[test]
fn unpublished_retiring_and_failed_records_are_not_live() {
    let state = empty_state();
    let phases = [
        Phase::Registering(Stage::RegistrationResidency),
        Phase::Retiring(Stage::Cache),
        Phase::Failed {
            stage: Stage::Residency,
            failure: Failure::Callback,
            disposition: Disposition::Retained,
        },
    ];
    for (index, phase) in phases.into_iter().enumerate() {
        let id = index as u64 + 1;
        state.buffers.lock().unwrap().insert(
            id,
            Record {
                phase,
                entry: None,
                retained_bytes: 11,
            },
        );
        let error = state
            .live_entry(id)
            .err()
            .expect("non-live record rejected");
        assert_eq!(
            error.downcast_ref::<AllocationStateError>(),
            Some(&AllocationStateError::NotLive)
        );
        let handle = GpuTensorHandle::new(vec![1], state.device_id, id);
        let error = state
            .free_with_residency(&handle, |_| panic!("no callback for non-live record"))
            .unwrap_err();
        assert_eq!(
            error.downcast_ref::<AllocationStateError>(),
            Some(&AllocationStateError::NotLive)
        );
    }
    assert_eq!(state.retained_bytes(), 33);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn device_loss_is_explicit_owner_state_not_observation_progress_state() {
    let state = empty_state();
    let observation_stopped = AtomicBool::new(false);
    observation_stopped.store(true, Ordering::Release);
    assert!(!state.is_lost());
    state.mark_device_lost();
    assert!(state.is_lost());
    assert_eq!(
        state
            .live_entry(1)
            .err()
            .unwrap()
            .downcast_ref::<AllocationStateError>(),
        Some(&AllocationStateError::DeviceLost)
    );
}

#[cfg(all(not(target_arch = "wasm32"), panic = "unwind"))]
#[test]
#[ignore = "requires an explicitly provisioned WGPU adapter; run with --ignored"]
fn checked_registration_and_retirement_callbacks_are_unpublished_and_nonrepeating() {
    use super::super::backend_types::{WgpuProvider, WgpuProviderOptions};
    use runmat_accelerate_api::{AccelProvider, HostTensorView};
    let provider =
        WgpuProvider::new(WgpuProviderOptions::default()).expect("required WGPU adapter");
    let original = provider
        .upload(&HostTensorView {
            data: &[2.0, 4.0],
            shape: &[1, 2],
        })
        .unwrap();
    let entry = provider.get_entry_raw(&original).unwrap();
    let id = provider.allocations.next_id.load(Ordering::Relaxed);
    let calls = std::cell::Cell::new(0);
    let error = provider
        .allocations
        .register_with_residency(entry, |handle| {
            calls.set(calls.get() + 1);
            assert!(provider.allocations.live_entry(handle.buffer_id).is_err());
            panic!("registration callback failed");
        })
        .unwrap_err();
    assert_eq!(
        error.downcast_ref::<AllocationStateError>(),
        Some(&AllocationStateError::Cleanup {
            stage: Stage::RegistrationResidency,
            failure: Failure::Callback
        })
    );
    assert_eq!(calls.get(), 1);
    assert_eq!(
        provider
            .allocations
            .buffers
            .lock()
            .unwrap()
            .get(&id)
            .unwrap()
            .phase,
        Phase::Failed {
            stage: Stage::RegistrationResidency,
            failure: Failure::Callback,
            disposition: Disposition::Retained
        }
    );
    assert_eq!(
        futures::executor::block_on(provider.download(&original))
            .unwrap()
            .data,
        vec![2.0, 4.0]
    );

    let fresh = provider
        .upload(&HostTensorView {
            data: &[8.0],
            shape: &[1, 1],
        })
        .unwrap();
    let retired_calls = std::cell::Cell::new(0);
    assert!(provider
        .allocations
        .free_with_residency(&fresh, |handle| {
            retired_calls.set(retired_calls.get() + 1);
            assert!(provider.allocations.live_entry(handle.buffer_id).is_err());
            assert!(provider.allocations.free(handle).is_err());
            panic!("retirement callback failed");
        })
        .is_err());
    assert!(provider
        .allocations
        .free_with_residency(&fresh, |_| {
            retired_calls.set(retired_calls.get() + 1);
            Ok(())
        })
        .is_err());
    assert_eq!(retired_calls.get(), 1);
    assert_eq!(
        provider
            .allocations
            .buffers
            .lock()
            .unwrap()
            .get(&fresh.buffer_id)
            .unwrap()
            .phase,
        Phase::Failed {
            stage: Stage::Residency,
            failure: Failure::Unwind,
            disposition: Disposition::Retained
        }
    );
    provider.free(&original).unwrap();
    let later = provider
        .upload(&HostTensorView {
            data: &[9.0],
            shape: &[1, 1],
        })
        .unwrap();
    let later_calls = std::cell::Cell::new(0);
    let error = provider
        .allocations
        .free_with_residency(&later, |_| {
            later_calls.set(later_calls.get() + 1);
            assert!(catch_allocation_cleanup_unwind(|| {
                let _guard = provider.allocations.buffers.lock().unwrap();
                panic!("bookkeeping poisoned after residency callback");
            })
            .is_err());
            Ok(())
        })
        .unwrap_err();
    assert_eq!(
        error.downcast_ref::<AllocationStateError>(),
        Some(&AllocationStateError::Poisoned)
    );
    assert!(provider
        .allocations
        .free_with_residency(&later, |_| {
            later_calls.set(later_calls.get() + 1);
            Ok(())
        })
        .is_err());
    assert_eq!(later_calls.get(), 1);
    let records = provider
        .allocations
        .buffers
        .lock()
        .err()
        .expect("poisoned bookkeeping")
        .into_inner();
    assert_eq!(
        records.get(&later.buffer_id).unwrap().phase,
        Phase::Failed {
            stage: Stage::Residency,
            failure: Failure::MapPoisoned,
            disposition: Disposition::Retained
        }
    );
    drop(records);
}

#[cfg(all(not(target_arch = "wasm32"), panic = "unwind"))]
#[test]
#[ignore = "requires an explicitly provisioned WGPU adapter; run with --ignored"]
fn registration_batches_rollback_failure_cancel_and_unwind_without_freeing_borrowed_views() {
    use super::super::backend_types::{WgpuProvider, WgpuProviderOptions};
    use runmat_accelerate_api::{AccelProvider, HostTensorView};
    let provider =
        WgpuProvider::new(WgpuProviderOptions::default()).expect("required WGPU adapter");
    let borrowed = provider
        .upload(&HostTensorView {
            data: &[1.0],
            shape: &[1, 1],
        })
        .unwrap();
    let baseline = provider.allocations.record_count().unwrap();
    let upload = || {
        provider.upload(&HostTensorView {
            data: &[3.0],
            shape: &[1, 1],
        })
    };
    {
        let mut batch = provider.registration_batch();
        let first = batch.track(upload()).unwrap();
        let second = batch.track(upload()).unwrap();
        let published = batch
            .finish(Ok(first), |batch, handle| batch.publish(handle))
            .unwrap();
        assert!(provider.get_entry_raw(&published).is_ok());
        assert!(
            provider.get_entry_raw(&second).is_err(),
            "unpublished intermediate retired on success"
        );
        provider.free(&published).unwrap();
    }
    {
        let mut batch = provider.registration_batch();
        batch.track(upload()).unwrap();
        let result: Result<GpuTensorHandle> = Err(anyhow!("later registration failed"));
        assert!(batch
            .finish(result, |batch, handle| batch.publish(handle))
            .is_err());
        assert_eq!(provider.allocations.record_count().unwrap(), baseline);
    }
    {
        use std::future::Future;
        let mut future = Box::pin(async {
            let mut batch = provider.registration_batch();
            batch.track(upload()).unwrap();
            std::future::pending::<()>().await;
        });
        let mut context = std::task::Context::from_waker(futures::task::noop_waker_ref());
        assert!(future.as_mut().poll(&mut context).is_pending());
        drop(future);
        assert_eq!(provider.allocations.record_count().unwrap(), baseline);
    }
    {
        let mut batch = provider.registration_batch();
        batch.track(upload()).unwrap();
        let future = async move {
            std::future::pending::<()>().await;
            drop(batch);
        };
        drop(future);
        assert_eq!(
            provider.allocations.record_count().unwrap(),
            baseline,
            "an unpolled operation future already owns its registration rollback"
        );
    }
    assert!(catch_allocation_cleanup_unwind(|| {
        let mut batch = provider.registration_batch();
        batch.track(upload()).unwrap();
        panic!("operation unwind");
    })
    .is_err());
    assert_eq!(provider.allocations.record_count().unwrap(), baseline);
    assert!(provider.get_entry_raw(&borrowed).is_ok());
    {
        let mut batch = provider.registration_batch();
        batch.track(upload()).unwrap();
        provider
            .allocations
            .next_id
            .store(u64::MAX, Ordering::Relaxed);
        let failed = batch.track(upload());
        assert_eq!(
            failed
                .as_ref()
                .err()
                .unwrap()
                .downcast_ref::<AllocationStateError>(),
            Some(&AllocationStateError::IdentifierExhausted)
        );
        assert!(batch
            .finish(failed, |batch, handle| batch.publish(handle))
            .is_err());
        assert_eq!(provider.allocations.record_count().unwrap(), baseline);
        assert!(provider.get_entry_raw(&borrowed).is_ok());
    }
    provider.free(&borrowed).unwrap();
}
