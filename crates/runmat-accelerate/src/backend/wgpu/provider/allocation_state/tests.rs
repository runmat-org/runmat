use super::*;

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
    assert_eq!(
        Arc::strong_count(&resources),
        1,
        "no future handle/lease cycle"
    );
    assert_eq!(
        Arc::strong_count(&cache),
        2,
        "same resource cache, not a duplicate"
    );
    drop(resources);
    assert!(state.kernel_resources.upgrade().is_none());
    let retained = state.clone();
    drop(state);
    assert_eq!(retained.record_count().unwrap(), 0);
    drop(retained);
    assert_eq!(Arc::strong_count(&cache), 1);
}

#[test]
fn empty_state_preserves_explicit_free_owner_validation_and_idempotence() {
    let device = runmat_accelerate_api::next_device_id();
    let state = AllocationState::new(
        device,
        1,
        1024,
        Arc::new(BindGroupCache::new()),
        Weak::new(),
    );
    let unknown = GpuTensorHandle::new(vec![1], device, 1);
    let foreign = GpuTensorHandle::new(vec![1], device.wrapping_add(1), 1);
    assert!(state
        .free(&foreign)
        .unwrap_err()
        .to_string()
        .contains("belongs to device"));
    state.free(&unknown).unwrap();
    state.free(&unknown).unwrap();
    assert_eq!(state.pooled_bytes(4), 0);
}

#[cfg(panic = "unwind")]
#[test]
fn poisoned_allocation_map_does_not_report_successful_retirement() {
    let device = runmat_accelerate_api::next_device_id();
    let state = AllocationState::new(
        device,
        1,
        1024,
        Arc::new(BindGroupCache::new()),
        Weak::new(),
    );
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = state.buffers.lock().unwrap();
        panic!("poison allocation map");
    }))
    .is_err());
    let handle = GpuTensorHandle::new(vec![1], device, 1);
    let error = state.free(&handle).unwrap_err();
    assert_eq!(
        error.downcast_ref::<AllocationStateError>(),
        Some(&AllocationStateError::Poisoned)
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires an explicitly provisioned WGPU adapter; run with --ignored"]
fn wgpu_allocation_views_exports_and_pool_keep_existing_physical_lifetimes() {
    use super::super::backend_types::{WgpuProvider, WgpuProviderOptions};
    use runmat_accelerate_api::{AccelProvider, HostTensorView};
    let mut provider =
        WgpuProvider::new(WgpuProviderOptions::default()).expect("required WGPU adapter");
    provider.allocations = Arc::new(AllocationState::new(
        provider.runtime_device_id,
        1,
        u64::MAX,
        provider.bind_group_cache.clone(),
        Arc::downgrade(&provider.kernel_resources),
    ));
    let input = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[1, 2],
        })
        .unwrap();
    let snapshot = provider.get_entry_raw(&input).unwrap();
    let expected_type = snapshot.element_type;
    let alias = provider
        .register_storage_move_alias(&snapshot, vec![2, 1])
        .unwrap();
    assert_ne!(input.buffer_id, alias.buffer_id);
    let alias_snapshot = provider.get_entry_raw(&alias).unwrap();
    assert!(Arc::ptr_eq(&snapshot.buffer, &alias_snapshot.buffer));
    assert_eq!(alias_snapshot.element_type, expected_type);
    assert_eq!(alias_snapshot.storage, snapshot.storage);
    assert_eq!(alias_snapshot.shape, vec![2, 1]);
    drop(alias_snapshot);
    drop(snapshot);
    let export = provider
        .export_wgpu_buffer(&input)
        .expect("real floating export");
    provider.free(&input).unwrap();
    assert_eq!(provider.allocations.pooled_bytes(provider.element_size), 0);
    assert_eq!(
        futures::executor::block_on(provider.download(&alias))
            .unwrap()
            .data,
        vec![1.0, 2.0]
    );
    provider.free(&alias).unwrap();
    assert_eq!(
        provider.allocations.pooled_bytes(provider.element_size),
        0,
        "export prevents repooling"
    );
    assert_eq!(export.len, 2);
    drop(export);

    let fresh = provider
        .upload(&HostTensorView {
            data: &[3.0, 4.0],
            shape: &[1, 2],
        })
        .unwrap();
    let (pointer, bytes) = {
        let entry = provider.allocations.live_entry(fresh.buffer_id).unwrap();
        (Arc::as_ptr(&entry.buffer), entry.allocated_bytes)
    };
    provider.free(&fresh).unwrap();
    assert_eq!(
        provider.allocations.pooled_bytes(provider.element_size),
        bytes
    );
    let (reused, did_reuse) = provider.create_storage_buffer_for_usage(
        BufferUsageClass::Generic,
        2,
        "allocation state reuse",
    );
    assert!(did_reuse);
    assert_eq!(Arc::as_ptr(&reused), pointer);
    assert_eq!(provider.allocations.pooled_bytes(provider.element_size), 0);
    assert_eq!(provider.allocations.record_count().unwrap(), 0);
}

#[cfg(all(not(target_arch = "wasm32"), panic = "unwind"))]
#[test]
#[ignore = "requires an explicitly provisioned WGPU adapter; run with --ignored"]
fn failed_cache_invalidation_preserves_allocation_and_annotations_without_pooling() {
    use super::super::backend_types::{WgpuProvider, WgpuProviderOptions};
    use runmat_accelerate_api::{AccelProvider, HostTensorView};
    let provider =
        WgpuProvider::new(WgpuProviderOptions::default()).expect("required WGPU adapter");
    let input = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[1, 2],
        })
        .unwrap();
    runmat_accelerate_api::record_handle_transpose(&input, 2, 1);
    let transpose = runmat_accelerate_api::handle_transpose_info(&input);
    let original = provider.get_entry_raw(&input).unwrap();
    let pooled_before = provider.allocations.pooled_bytes(provider.element_size);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        provider.bind_group_cache.poison_for_retirement_test();
    }))
    .is_err());
    let error = provider.free(&input).unwrap_err();
    assert_eq!(
        error.downcast_ref::<crate::backend::wgpu::cache::bind_group::CacheInvalidationError>(),
        Some(&crate::backend::wgpu::cache::bind_group::CacheInvalidationError::Poisoned)
    );
    assert!(provider.get_entry_raw(&input).is_err());
    {
        let records = provider.allocations.buffers.lock().unwrap();
        let retained = records.get(&input.buffer_id).unwrap();
        assert_eq!(
            retained.phase,
            Phase::Failed {
                stage: Stage::Cache,
                failure: Failure::Cache,
                disposition: Disposition::Retained
            }
        );
        assert!(Arc::ptr_eq(
            &original.buffer,
            &retained.entry.as_ref().unwrap().buffer
        ));
    }
    assert_eq!(
        runmat_accelerate_api::handle_transpose_info(&input),
        transpose
    );
    assert_eq!(
        provider.allocations.pooled_bytes(provider.element_size),
        pooled_before
    );
    // Final owner teardown disposes the quarantined reference and annotations,
    // even though the poisoned cache cannot report successful invalidation.
    drop(provider);
    assert_eq!(runmat_accelerate_api::handle_transpose_info(&input), None);
}
