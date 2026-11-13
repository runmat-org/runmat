use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelProvider, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
    NumericElementType,
};

fn assert_values(provider: &WgpuProvider, handle: &GpuTensorHandle, expected: &[f64]) {
    let tensor = block_on(provider.download_numeric_exec(handle)).unwrap();
    let values = match tensor.data {
        HostNumericDataOwned::F32(values) => {
            assert_eq!(provider.precision, NumericPrecision::F32);
            values.into_iter().map(f64::from).collect::<Vec<_>>()
        }
        HostNumericDataOwned::F64(values) => {
            assert_eq!(provider.precision, NumericPrecision::F64);
            values
        }
        other => panic!("expected native real floating result, got {other:?}"),
    };
    assert_eq!(values.len(), expected.len());
    for (actual, expected) in values.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
}

fn exercise_lazy_mean_and_moments(precision: NumericPrecision) {
    let provider = WgpuProvider::new(WgpuProviderOptions::default())
        .expect("qualification requires an actual WGPU adapter");
    assert_eq!(
        provider.precision, precision,
        "set RUNMAT_WGPU_FORCE_PRECISION to the test precision; F64 requires SHADER_F64"
    );
    let single = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let double = single.map(f64::from);
    let data = match precision {
        NumericPrecision::F32 => HostNumericDataView::F32(&single),
        NumericPrecision::F64 => HostNumericDataView::F64(&double),
    };
    let original = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data,
            shape: &[2, 3],
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
        })
        .unwrap();
    let lazy = provider.transpose_exec(&original).unwrap();
    assert!(runmat_accelerate_api::handle_is_transposed(&lazy));
    assert!(lazy.shares_allocation_with(&original));
    let original_view = original.view().clone();
    let lazy_view = lazy.view().clone();
    let identity = lazy.view_identity();
    let class = match precision {
        NumericPrecision::F32 => NumericElementType::F32,
        NumericPrecision::F64 => NumericElementType::F64,
    };
    let native_bytes = 6 * class.element_size() as u64;
    // Entry lookup and the returned tensor each own a logical shape vector.
    let shape_bytes = 2 * original.shape().len() as u64 * std::mem::size_of::<usize>() as u64;
    assert_eq!(
        provider.download_numeric_host_workspace_bound(&original),
        runmat_execution::ResourceByteBound::Known(native_bytes + shape_bytes)
    );
    assert_eq!(
        provider.download_numeric_host_workspace_bound(&lazy),
        runmat_execution::ResourceByteBound::Known(native_bytes * 2 + shape_bytes)
    );

    for (axis, shape, means, squares) in [
        (0, vec![1, 2], vec![3.0, 4.0], vec![35.0 / 3.0, 56.0 / 3.0]),
        (1, vec![3, 1], vec![1.5, 3.5, 5.5], vec![2.5, 12.5, 30.5]),
    ] {
        let before = provider.telemetry_snapshot().download_bytes;
        let mean = block_on(provider.reduce_mean_nd(&lazy, &[axis]))
            .unwrap()
            .expect("admitted mean");
        assert_eq!(provider.telemetry_snapshot().download_bytes, before);
        assert_eq!(mean.shape(), shape);
        assert_eq!(mean.physical().element_type(), class);
        assert!(!mean.shares_allocation_with(&lazy));
        assert_values(&provider, &mean, &means);

        // A second call must reuse the cache under the original view, not a
        // transient materialized view whose lease has already been released.
        let before = provider.telemetry_snapshot().download_bytes;
        let repeated = block_on(provider.reduce_mean_nd(&lazy, &[axis]))
            .unwrap()
            .expect("admitted mean");
        assert_eq!(provider.telemetry_snapshot().download_bytes, before);
        assert!(repeated.shares_allocation_with(&mean));
        assert!(provider
            .reduction_reuse
            .lock()
            .unwrap()
            .moment_pair(&(identity, vec![axis]))
            .is_some());
        let pinned = provider
            .reserve_cached_mean_nd(&lazy, &[axis])
            .unwrap()
            .expect("same exact source view has a pinned warm pair");
        let two_outputs = 2 * shape.iter().product::<usize>() as u64 * class.element_size() as u64;
        assert!(pinned.retained_pair_bytes() >= two_outputs);
        let pinned_mean = pinned.into_mean();
        assert!(pinned_mean.shares_allocation_with(&mean));
        assert!(
            provider
                .reserve_cached_mean_nd(&original, &[axis])
                .unwrap()
                .is_none(),
            "sibling view is not a warm hit"
        );
        drop(pinned_mean);
        assert_values(&provider, &repeated, &means);

        // Direct moments is also used independently of mean's cache. Both
        // returned allocations must reflect logical order, including E[x²].
        let before = provider.telemetry_snapshot().download_bytes;
        let moments = block_on(provider.reduce_moments_nd(&lazy, &[axis])).unwrap();
        assert_eq!(provider.telemetry_snapshot().download_bytes, before);
        assert!(!moments.mean.shares_allocation_with(&moments.ex2));
        for output in [&moments.mean, &moments.ex2] {
            assert_eq!(output.shape(), shape);
            assert_eq!(output.physical().element_type(), class);
            assert!(!output.shares_allocation_with(&original));
            output.allocation_lease().validate_live().unwrap();
        }
        assert_values(&provider, &moments.mean, &means);
        assert_values(&provider, &moments.ex2, &squares);
        assert_eq!(lazy.view_identity(), identity);
        assert_eq!(lazy.view(), &lazy_view);
        assert_eq!(original.view(), &original_view);
        assert_values(&provider, &original, &double);
        assert_values(&provider, &lazy, &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }
    drop(lazy);
    assert_values(&provider, &original, &double);
}

#[test]
#[ignore = "requires real WGPU adapter; run with RUNMAT_WGPU_FORCE_PRECISION=f32"]
fn actual_lazy_mean_and_moments_f32() {
    exercise_lazy_mean_and_moments(NumericPrecision::F32);
}

#[test]
#[ignore = "requires SHADER_F64 adapter; run with RUNMAT_WGPU_FORCE_PRECISION=f64"]
fn actual_lazy_mean_and_moments_f64() {
    exercise_lazy_mean_and_moments(NumericPrecision::F64);
}

fn exercise_source_retirement_with_cached_pair(poison_cache: bool) {
    let provider = WgpuProvider::new(WgpuProviderOptions::default())
        .expect("qualification requires an actual WGPU adapter");
    assert_eq!(provider.precision, NumericPrecision::F32);
    let baseline_records = provider.allocations.record_count().unwrap();
    let input = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 2.0, 3.0, 4.0]),
            shape: &[2, 2],
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
        })
        .unwrap();
    let input_weak = input.downgrade();
    let mean = block_on(provider.reduce_mean_nd(&input, &[0]))
        .unwrap()
        .expect("admitted mean");
    let (cached_mean, cached_ex2) = provider
        .reduction_reuse
        .lock()
        .unwrap()
        .moment_pair(&(input.view_identity(), vec![0]))
        .expect("native moments pair retained for exact source view");
    let mean_weak = cached_mean.downgrade();
    let ex2_weak = cached_ex2.downgrade();
    drop(mean);
    drop(cached_mean);
    drop(cached_ex2);
    if poison_cache {
        let cache = provider.reduction_reuse.clone();
        assert!(std::thread::spawn(move || {
            let _guard = cache.lock().unwrap();
            panic!("injected optional cache bookkeeping poison");
        })
        .join()
        .is_err());
        assert!(provider.reduction_reuse.is_poisoned());
    }
    drop(input);
    assert!(input_weak.acquire().is_err());
    assert!(mean_weak.acquire().is_err());
    assert!(ex2_weak.acquire().is_err());
    assert_eq!(
        provider.allocations.record_count().unwrap(),
        baseline_records
    );
}

#[test]
#[ignore = "requires actual F32 WGPU adapter; verifies source-driven cache output retirement"]
fn actual_source_retirement_releases_cached_moment_output_owners() {
    exercise_source_retirement_with_cached_pair(false);
}

#[test]
#[ignore = "requires actual F32 WGPU adapter; verifies pinned warm pair survives cache retirement"]
fn actual_pinned_warm_pair_survives_source_retirement_until_consumed() {
    let provider = WgpuProvider::new(WgpuProviderOptions::default())
        .expect("qualification requires an actual WGPU adapter");
    assert_eq!(provider.precision, NumericPrecision::F32);
    let input = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 2.0, 3.0, 4.0]),
            shape: &[2, 2],
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
        })
        .unwrap();
    let produced = block_on(provider.reduce_mean_nd(&input, &[0]))
        .unwrap()
        .expect("admitted mean");
    let (cached_mean, cached_ex2) = provider
        .reduction_reuse
        .lock()
        .unwrap()
        .moment_pair(&(input.view_identity(), vec![0]))
        .unwrap();
    let mean_weak = cached_mean.downgrade();
    let ex2_weak = cached_ex2.downgrade();
    let pinned = provider
        .reserve_cached_mean_nd(&input, &[0])
        .unwrap()
        .expect("exact warm pair");
    drop((produced, cached_mean, cached_ex2));
    drop(input);
    assert!(mean_weak.acquire().is_ok());
    assert!(ex2_weak.acquire().is_ok());
    let selected_mean = pinned.into_mean();
    assert!(
        ex2_weak.acquire().is_err(),
        "hidden pair is no longer pinned"
    );
    assert_values(&provider, &selected_mean, &[1.5, 3.5]);
    drop(selected_mean);
    assert!(mean_weak.acquire().is_err());
}

#[test]
#[ignore = "requires actual F32 WGPU adapter; verifies selected cold moments geometry and cache"]
fn actual_selected_cold_mean_executes_one_exact_moments_route() {
    use runmat_runtime::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};
    use runmat_runtime::context::host_effect::ManagedHostEffectLedger;
    use runmat_runtime::context::{
        ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
        RuntimeServicePorts,
    };
    use std::rc::Rc;

    struct ManagedParallel(ManagedHostEffectLedger);
    impl RuntimeParallelService for ManagedParallel {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..RuntimeParallelResources::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }

    let provider = WgpuProvider::new(WgpuProviderOptions::default())
        .expect("qualification requires an actual WGPU adapter");
    assert_eq!(provider.precision, NumericPrecision::F32);
    let input = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 2.0, 3.0, 4.0]),
            shape: &[2, 2],
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
        })
        .unwrap();
    assert!(provider
        .reserve_cached_mean_nd(&input, &[0])
        .unwrap()
        .is_none());
    let selected = provider
        .prepare_cold_mean_nd(&input, &[0])
        .unwrap()
        .expect("canonical cold input has a selected route");
    assert_eq!(selected.output_shape(), [1, 2]);
    assert_eq!(selected.requested_pair_bytes(), Some(16));
    assert!(selected.requested_uniform_bytes() > 0);
    assert_eq!(
        selected.requested_incremental_peak_bytes(),
        Some(16 + selected.requested_uniform_bytes())
    );
    // A concurrent ordinary call may warm the cache after selection. The
    // selected cold executor must not silently switch to that warm output.
    let concurrently_warmed = block_on(provider.reduce_mean_nd(&input, &[0]))
        .unwrap()
        .expect("admitted mean");
    let registration = provider.registered_output_host_metadata(2).unwrap();
    let table = provider.output_registration_table_upper_bytes(2).unwrap();
    let preparation = provider
        .output_registration_preparation_host_bytes()
        .unwrap();
    let view = HostAllocationPlan::retained_view_bytes(registration.view_payload_bytes).unwrap();
    let record =
        HostAllocationPlan::retained_metadata_bytes(registration.record_payload_bytes).unwrap();
    let lease =
        HostAllocationPlan::retained_metadata_bytes(registration.lease_payload_bytes).unwrap();
    let cache_payload = selected.requested_cache_metadata_payload_bytes();
    let cache_key = HostAllocationPlan::retained_metadata_bytes(cache_payload).unwrap();
    let retained = [
        HostAllocationPlan::retained_view_issuer_bytes().unwrap(),
        HostAllocationPlan::retained_metadata_bytes(table).unwrap(),
        cache_key,
        record.checked_add(lease).unwrap().checked_mul(2).unwrap(),
        view.checked_mul(4).unwrap(),
    ]
    .into_iter()
    .try_fold(0_u64, |sum, bytes| sum.checked_add(bytes))
    .unwrap();
    let capacity = retained
        .checked_add(preparation)
        .and_then(|bytes| bytes.checked_add(view))
        .unwrap();
    let ledger = ManagedHostEffectLedger::new_with_epoch(capacity, 11);
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let receipt = HostAllocationReceipt::admit(
        Some(&runtime),
        HostAllocationPlan::new(preparation, retained).unwrap(),
    )
    .unwrap()
    .expect("finite managed output receipt");
    let permit = receipt.retained_view_permit(&runtime).unwrap();
    let selected = selected
        .with_cache_view_grants([
            permit.claim_view(registration.view_payload_bytes).unwrap(),
            permit.claim_view(registration.view_payload_bytes).unwrap(),
        ])
        .with_cache_metadata_claim(permit.claim_metadata(cache_payload).unwrap());
    let admission = provider
        .prepare_output_host_metadata(Box::new(permit.clone()), 2)
        .unwrap();
    let executed = provider
        .execute_cold_mean_nd(selected, Some(&*admission))
        .unwrap();
    drop(admission);
    drop(permit);
    drop(receipt);
    let retained_after_publication = ledger.snapshot().live_claimed_bytes;
    assert_eq!(retained_after_publication, retained - 2 * view);
    assert_eq!(
        ledger.snapshot().live_reserved_bytes,
        retained_after_publication
    );
    assert_eq!(executed.requested_pair_bytes, 16);
    assert!(executed.observed_pair_bytes >= executed.requested_pair_bytes);
    assert_eq!(executed.selected_submissions, 1);
    let mean = executed.mean;
    let ex2 = executed.ex2;
    assert!(!mean.shares_allocation_with(&concurrently_warmed));
    assert_values(&provider, &mean, &[1.5, 3.5]);
    let shared = mean.clone();
    assert_eq!(
        ledger.snapshot().live_claimed_bytes,
        retained_after_publication
    );
    let derived = mean
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .unwrap();
    assert_eq!(
        ledger.snapshot().live_claimed_bytes,
        retained_after_publication + view
    );
    drop(derived);
    drop(shared);
    assert_eq!(
        ledger.snapshot().live_claimed_bytes,
        retained_after_publication
    );
    let warm = provider
        .reserve_cached_mean_nd(&input, &[0])
        .unwrap()
        .expect("cold route retains its hidden moment pair");
    assert_eq!(warm.retained_pair_bytes(), 16);
    assert!(warm.into_mean().shares_allocation_with(&mean));
    assert_eq!(
        ledger.snapshot().live_claimed_bytes,
        retained_after_publication
    );
    let weak_mean = mean.downgrade();
    let sibling_view = mean
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .unwrap();
    drop(ex2);
    drop(mean);
    drop(concurrently_warmed);
    assert_eq!(
        ledger.snapshot().live_claimed_bytes,
        retained_after_publication + view
    );
    drop(input);
    assert!(
        weak_mean.acquire().is_err(),
        "retired source view stays dead"
    );
    assert!(
        ledger.snapshot().live_claimed_bytes > 0,
        "sibling physical view and dead weak view still retain their claims"
    );
    drop(sibling_view);
    assert!(
        ledger.snapshot().live_claimed_bytes > 0,
        "weak view backing outlives the last strong physical view"
    );
    drop(weak_mean);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
#[ignore = "requires actual F32 WGPU adapter; verifies optional cache poison cannot block retirement"]
fn poisoned_cache_does_not_quarantine_actual_source_or_cached_output_owners() {
    exercise_source_retirement_with_cached_pair(true);
}

#[test]
#[ignore = "requires actual F32 WGPU adapter; verifies exact-view lazy pruning"]
fn dropped_source_view_is_pruned_while_physical_sibling_remains_live() {
    let provider = WgpuProvider::new(WgpuProviderOptions::default())
        .expect("qualification requires an actual WGPU adapter");
    assert_eq!(provider.precision, NumericPrecision::F32);
    let original = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            shape: &[2, 3],
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
        })
        .unwrap();
    let view = provider.transpose_exec(&original).unwrap();
    assert_ne!(view.view_identity(), original.view_identity());
    let view_identity = view.view_identity();
    let mean = block_on(provider.reduce_mean_nd(&view, &[0]))
        .unwrap()
        .expect("admitted mean");
    let weak_mean = mean.downgrade();
    drop(mean);
    drop(view);
    assert!(original.allocation_lease().validate_live().is_ok());
    assert!(
        provider
            .reduction_reuse
            .lock()
            .unwrap()
            .moment_pair(&(view_identity, vec![0]))
            .is_some(),
        "bounded cache retains pair until next insertion or eviction"
    );
    let replacement = block_on(provider.reduce_mean_nd(&original, &[0]))
        .unwrap()
        .expect("admitted mean");
    assert!(
        provider
            .reduction_reuse
            .lock()
            .unwrap()
            .moment_pair(&(view_identity, vec![0]))
            .is_none(),
        "dead exact view was lazily pruned"
    );
    assert!(weak_mean.acquire().is_err());
    assert!(original.allocation_lease().validate_live().is_ok());
    drop(replacement);
}
