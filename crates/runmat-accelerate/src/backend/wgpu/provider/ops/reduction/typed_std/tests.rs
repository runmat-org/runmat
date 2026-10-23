use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelProvider, GpuTensorStorage, HostTensorView,
    NumericElementType, ProviderExecutionCompletionError,
};
use std::error::Error as _;

async fn owner() -> &'static WgpuProvider {
    #[cfg(not(target_arch = "wasm32"))]
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter");
    #[cfg(target_arch = "wasm32")]
    let provider = WgpuProvider::new_async(Default::default())
        .await
        .expect("actual browser WebGPU adapter");
    let provider = Box::leak(Box::new(provider));
    // SAFETY: this actual provider is retained for the process or worker.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    provider
}

fn near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} != {expected}"
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn registered_std_intermediate_completion_keeps_primary_and_cleanup_owners() {
    let owner = TestAllocationOwner::new();
    let borrowed = owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let temporary = owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let probe = temporary.probe.clone();
    probe.fail_retirement("registered std temporary retirement failed");
    let mut ledger = StdCleanupOwner::new(borrowed.handle.allocation_identity());
    ledger.track(&temporary.handle);
    drop(temporary.handle);
    assert_eq!(
        ledger.live_count(),
        1,
        "early failure must drain still-live work"
    );
    let primary = ProviderExecutionError::from(anyhow::anyhow!("std arithmetic primary failure"));
    let error = ledger.complete(Err(primary), true).unwrap_err();
    assert!(error.is_after_effect());
    assert!(!error.is_unsupported());
    let completion = error
        .source()
        .and_then(|source| source.downcast_ref::<ProviderExecutionCompletionError>())
        .expect("typed std completion carrier");
    assert_eq!(
        completion.primary().unwrap().to_string(),
        "std arithmetic primary failure"
    );
    assert_eq!(completion.cleanup_errors().len(), 1);
    assert!(completion.cleanup_errors()[0].physical.is_err());
    assert_eq!(probe.retirements(), 1);
    borrowed.handle.allocation_lease().validate_live().unwrap();

    let temporary = owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let probe = temporary.probe.clone();
    let mut ledger = StdCleanupOwner::new(borrowed.handle.allocation_identity());
    ledger.track(&temporary.handle);
    let temporary_id = temporary.handle.allocation_identity();
    drop(temporary.handle);
    ledger.retire(temporary_id);
    let unsupported = ProviderExecutionError::from(
        runmat_accelerate_api::unsupported_provider_operation("late std primitive unsupported"),
    );
    let error = ledger.complete(Err(unsupported), true).unwrap_err();
    assert!(error.is_after_effect());
    assert!(
        !error.is_unsupported(),
        "registered std work forbids replay"
    );
    assert_eq!(probe.retirements(), 1);
    borrowed.handle.allocation_lease().validate_live().unwrap();

    let temporary = owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let output = owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let temporary_probe = temporary.probe.clone();
    let output_probe = output.probe.clone();
    temporary_probe.fail_retirement("std temporary completion failed");
    output_probe.fail_retirement("std unpublished result completion failed");
    let mut ledger = StdCleanupOwner::new(borrowed.handle.allocation_identity());
    ledger.track(&temporary.handle);
    let temporary_id = temporary.handle.allocation_identity();
    drop(temporary.handle);
    ledger.retire(temporary_id);
    assert_eq!(ledger.live_count(), 0);
    assert_eq!(temporary_probe.retirements(), 1);
    let error = ledger.complete(Ok(output.handle), true).unwrap_err();
    let completion = error
        .source()
        .and_then(|source| source.downcast_ref::<ProviderExecutionCompletionError>())
        .expect("typed std completion with unpublished output");
    assert!(completion.primary().is_none());
    assert_eq!(completion.cleanup_errors().len(), 2);
    assert_eq!(temporary_probe.retirements(), 1);
    assert_eq!(output_probe.retirements(), 1);
    borrowed.handle.allocation_lease().validate_live().unwrap();

    let mut ledger = StdCleanupOwner::new(borrowed.handle.allocation_identity());
    let borrowed_view = borrowed.handle.clone();
    ledger.track(&borrowed_view);
    assert_eq!(
        ledger.live_count(),
        0,
        "a borrowed alias is not an owned temporary"
    );
    drop(borrowed_view);
    let unsupported = ProviderExecutionError::from(
        runmat_accelerate_api::unsupported_provider_operation("singleton pre-binary failure"),
    );
    let error = ledger.complete(Err(unsupported), false).unwrap_err();
    assert!(!error.is_after_effect());
    assert!(!error.is_unsupported());
    borrowed.handle.allocation_lease().validate_live().unwrap();
}

async fn actual_two_std_hooks_preserve_numerics_and_source() {
    let provider = owner().await;
    let input = provider
        .upload(&HostTensorView {
            data: &[1.0, 3.0, 2.0, 4.0],
            shape: &[2, 2],
        })
        .unwrap();
    let before = provider.validated_fusion_cache.lock().unwrap().len();
    let global = provider
        .reduce_std(
            &input,
            ProviderStdNormalization::Population,
            ProviderNanMode::Include,
        )
        .await
        .unwrap();
    near(
        provider.download(&global).await.unwrap().data[0],
        1.118033988749895,
        1e-5,
    );
    assert_eq!(global.shape(), [1, 1]);
    assert!(!global.shares_allocation_with(&input));
    assert_eq!(
        global.physical().element_type(),
        input.physical().element_type()
    );
    let cold = provider.validated_fusion_cache.lock().unwrap().len();
    assert!(
        cold >= before + 5,
        "std must prepare fill, sum, binary, abs, sqrt"
    );
    let warm_hits = provider.telemetry_snapshot().fusion_cache_hits;
    let again = provider
        .reduce_std(
            &input,
            ProviderStdNormalization::Sample,
            ProviderNanMode::Include,
        )
        .await
        .unwrap();
    near(
        provider.download(&again).await.unwrap().data[0],
        (5.0f64 / 3.0).sqrt(),
        1e-5,
    );
    assert!(provider.telemetry_snapshot().fusion_cache_hits > warm_hits);
    let dim = provider
        .reduce_std_dim(
            &input,
            0,
            ProviderStdNormalization::Sample,
            ProviderNanMode::Include,
        )
        .await
        .unwrap();
    assert_eq!(dim.shape(), [1, 2]);
    assert!(!dim.shares_allocation_with(&input));
    assert_eq!(
        dim.physical().element_type(),
        input.physical().element_type()
    );
    let data = provider.download(&dim).await.unwrap().data;
    for value in data {
        near(value, 2.0f64.sqrt(), 1e-5);
    }
    assert_eq!(
        provider.download(&input).await.unwrap().data,
        vec![1.0, 3.0, 2.0, 4.0]
    );
    input.allocation_lease().validate_live().unwrap();
    let rank_three = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            shape: &[2, 2, 2],
        })
        .unwrap();
    let nd = provider
        .reduce_std(
            &rank_three,
            ProviderStdNormalization::Sample,
            ProviderNanMode::Include,
        )
        .await
        .unwrap();
    assert_eq!(nd.shape(), [1, 1, 1]);
    assert!(!nd.shares_allocation_with(&rank_three));
    near(
        provider.download(&nd).await.unwrap().data[0],
        6.0f64.sqrt(),
        1e-5,
    );
    let empty = provider
        .upload(&HostTensorView {
            data: &[],
            shape: &[0, 3],
        })
        .unwrap();
    let empty_global = provider
        .reduce_std(
            &empty,
            ProviderStdNormalization::Sample,
            ProviderNanMode::Include,
        )
        .await
        .unwrap();
    assert!(!empty_global.shares_allocation_with(&empty));
    assert_eq!(
        empty_global.physical().element_type(),
        empty.physical().element_type()
    );
    assert!(provider.download(&empty_global).await.unwrap().data[0].is_nan());
    let empty_dim = provider
        .reduce_std_dim(
            &empty,
            0,
            ProviderStdNormalization::Sample,
            ProviderNanMode::Include,
        )
        .await
        .unwrap();
    assert_eq!(empty_dim.shape(), [1, 3]);
    assert!(provider
        .download(&empty_dim)
        .await
        .unwrap()
        .data
        .iter()
        .all(|value| value.is_nan()));
    let before_refusal = provider.allocations.record_count().unwrap();
    let bad = provider
        .reduce_std_dim(
            &input,
            2,
            ProviderStdNormalization::Sample,
            ProviderNanMode::Include,
        )
        .await;
    let bad = match bad {
        Ok(_) => panic!("invalid std axis admitted"),
        Err(error) => error,
    };
    assert!(bad.is_unsupported());
    assert!(!bad.is_after_effect());
    let omit = provider
        .reduce_std(
            &input,
            ProviderStdNormalization::Sample,
            ProviderNanMode::Omit,
        )
        .await;
    let omit = match omit {
        Ok(_) => panic!("std omitnan admitted"),
        Err(error) => error,
    };
    assert!(omit.is_unsupported());
    assert_eq!(provider.allocations.record_count().unwrap(), before_refusal);
}

async fn actual_std_allocation_faults_retire_unpublished_outputs() {
    let provider = owner().await;
    let input = provider
        .upload(&HostTensorView {
            data: &[1.0, 3.0, 2.0, 4.0],
            shape: &[2, 2],
        })
        .unwrap();
    let empty = provider
        .upload(&HostTensorView {
            data: &[],
            shape: &[0, 3],
        })
        .unwrap();
    let observed_bytes = Cell::new(0u64);
    #[cfg(not(target_arch = "wasm32"))]
    let weak: std::cell::RefCell<Option<std::sync::Weak<wgpu::Buffer>>> =
        std::cell::RefCell::new(None);
    #[cfg(target_arch = "wasm32")]
    let weak: std::cell::RefCell<Option<std::rc::Weak<wgpu::Buffer>>> =
        std::cell::RefCell::new(None);
    let fault = |buffer: &runmat_accelerate_api::WgpuBufferOwner| -> Result<()> {
        observed_bytes.set(buffer.size());
        #[cfg(not(target_arch = "wasm32"))]
        let observed = runmat_accelerate_api::WgpuBufferOwner::downgrade(buffer);
        #[cfg(target_arch = "wasm32")]
        let observed = runmat_accelerate_api::WgpuBufferOwner::downgrade(buffer);
        *weak.borrow_mut() = Some(observed);
        Err(anyhow::Error::new(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "std allocation fault",
        )))
    };
    for (source, dim, expected_bytes) in [
        (&input, None, provider.element_size as u64),
        (&input, Some(0), 2 * provider.element_size as u64),
        (&empty, None, provider.element_size as u64),
    ] {
        *weak.borrow_mut() = None;
        observed_bytes.set(0);
        let records = provider.allocations.record_count().unwrap();
        let failure = provider
            .typed_std_with_fault(
                source,
                dim,
                ProviderStdNormalization::Sample,
                ProviderNanMode::Include,
                Some(&fault),
            )
            .await;
        let failure = match failure {
            Ok(_) => panic!("injected std allocation fault was lost"),
            Err(error) => error,
        };
        assert!(failure.is_after_effect());
        assert!(!failure.is_unsupported());
        assert_eq!(
            failure
                .source()
                .and_then(|source| source.downcast_ref::<std::io::Error>())
                .map(std::io::Error::kind),
            Some(std::io::ErrorKind::BrokenPipe)
        );
        assert_eq!(observed_bytes.get(), expected_bytes);
        assert!(weak
            .borrow()
            .as_ref()
            .and_then(|owner| owner.upgrade())
            .is_none());
        assert_eq!(provider.allocations.record_count().unwrap(), records);
        source.allocation_lease().validate_live().unwrap();
    }
    assert_eq!(
        provider.download(&input).await.unwrap().data,
        vec![1.0, 3.0, 2.0, 4.0]
    );
}

async fn actual_singleton_alias_pre_binary_failure_has_no_physical_effect() {
    assert!(
        std::env::var("RUNMAT_PROVIDER_REDUCTION_SNAPSHOT").is_err(),
        "singleton no-effect witness requires reduction snapshot mode disabled"
    );
    let provider = owner().await;
    let input = provider
        .upload(&HostTensorView {
            data: &[7.0],
            shape: &[1, 1],
        })
        .unwrap();
    let records = provider.allocations.record_count().unwrap();
    let retained = provider.allocations.retained_bytes();
    let seen = Cell::new(false);
    let before_binary = |effect: bool, sum: &GpuTensorHandle| -> Result<()> {
        seen.set(true);
        assert!(!effect, "singleton global sum must remain a borrowed alias");
        assert!(sum.shares_allocation_with(&input));
        assert_eq!(provider.allocations.record_count().unwrap(), records);
        assert_eq!(provider.allocations.retained_bytes(), retained);
        Err(anyhow::Error::new(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "pre-binary allocation failure",
        )))
    };
    let failure = provider
        .typed_std_with_phase_fault(
            &input,
            None,
            ProviderStdNormalization::Sample,
            ProviderNanMode::Include,
            execute::StdFaultSeams {
                after_allocation: None,
                before_binary: Some(&before_binary),
                after_square_retirement: None,
            },
        )
        .await
        .expect_err("injected singleton failure");
    assert!(seen.get());
    assert!(!failure.is_after_effect());
    assert!(!failure.is_unsupported());
    assert_eq!(
        failure
            .source()
            .and_then(|source| source.downcast_ref::<std::io::Error>())
            .map(std::io::Error::kind),
        Some(std::io::ErrorKind::BrokenPipe)
    );
    assert_eq!(provider.allocations.record_count().unwrap(), records);
    assert_eq!(provider.allocations.retained_bytes(), retained);
    input.allocation_lease().validate_live().unwrap();
    assert_eq!(provider.download(&input).await.unwrap().data, vec![7.0]);
}

async fn actual_full_size_square_retires_before_scalar_formula() {
    let provider = owner().await;
    let values = (1..=64).map(f64::from).collect::<Vec<_>>();
    let input = provider
        .upload(&HostTensorView {
            data: &values,
            shape: &[8, 8],
        })
        .unwrap();
    let retained_input = provider.allocations.retained_bytes();
    let seen = Cell::new(false);
    let at_boundary = |provider: &WgpuProvider,
                       square: &runmat_accelerate_api::GpuTensorWireDescriptor,
                       (live_count, live_bytes): (usize, u64)|
     -> Result<()> {
        seen.set(true);
        assert!(
            provider.acquire_live(square).is_err(),
            "squared owner still live"
        );
        assert!(live_count <= 2, "only reduced scalar sums may remain owned");
        assert!(
            live_bytes <= 2 * provider.element_size as u64,
            "full-size square remains retained into scalar formula"
        );
        assert!(
            provider.allocations.retained_bytes()
                <= retained_input + 2 * provider.element_size as u64,
            "registered peak lifetime exceeds input plus two scalar sums"
        );
        Ok(())
    };
    let output = provider
        .typed_std_with_phase_fault(
            &input,
            None,
            ProviderStdNormalization::Population,
            ProviderNanMode::Include,
            execute::StdFaultSeams {
                after_allocation: None,
                before_binary: None,
                after_square_retirement: Some(&at_boundary),
            },
        )
        .await
        .unwrap();
    assert!(seen.get());
    near(
        provider.download(&output).await.unwrap().data[0],
        ((64.0f64 * 64.0 - 1.0) / 12.0).sqrt(),
        1e-4,
    );
    input.allocation_lease().validate_live().unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual WGPU adapter; no adapter skip"]
async fn native_two_std_hooks_preserve_numerics_and_source() {
    actual_two_std_hooks_preserve_numerics_and_source().await;
    actual_std_allocation_faults_retire_unpublished_outputs().await;
    actual_singleton_alias_pre_binary_failure_has_no_physical_effect().await;
    actual_full_size_square_retires_before_scalar_formula().await;
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual SHADER_F64 adapter and RUNMAT_WGPU_FORCE_PRECISION=f64"]
async fn native_f64_std_requires_real_hardware_capability() {
    let provider = owner().await;
    assert_eq!(
        provider.precision,
        crate::backend::wgpu::types::NumericPrecision::F64
    );
    actual_two_std_hooks_preserve_numerics_and_source().await;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_two_std_hooks_preserve_numerics_and_source() {
    actual_two_std_hooks_preserve_numerics_and_source().await;
    actual_std_allocation_faults_retire_unpublished_outputs().await;
    actual_singleton_alias_pre_binary_failure_has_no_physical_effect().await;
    actual_full_size_square_retires_before_scalar_formula().await;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_cold_std_cancellation_preserves_borrowed_source() {
    use runmat_accelerate_api::observation::{
        with_provider_observation, ProviderObservationStatus, ProviderOperationReceipt,
    };
    use std::future::Future;
    let provider = owner().await;
    let input = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[1, 2],
        })
        .unwrap();
    let mut future = Box::pin(provider.reduce_std(
        &input,
        ProviderStdNormalization::Sample,
        ProviderNanMode::Include,
    ));
    let records = provider.allocations.record_count().unwrap();
    let retained_bytes = provider.allocations.retained_bytes();
    let cache_entries = provider.validated_fusion_cache.lock().unwrap().len();
    let telemetry = provider.telemetry_snapshot();
    let mut receipt = ProviderOperationReceipt::new();
    let pending = with_provider_observation(Some(receipt.recorder()), || {
        let waker = futures::task::noop_waker();
        let mut context = std::task::Context::from_waker(&waker);
        future.as_mut().poll(&mut context).is_pending()
    });
    assert!(
        pending,
        "cold std validation should suspend before numerical effects"
    );
    receipt.publish().unwrap();
    assert!(matches!(
        receipt.take_completed().unwrap().status,
        ProviderObservationStatus::Completed { submissions: 0, .. }
    ));
    assert_eq!(provider.allocations.record_count().unwrap(), records);
    assert_eq!(provider.allocations.retained_bytes(), retained_bytes);
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        cache_entries
    );
    assert_eq!(provider.telemetry_snapshot(), telemetry);
    drop(future);
    input.allocation_lease().validate_live().unwrap();
    assert_eq!(
        provider.download(&input).await.unwrap().data,
        vec![1.0, 2.0]
    );
}
