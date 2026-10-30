//! An actual-adapter, internal probe-scope witness. This is not a public POWER
//! operation receipt or a GPU memory/timing measurement.
use super::*;
#[cfg(target_arch = "wasm32")]
use runmat_accelerate_api::observation::ProviderObservationFailure;
use runmat_accelerate_api::observation::{
    observe_provider_future, ProviderObservationStatus, ProviderObservationSupport,
};
use runmat_accelerate_api::{
    ProviderElementType, ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily,
    ProviderOperationIdentity, ProviderRepresentation, ProviderResidency, ProviderStorage,
};

fn begin_flag_probe_observation(
    provider: &WgpuProvider,
) -> runmat_accelerate_api::observation::ProviderOperationReceipt {
    // The existing fusion-elementwise capability is the flag-kernel observation
    // contract. It is not a fabricated RealPower domain or public POWER query.
    assert_eq!(provider.precision, NumericPrecision::F32);
    let flag = ProviderRepresentation {
        element_type: ProviderElementType::F32,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        residency: ProviderResidency::Device,
        shape: vec![2, 2],
    };
    let exponent = ProviderRepresentation {
        shape: vec![1, 1],
        ..flag.clone()
    };
    let query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new("fusion.elementwise"),
        family: ProviderOperationFamily::Fusion,
        inputs: vec![flag.clone(), exponent],
        outputs: vec![flag],
        workload: Default::default(),
    };
    match provider.begin_operation_observation(&query) {
        ProviderObservationSupport::Observed(receipt) => receipt,
        ProviderObservationSupport::Unsupported => panic!("actual flag observation is required"),
    }
}

async fn actual_probe_scope(provider: &'static WgpuProvider) {
    let (base, exponent) = inputs(provider, true);
    let mut operation = begin_flag_probe_observation(provider);
    let (domain, record) = observe_provider_future(
        Some(operation.recorder()),
        provider.classify_real_power_observed_exec(&base, &exponent),
    )
    .await
    .expect("actual classifier emission and readback");
    assert_eq!(domain, PowerResultDomain::Complex);
    assert!(matches!(
        record.expect("physical submissions").completion_state(),
        ProbeCompletionState::Pending | ProbeCompletionState::Completed
    ));
    let snapshot = operation
        .resource_snapshot()
        .expect("actual resource scope");
    assert_eq!(snapshot.scopes.len(), 1);
    let scope = &snapshot.scopes[0];
    assert!(scope.finished);
    assert_eq!(scope.phase_limits.len(), 4);
    assert_eq!(scope.phase_peaks.len(), 4);
    assert!(scope.issued > 0);
    assert!(snapshot.requested_peak_bytes >= 16, "four F32 flag words");
    for (peak, limit) in scope.phase_peaks.iter().zip(&scope.phase_limits) {
        assert!(peak.is_some_and(|peak| peak <= *limit));
    }
    assert!(snapshot.requested_peak_bytes >= scope.requested_peak_bytes);
    base.allocation_lease().validate_live().unwrap();
    exponent.allocation_lease().validate_live().unwrap();

    operation.publish().expect("selected probe published once");
    let evidence = futures::future::poll_fn(|cx| operation.poll_completed(cx))
        .await
        .expect("completion callback evidence");
    assert!(matches!(
        evidence.status,
        ProviderObservationStatus::Completed { submissions, .. } if submissions > 0
    ));
    let resources = evidence.resources.expect("resource evidence retained");
    assert_eq!(resources.scopes.len(), 1);
    assert_eq!(resources.scopes[0].scope_id, scope.scope_id);
    assert_eq!(resources.scopes[0].phase_peaks, scope.phase_peaks);
    assert_eq!(resources.requested_live_bytes, 0);
    assert_eq!(resources.issued, resources.retired);
    // A pooled handoff is an owner transfer, not proof of VRAM destruction.
    assert!(resources.pooled_handoffs <= resources.retired);
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires actual F32 WGPU adapter and observation progress; RUNMAT_WGPU_FORCE_PRECISION=f32"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn actual_flag_probe_has_four_scoped_resource_phases_and_completion() {
    #[cfg(not(target_arch = "wasm32"))]
    let _ = env_logger::builder().is_test(true).try_init();
    actual_probe_scope(owner().await).await;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn actual_browser_pending_flag_probe_cancel_keeps_borrowed_sources() {
    use std::{
        future::Future,
        task::{Context, Poll},
    };
    let provider = owner().await;
    let (base, exponent) = inputs(provider, true);
    let mut operation = begin_flag_probe_observation(provider);
    let mut future = Box::pin(observe_provider_future(
        Some(operation.recorder()),
        provider.classify_real_power_exec(&base, &exponent),
    ));
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
    let active = operation
        .resource_snapshot()
        .expect("effectful scope issued");
    assert_eq!(active.scopes.len(), 1);
    assert!(active.issued > 0);
    assert!(active.scopes[0].phase_peaks.iter().any(Option::is_some));
    operation.fail(ProviderObservationFailure::Cancelled);
    drop(future);
    assert_eq!(
        operation
            .take_completed()
            .expect("terminal cancellation")
            .status,
        ProviderObservationStatus::Failed(ProviderObservationFailure::Cancelled)
    );
    base.allocation_lease().validate_live().unwrap();
    exponent.allocation_lease().validate_live().unwrap();
    // Failure feedback is terminal independently; actual owner/callback drops
    // still drive the separate resource-retirement waiter after feedback take.
    let retired = futures::future::poll_fn(|cx| operation.poll_resource_retirement(cx))
        .await
        .expect("probe issued a scope");
    assert!(retired.issued > 0);
    assert_eq!(retired.retired, retired.issued);
    assert_eq!(retired.requested_live_bytes, 0);
    assert!(!retired.scopes[0].finished);
    assert_eq!(
        operation.status(),
        ProviderObservationStatus::Failed(ProviderObservationFailure::Cancelled)
    );
    base.allocation_lease().validate_live().unwrap();
    exponent.allocation_lease().validate_live().unwrap();
}
