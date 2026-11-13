use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{AccelProvider, HostTensorView, ThreadProviderGuard};
use runmat_execution::ExecutionCandidateKind;
use runmat_value::Value;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::builtins::common::test_support;
use crate::builtins::math::signal::common::convolution::placement::tests::PlacementWitnessService;
use crate::context::RuntimePlacementService;

use super::conv_builtin;

struct FailingConvProvider {
    inner: InProcessProvider,
    hooks: AtomicUsize,
    downloads: AtomicUsize,
}

impl AccelProvider for FailingConvProvider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "conv-selected-failure".into()
    }
    fn upload(
        &self,
        host: &HostTensorView,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn download<'a>(
        &'a self,
        handle: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(handle)
    }
    fn query_feasibility(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::ProviderFeasibility {
        self.inner.query_feasibility(query)
    }
    fn capability_snapshot(&self) -> runmat_accelerate_api::ProviderCapabilitySnapshot {
        self.inner.capability_snapshot()
    }
    fn operation_observation_capability(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::observation::ProviderObservationCapability {
        self.inner.operation_observation_capability(query)
    }
    fn begin_operation_observation(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::observation::ProviderObservationSupport {
        self.inner.begin_operation_observation(query)
    }
    fn conv1d(
        &self,
        _: &runmat_accelerate_api::GpuTensorHandle,
        _: &runmat_accelerate_api::GpuTensorHandle,
        _: runmat_accelerate_api::ProviderConv1dOptions,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        self.hooks.fetch_add(1, Ordering::SeqCst);
        Err(anyhow::anyhow!(
            "selected convolution failed after dispatch"
        ))
    }
}

#[test]
fn selected_provider_failure_is_observed_once_without_host_replay() {
    let _serial = test_support::accel_test_lock();
    let provider: &'static FailingConvProvider = Box::leak(Box::new(FailingConvProvider {
        inner: InProcessProvider::new(),
        hooks: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let signal = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0],
            shape: &[1, 3],
        })
        .unwrap();
    let kernel = provider
        .upload(&HostTensorView {
            data: &[1.0, 1.0],
            shape: &[1, 2],
        })
        .unwrap();
    let service = PlacementWitnessService::new(true);
    let error = block_on(conv_builtin(
        Some(service.context()),
        Value::GpuTensor(signal.clone()),
        Value::GpuTensor(kernel.clone()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("selected operational failure must terminate");
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(service.selections(), 1);
    assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    let feedback = service.feedback();
    assert_eq!(feedback.len(), 1);
    assert!(!feedback[0].succeeded);
    assert_eq!(feedback[0].kind, ExecutionCandidateKind::ProviderOperation);
    assert!(signal.allocation_lease().validate_live().is_ok());
    assert!(kernel.allocation_lease().validate_live().is_ok());
}

#[test]
fn actual_inprocess_vector_host_and_provider_routes_report_completed_work() {
    let _serial = test_support::accel_test_lock();
    let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    let _selected = ThreadProviderGuard::set(Some(provider));
    for prefer_provider in [false, true] {
        let service = PlacementWitnessService::new(prefer_provider);
        let signal = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0],
                shape: &[1, 3],
            })
            .unwrap();
        let kernel = provider
            .upload(&HostTensorView {
                data: &[1.0, 1.0],
                shape: &[1, 2],
            })
            .unwrap();
        let result = block_on(conv_builtin(
            Some(service.context()),
            Value::GpuTensor(signal.clone()),
            Value::GpuTensor(kernel.clone()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("selected convolution route");
        let Value::GpuTensor(_) = &result else {
            panic!("both complete routes must publish a resident output")
        };
        let gathered = test_support::gather(result).unwrap();
        assert_eq!(gathered.materialize_f64(), vec![1.0, 3.0, 5.0, 3.0]);
        assert_eq!(service.selections(), 1);
        assert_eq!(service.pending(), 0);
        let feedback = service.feedback();
        assert_eq!(feedback.len(), 1, "only completed work trains placement");
        assert!(feedback[0].succeeded);
        assert_eq!(
            feedback[0].kind,
            if prefer_provider {
                ExecutionCandidateKind::ProviderOperation
            } else {
                ExecutionCandidateKind::SharedRuntime
            }
        );
        assert!(signal.allocation_lease().validate_live().is_ok());
        assert!(kernel.allocation_lease().validate_live().is_ok());
    }
}

#[test]
fn unpolled_vector_call_does_not_select_or_publish_feedback() {
    let _serial = test_support::accel_test_lock();
    let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let signal = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[1, 2],
        })
        .unwrap();
    let kernel = provider
        .upload(&HostTensorView {
            data: &[1.0],
            shape: &[1, 1],
        })
        .unwrap();
    let service = PlacementWitnessService::new(false);
    let call = conv_builtin(
        Some(service.context()),
        Value::GpuTensor(signal.clone()),
        Value::GpuTensor(kernel.clone()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    );
    drop(call);
    assert_eq!(service.selections(), 0);
    assert!(service.feedback().is_empty());
    assert_eq!(service.pending(), 0);
    assert!(signal.allocation_lease().validate_live().is_ok());
    assert!(kernel.allocation_lease().validate_live().is_ok());
}

#[test]
fn cancellation_after_actual_selection_cannot_train_success_or_failure() {
    let _serial = test_support::accel_test_lock();
    let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let signal = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0],
            shape: &[1, 3],
        })
        .unwrap();
    let kernel = provider
        .upload(&HostTensorView {
            data: &[1.0, 1.0],
            shape: &[1, 2],
        })
        .unwrap();
    let service = PlacementWitnessService::with_cancel_on_select(true, true);
    let _outcome = block_on(conv_builtin(
        Some(service.context()),
        Value::GpuTensor(signal.clone()),
        Value::GpuTensor(kernel.clone()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ));
    service.drain_feedback();
    assert_eq!(
        service.selections(),
        1,
        "cancellation follows a real selected invocation"
    );
    assert!(
        service.feedback().is_empty(),
        "cancellation is not a failed sample"
    );
    assert_eq!(service.pending(), 0);
    assert!(signal.allocation_lease().validate_live().is_ok());
    assert!(kernel.allocation_lease().validate_live().is_ok());
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires a live WGPU adapter; run explicitly for hardware qualification"]
fn actual_wgpu_host_upload_feedback_waits_for_fence_completion() {
    use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};

    let _serial = test_support::accel_test_lock();
    register_wgpu_provider(WgpuProviderOptions::default()).expect("live WGPU adapter required");
    let provider = runmat_accelerate_api::provider().expect("WGPU provider registered");
    let _selected = ThreadProviderGuard::set(Some(provider));
    let signal = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0],
            shape: &[1, 3],
        })
        .unwrap();
    let kernel = provider
        .upload(&HostTensorView {
            data: &[1.0, 1.0],
            shape: &[1, 2],
        })
        .unwrap();
    let service = PlacementWitnessService::new(false);
    let result = block_on(conv_builtin(
        Some(service.context()),
        Value::GpuTensor(signal),
        Value::GpuTensor(kernel),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("selected WGPU host/upload route");
    assert_eq!(service.selections(), 1);
    let gathered = test_support::gather(result).expect("read back completed upload");
    assert_eq!(gathered.materialize_f64(), vec![1.0, 3.0, 5.0, 3.0]);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while service.feedback().is_empty() && std::time::Instant::now() < deadline {
        service.drain_feedback();
        std::thread::yield_now();
    }
    assert_eq!(
        service.feedback().len(),
        1,
        "fence completion must deliver feedback"
    );
    assert!(service.feedback()[0].succeeded);
    assert_eq!(service.pending(), 0);
}
