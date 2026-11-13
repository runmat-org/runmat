//! Registered-owner witnesses for ttest2 admission and gather failure.
use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView, NumericElementType,
    ThreadProviderGuard,
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Default)]
struct RejectDownloads {
    owner: TestAllocationOwner,
    downloads: AtomicUsize,
}

impl AccelProvider for RejectDownloads {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "ttest2 registered-owner admission witness".into()
    }

    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("ttest2 must not upload inputs while admitting controls")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Err(std::io::Error::other("ttest2 fixture rejected download").into()) })
    }

    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Err(std::io::Error::other("ttest2 fixture rejected download").into()) })
    }
}

fn host_sample() -> Value {
    Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap())
}

fn assert_provider_cause(error: &RuntimeError) {
    assert_eq!(error.identifier(), Some("RunMat:ttest2:InvalidArgument"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let gather = error
        .source
        .as_ref()
        .and_then(|source| source.downcast_ref::<RuntimeError>())
        .expect("typed gather error is retained");
    assert_eq!(gather.identifier(), Some("RunMat:gather:DownloadFailed"));
    assert_eq!(error.span, gather.span);
    assert_eq!(error.gpu_gather_retry, gather.gpu_gather_retry);
    let mut source: &(dyn std::error::Error + 'static) = error;
    while !source.is::<std::io::Error>() {
        source = source
            .source()
            .expect("original provider error remains in cause chain");
    }
    assert_eq!(source.to_string(), "ttest2 fixture rejected download");
}

#[test]
fn known_bad_controls_and_output_count_precede_registered_owner_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloads::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let allocation = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let resident = || Value::GpuTensor(allocation.handle.clone());
    assert!(runmat_accelerate_api::same_provider_instance(
        runmat_accelerate_api::provider_for_handle(&allocation.handle).unwrap(),
        provider
    ));

    let error = block_on(ttest2_builtin(
        None,
        resident(),
        host_sample(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Tail"),
            Value::from("sideways"),
        ]),
    ))
    .unwrap_err();
    assert!(error.message().contains("unsupported Tail"), "{error}");
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);

    // An earlier deferred resident control must not hide a later known bad
    // host option; the whole host-visible admission pass precedes downloads.
    let error = block_on(ttest2_builtin(
        None,
        host_sample(),
        host_sample(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Dim"),
            resident(),
            Value::from("Tail"),
            Value::from("sideways"),
        ]),
    ))
    .unwrap_err();
    assert!(error.message().contains("unsupported Tail"), "{error}");
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);

    let _outputs = crate::output_count::push_output_count(Some(5)).expect("output-count scope");
    let error = block_on(ttest2_builtin(
        None,
        resident(),
        host_sample(),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(
        error.message().contains("too many output arguments"),
        "{error}"
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn rejected_sample_and_control_downloads_retain_cause_and_borrowed_owner() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloads::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let allocation = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let resident = || Value::GpuTensor(allocation.handle.clone());

    let error = block_on(ttest2_builtin(
        None,
        resident(),
        host_sample(),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_provider_cause(&error);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    assert_eq!(allocation.probe.retirements(), 0);

    let error = block_on(ttest2_builtin(
        None,
        host_sample(),
        host_sample(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Dim"),
            resident(),
        ]),
    ))
    .unwrap_err();
    assert_provider_cause(&error);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 2);
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn explicit_context_refuses_finite_work_before_download_and_retains_cancellation() {
    use crate::context::{
        host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeContext,
        RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use std::rc::Rc;
    struct Finite(ManagedHostEffectLedger);
    impl RuntimeParallelService for Finite {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..Default::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloads::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let allocation = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let ledger = ManagedHostEffectLedger::new(1024 * 1024);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Finite(ledger.clone()))),
        );
    let error = block_on(ttest2_builtin(
        Some(runtime),
        Value::GpuTensor(allocation.handle.clone()),
        host_sample(),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(ledger.snapshot().peak_reserved_bytes, 0);
    let source = runmat_execution::cancellation::CancellationSource::new();
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_cancellation_token(source.token());
    runtime.set_error_namespace("TtestInvocation");
    source.cancel(runmat_execution::CancellationReason::User);
    let error = block_on(ttest2_builtin(
        Some(runtime),
        Value::GpuTensor(allocation.handle.clone()),
        host_sample(),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("TtestInvocation:Cancelled"));
    assert!(crate::runtime_error::error_has_control_cancellation_origin(
        &error
    ));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}
