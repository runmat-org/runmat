//! Post-sampling restoration never authorizes replay of the consumed stream.
use super::*;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, HostNumericTensorView, HostTensorView, ProviderExecutionError,
    ProviderExecutionResult,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct RestoreFailureProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    fail_restore: AtomicBool,
    restore_attempts: AtomicUsize,
}

impl AccelProvider for RestoreFailureProvider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "post-RNG restore failure fixture".into()
    }
    fn upload(&self, host: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn download<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        self.inner.download_numeric(handle)
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        if self.fail_restore.load(Ordering::Relaxed) {
            self.restore_attempts.fetch_add(1, Ordering::Relaxed);
            return Err(ProviderExecutionError::from_source(
                build_runtime_error("retained provider restore failure")
                    .with_identifier("RunMat:test:RandsampleRestoreFailure")
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Requested)
                    .build(),
            )
            .with_unproven_effects());
        }
        self.inner.upload_numeric_raw(host)
    }
}

#[test]
fn resident_restore_failure_retains_cause_and_consumes_rng_exactly_once() {
    let _rng_guard = random::test_guard();
    let _provider_guard = crate::builtins::common::test_support::accel_test_lock();
    let provider: &'static RestoreFailureProvider = Box::leak(Box::new(RestoreFailureProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        fail_restore: AtomicBool::new(false),
        restore_attempts: AtomicUsize::new(0),
    }));
    // SAFETY: the leaked fixture has a fresh device identity and process lifetime.
    unsafe {
        runmat_accelerate_api::register_device_provider(provider);
    }
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let host = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![1, 4]).unwrap();
    let source = crate::builtins::common::gpu_helpers::upload_tensor(provider, &host).unwrap();
    random::reset_rng();
    let before = random::snapshot().unwrap();
    futures::executor::block_on(randsample::randsample_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(host),
            Value::Num(3.0),
        ]),
    ))
    .unwrap();
    let expected_after_one_sample = random::snapshot().unwrap();
    assert_ne!(before.state, expected_after_one_sample.state);

    random::reset_rng();
    provider.fail_restore.store(true, Ordering::Relaxed);
    let error = crate::call_builtin(
        "randsample",
        &[Value::GpuTensor(source.clone()), Value::Num(3.0)],
    )
    .unwrap_err();
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(provider.restore_attempts.load(Ordering::Relaxed), 1);
    assert_eq!(
        random::snapshot().unwrap().state,
        expected_after_one_sample.state
    );
    let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
    let mut retained_requested_cause = false;
    while let Some(current) = cause {
        if let Some(runtime) = current.downcast_ref::<RuntimeError>() {
            retained_requested_cause |= runtime.identifier()
                == Some("RunMat:test:RandsampleRestoreFailure")
                && runtime.gpu_gather_retry == crate::GpuGatherRetry::Requested;
        }
        cause = current.source();
    }
    assert!(
        retained_requested_cause,
        "original typed provider cause must remain owned"
    );
    assert!(
        futures::executor::block_on(provider.download(&source)).is_ok(),
        "borrowed source must remain live"
    );

    // Both policy changes publish automatic results on the host without a
    // restore attempt or a second draw. Explicit residency remains strict and
    // must be rejected before the stream is consumed.
    use crate::context::{RuntimeAccelerationPolicy, RuntimeContext, RuntimeProviderPreference};
    for (enabled, preference) in [
        (false, RuntimeProviderPreference::Auto),
        (true, RuntimeProviderPreference::InProcess),
    ] {
        let runtime = RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
        .with_acceleration_policy(RuntimeAccelerationPolicy {
            enabled,
            provider: preference,
            automatic_offload: false,
            allow_host_fallback: true,
            calibrate: false,
        });
        let _scope = runtime.enter();
        random::reset_rng();
        let output = futures::executor::block_on(randsample::randsample_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(source.clone()),
                Value::Num(3.0),
            ]),
        ))
        .unwrap();
        assert!(!matches!(output, Value::GpuTensor(_)));
        assert_eq!(provider.restore_attempts.load(Ordering::Relaxed), 1);
        assert_eq!(
            random::snapshot().unwrap().state,
            expected_after_one_sample.state
        );

        random::reset_rng();
        let before_explicit = random::snapshot().unwrap();
        let explicit = source
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .unwrap();
        let error = futures::executor::block_on(randsample::randsample_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(explicit),
                Value::Num(3.0),
            ]),
        ))
        .unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(random::snapshot().unwrap().state, before_explicit.state);
        assert_eq!(provider.restore_attempts.load(Ordering::Relaxed), 1);
    }
}
