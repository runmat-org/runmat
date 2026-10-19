use super::*;

#[test]
fn prepared_host_staged_row_pass_uses_order_first_multistate_geometry() {
    let _lock = test_support::accel_test_lock();
    let provider: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
        Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
    );
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let coeffs = NormalizedCoefficients::new(
        &[
            Complex::new(0.25, 0.0),
            Complex::new(0.5, 0.0),
            Complex::new(0.25, 0.0),
        ],
        &[Complex::new(1.0, 0.0)],
    )
    .unwrap();
    let prepared = ProviderPreparedIirCoefficients::upload(
        provider,
        &[0.25, 0.5, 0.25],
        &[1.0, 0.0, 0.0],
        provider.precision(),
        &[],
    )
    .unwrap();
    let signal = [
        Complex::new(1.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(3.0, 0.0),
        Complex::new(4.0, 0.0),
    ];
    let output = block_on(gpu_filter_pass(
        provider,
        &prepared,
        &coeffs,
        &signal,
        &[1, 4],
        &[Complex::new(0.0, 0.0); 2],
        &[],
    ))
    .expect("prepared row pass must admit order-first state");
    let actual = output.into_iter().map(|value| value.re).collect::<Vec<_>>();
    approx(&actual, &[0.25, 1.0, 2.0, 3.0], 1e-12);
}

fn value_tolerance(value: &Value) -> f64 {
    match value {
        Value::GpuTensor(handle) => match handle_precision(handle)
            .or_else(|| provider_for_handle(handle).map(|provider| provider.precision()))
            .unwrap_or(ProviderPrecision::F64)
        {
            ProviderPrecision::F64 => 1e-8,
            ProviderPrecision::F32 => 1e-6,
        },
        _ => 1e-8,
    }
}

struct OperationalFailureProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    calls: std::sync::atomic::AtomicUsize,
    downloads: std::sync::atomic::AtomicUsize,
}

struct NoIirProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    uploads: std::sync::atomic::AtomicUsize,
}

impl AccelProvider for NoIirProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.uploads
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.inner.upload(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        self.inner.download_numeric(handle)
    }
    fn device_info(&self) -> String {
        "filtfilt-no-iir".into()
    }
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
}

#[test]
fn absent_iir_hook_falls_back_before_coefficient_uploads() {
    let _lock = test_support::accel_test_lock();
    let provider: &'static NoIirProvider = Box::leak(Box::new(NoIirProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        uploads: std::sync::atomic::AtomicUsize::new(0),
    }));
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let x = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            shape: &[1, 8],
        })
        .expect("resident signal");
    let b = Tensor::new(vec![0.5, 0.5], vec![1, 2]).unwrap();
    let result = call(Value::Tensor(b), Value::Num(1.0), Value::GpuTensor(x))
        .expect("host fallback before unsupported provider effects");
    assert_eq!(
        provider.uploads.load(std::sync::atomic::Ordering::SeqCst),
        1
    );
    let (values, shape) = tensor_data(result);
    assert_eq!(shape, vec![1, 8]);
    approx(&values[3..7], &[4.0, 5.0, 6.0, 7.0], 1e-9);
}

#[test]
fn resident_row_filter_uses_axis_aligned_multistate_geometry() {
    let b = Tensor::new(vec![0.25, 0.5, 0.25], vec![1, 3]).unwrap();
    let values = (1..=12).map(f64::from).collect::<Vec<_>>();
    let host = call(
        Value::Tensor(b.clone()),
        Value::Num(1.0),
        Value::Tensor(Tensor::new(values.clone(), vec![1, 12]).unwrap()),
    )
    .expect("host reference");
    let (expected, expected_shape) = tensor_data(host);
    test_support::with_test_provider(|provider| {
        let signal = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &[1, 12],
            })
            .expect("resident signal");
        let result = call(Value::Tensor(b), Value::Num(1.0), Value::GpuTensor(signal))
            .expect("resident row signal with two filter states");
        assert!(matches!(result, Value::GpuTensor(_)));
        let gathered = test_support::gather(result).expect("resident result");
        assert_eq!(gathered.shape(), expected_shape);
        approx(&gathered.materialize_f64(), &expected, 1e-9);
    });
}

impl AccelProvider for OperationalFailureProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.inner.download(handle)
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.inner.download_numeric(handle)
    }
    fn device_info(&self) -> String {
        "filtfilt-operational-failure".into()
    }
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn query_feasibility(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::ProviderFeasibility {
        self.inner.query_feasibility(query)
    }
    fn iir_zero_phase<'a>(
        &'a self,
        _request: &'a runmat_accelerate_api::ProviderIirZeroPhaseRequest<'a>,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Box::pin(async {
            Err(
                runmat_accelerate_api::ProviderExecutionError::from(anyhow::anyhow!(
                    "device failed during zero-phase pass"
                ))
                .after_effect(),
            )
        })
    }
}

#[test]
fn resident_pass_operational_failure_keeps_cause_and_does_not_replay() {
    let _lock = test_support::accel_test_lock();
    let provider: &'static OperationalFailureProvider =
        Box::leak(Box::new(OperationalFailureProvider {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            calls: std::sync::atomic::AtomicUsize::new(0),
            downloads: std::sync::atomic::AtomicUsize::new(0),
        }));
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let x = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            shape: &[1, 8],
        })
        .unwrap();
    let b = Tensor::new(vec![0.5, 0.5], vec![1, 2]).unwrap();
    let error = call(Value::Tensor(b), Value::Num(1.0), Value::GpuTensor(x))
        .expect_err("operational pass failure must terminate");
    assert_eq!(error.identifier(), FILTFILT_ERROR_PASS_FAILED.identifier);
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(std::error::Error::source(&error).is_some());
    assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[test]
fn host_invalid_denominator_precedes_resident_numerator_readback() {
    let _lock = test_support::accel_test_lock();
    let provider: &'static OperationalFailureProvider =
        Box::leak(Box::new(OperationalFailureProvider {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            calls: std::sync::atomic::AtomicUsize::new(0),
            downloads: std::sync::atomic::AtomicUsize::new(0),
        }));
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let b = provider
        .upload(&HostTensorView {
            data: &[0.5, 0.5],
            shape: &[1, 2],
        })
        .expect("resident coefficients");
    let empty_a = Tensor::new(Vec::new(), vec![0, 1]).expect("empty denominator");
    let error = call(
        Value::GpuTensor(b.clone()),
        Value::Tensor(empty_a),
        Value::Num(1.0),
    )
    .expect_err("known-invalid denominator");
    assert_eq!(
        error.identifier(),
        FILTFILT_ERROR_INVALID_COEFFICIENTS.identifier
    );
    assert_eq!(
        provider.downloads.load(std::sync::atomic::Ordering::SeqCst),
        0
    );
    assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    b.release().expect("release original resident coefficients");
}

#[test]
fn later_stale_resident_signal_precedes_numerator_download() {
    let _lock = test_support::accel_test_lock();
    let provider = crate::builtins::math::signal::common::convolution::test_support::RejectTransferProvider::new();
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let numerator = provider.input(vec![1.0, 1.0], vec![1, 2]);
    let signal = provider.input(vec![1.0; 8], vec![1, 8]);
    signal.probe.mark_unavailable();
    let error = call(
        Value::GpuTensor(numerator.handle.clone()),
        Value::Num(1.0),
        Value::GpuTensor(signal.handle.clone()),
    )
    .expect_err("signal lease must be checked before numerator transfer");
    assert_eq!(error.identifier(), FILTFILT_ERROR_INVALID_SIGNAL.identifier);
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(std::error::Error::source(&error).is_some());
    assert_eq!(provider.downloads(), 0);
    assert!(numerator.handle.allocation_lease().validate_live().is_ok());
    assert_eq!(numerator.probe.retirements(), 0);
}

#[test]
fn gpu_matches_cpu() {
    test_support::with_test_provider(|provider| {
        let b = Tensor::new(vec![1.0 / 3.0; 3], vec![1, 3]).unwrap();
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let x = Tensor::new((1..=20).map(|v| v as f64).collect(), vec![1, 20]).unwrap();
        let cpu = call(
            Value::Tensor(b.clone()),
            Value::Tensor(a.clone()),
            Value::Tensor(x.clone()),
        )
        .unwrap();
        let (cpu_data, _) = tensor_data(cpu);
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &x.materialize_f64(),
                shape: x.shape(),
            })
            .expect("upload");
        let gpu = call(Value::Tensor(b), Value::Tensor(a), Value::GpuTensor(handle)).unwrap();
        let tol = value_tolerance(&gpu);
        let (gpu_data, shape) = tensor_data(gpu);
        assert_eq!(shape, vec![1, 20]);
        approx(&gpu_data, &cpu_data, tol);
    });
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires a live WGPU adapter; run explicitly for hardware qualification"]
fn wgpu_matches_cpu() {
    let _serial = test_support::accel_test_lock();
    let _restore = runmat_accelerate_api::ThreadProviderGuard::set(None);
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("live WGPU adapter required");
    let b = Tensor::new(vec![1.0 / 3.0; 3], vec![1, 3]).unwrap();
    let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let x = Tensor::new((1..=20).map(|v| v as f64).collect(), vec![1, 20]).unwrap();
    let cpu = call(
        Value::Tensor(b.clone()),
        Value::Tensor(a.clone()),
        Value::Tensor(x.clone()),
    )
    .unwrap();
    let (cpu_data, _) = tensor_data(cpu);
    let handle = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &x.materialize_f64(),
            shape: x.shape(),
        })
        .expect("upload");
    let borrowed_signal = handle.clone();
    let gpu = call(Value::Tensor(b), Value::Tensor(a), Value::GpuTensor(handle)).unwrap();
    assert!(matches!(&gpu, Value::GpuTensor(_)));
    let tolerance = value_tolerance(&gpu);
    let (gpu_data, _) = tensor_data(gpu);
    approx(&gpu_data, &cpu_data, tolerance);
    assert!(borrowed_signal.allocation_lease().validate_live().is_ok());
}
