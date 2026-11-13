//! Standalone dispatch avoids the duplicate inventory in runtime's unit harness.
#![cfg(not(target_arch = "wasm32"))]

#[path = "support/circshift_preflight.rs"]
mod circshift_preflight;
#[path = "support/fft_catalog_binding.rs"]
mod fft_catalog_binding;

use futures::executor::block_on;
use runmat_accelerate_api::test_support::{RetirementProbe, TestAllocationOwner};
use runmat_accelerate_api::*;
use runmat_runtime::GpuGatherRetry;
use runmat_value::Value;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Debug)]
struct FftFailure;
impl std::fmt::Display for FftFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Deliberately resembles the old string-based fallback classifier.
        f.write_str("ifft_dim not supported by provider")
    }
}
impl std::error::Error for FftFailure {}

struct Provider {
    owner: TestAllocationOwner,
    calls: AtomicUsize,
    fail_after: usize,
    suspend_second: bool,
    partials: Mutex<Vec<RetirementProbe>>,
    unsupported: bool,
    source: Mutex<Option<RetirementProbe>>,
    downloads: AtomicUsize,
}
impl Provider {
    async fn fft(&self, input: &GpuTensorHandle) -> ProviderExecutionResult<GpuTensorHandle> {
        let pass = self.calls.fetch_add(1, Ordering::SeqCst);
        if self.suspend_second && pass == 1 {
            return std::future::pending().await;
        }
        if self.unsupported {
            return Err(unsupported_provider_operation("fixture FFT unsupported").into());
        }
        if pass == self.fail_after {
            return Err(ProviderExecutionError::from_source(FftFailure));
        }
        let partial = self.owner.allocate_zeroed(
            input.shape().to_vec(),
            NumericElementType::F64,
            GpuTensorStorage::ComplexInterleaved,
        )?;
        self.partials.lock().unwrap().push(partial.probe);
        Ok(partial.handle)
    }
    fn fail<T>(&self) -> ProviderExecutionResult<T> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(ProviderExecutionError::from_source(FftFailure))
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "FFT execution source fixture".into()
    }
    fn precision(&self) -> ProviderPrecision {
        if self.unsupported {
            ProviderPrecision::F32
        } else {
            ProviderPrecision::F64
        }
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("operational failure must not replay/upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("operational failure must not gather")
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        assert!(
            self.unsupported,
            "operational failure must not gather native payload"
        );
        Box::pin(async move {
            self.downloads.fetch_add(1, Ordering::SeqCst);
            self.source
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .with_data(handle, |data| {
                    let HostNumericDataView::F64(values) = data else {
                        panic!("fixture payload is native F64")
                    };
                    HostNumericTensorOwned {
                        data: HostNumericDataOwned::F64(values.to_vec()),
                        shape: handle.shape().to_vec(),
                        storage: handle_storage(handle),
                    }
                })
        })
    }
    fn fft_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        _: Option<usize>,
        _: usize,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fft(input).await })
    }
    fn ifft_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        _: Option<usize>,
        _: usize,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fft(input).await })
    }
    fn uniform_spectral_estimate<'a>(
        &'a self,
        _: &'a ProviderSpectralRequest<'a>,
    ) -> ProviderExecutionFuture<'a, ProviderSpectralResult> {
        Box::pin(async move { self.fail() })
    }
    fn supports_signal_envelope(&self, _: &ProviderEnvelopeRequest<'_>) -> bool {
        true
    }
    fn signal_envelope<'a>(
        &'a self,
        _: &'a ProviderEnvelopeRequest<'a>,
    ) -> ProviderExecutionFuture<'a, ProviderEnvelopeResult> {
        Box::pin(async move { self.fail() })
    }
    fn supports_signal_hilbert(&self, _: &ProviderHilbertRequest<'_>) -> bool {
        true
    }
    fn signal_hilbert<'a>(
        &'a self,
        _: &'a ProviderHilbertRequest<'a>,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fail() })
    }
}

fn check_failure(name: &str, fail_after: usize) {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        calls: AtomicUsize::new(0),
        fail_after,
        suspend_second: false,
        partials: Mutex::new(Vec::new()),
        unsupported: false,
        source: Mutex::new(None),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_zeroed(
            vec![512, 1],
            NumericElementType::F64,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let args = vec![Value::GpuTensor(input.handle.clone())];
    let error = block_on(runmat_runtime::call_builtin_async(name, &args)).unwrap_err();
    assert_eq!(
        provider.calls.load(Ordering::SeqCst),
        fail_after + 1,
        "{name}"
    );
    assert_eq!(
        error.gpu_gather_retry,
        GpuGatherRetry::Never,
        "{name}: {error}"
    );
    let mut cause: &(dyn std::error::Error + 'static) = &error;
    while !cause.is::<FftFailure>() {
        cause = cause
            .source()
            .unwrap_or_else(|| panic!("{name} lost original source: {error}"));
    }
    for partial in provider.partials.lock().unwrap().iter() {
        assert_eq!(
            partial.retirements(),
            1,
            "{name}: unpublished intermediate leaked"
        );
    }
    input.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(input.probe.retirements(), 0);
    drop(error);
    drop(args);
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

#[test]
fn fft_and_analytic_signal_failures_retain_source_without_gather_or_retry() {
    for name in [
        "fft",
        "fft2",
        "fftn",
        "ifft",
        "ifft2",
        "ifftn",
        "hilbert",
        "envelope",
        "pwelch",
        "periodogram",
        "spectrogram",
    ] {
        check_failure(name, 0);
    }
}

#[test]
fn later_fft_pass_failure_retires_partial_but_keeps_borrowed_input() {
    for name in ["fft2", "fftn", "ifft2", "ifftn"] {
        check_failure(name, 1);
    }
}

#[test]
fn typed_unsupported_runs_existing_host_fft_without_replaying_provider() {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        calls: AtomicUsize::new(0),
        fail_after: 0,
        suspend_second: false,
        partials: Mutex::new(Vec::new()),
        unsupported: true,
        source: Mutex::new(None),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![1.0, 2.0, 3.0, 4.0]),
            shape: vec![4, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    *provider.source.lock().unwrap() = Some(input.probe);
    let args = vec![Value::GpuTensor(input.handle.clone())];
    let result = block_on(runmat_runtime::call_builtin_async("fft", &args)).unwrap();
    let Value::ComplexTensor(result) = result.value() else {
        panic!("existing precision-preserving restoration retains host result")
    };
    assert_eq!(result.shape(), &[4, 1]);
    let runmat_value::ComplexStorage::F64(data) = result.complex_storage() else {
        panic!("FFT preserves native double output")
    };
    for (actual, expected) in data
        .iter()
        .zip([(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)])
    {
        assert!((actual.0 - expected.0).abs() < 1e-12);
        assert!((actual.1 - expected.1).abs() < 1e-12);
    }
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    assert_eq!(
        provider
            .source
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .retirements(),
        0
    );
    drop(args);
    drop(input.handle);
    assert_eq!(
        provider
            .source
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .retirements(),
        1
    );
}

#[test]
fn cancellation_during_second_fft_pass_retires_only_execution_owned_partial() {
    use std::{
        future::Future,
        task::{Context, Poll},
    };
    for name in ["fft2", "fftn", "ifft2", "ifftn"] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            calls: AtomicUsize::new(0),
            fail_after: usize::MAX,
            suspend_second: true,
            partials: Mutex::new(Vec::new()),
            unsupported: false,
            source: Mutex::new(None),
            downloads: AtomicUsize::new(0),
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_zeroed(vec![4, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let args = vec![Value::GpuTensor(input.handle.clone())];
        let mut future = Box::pin(runmat_runtime::call_builtin_async(name, &args));
        let waker = futures::task::noop_waker();
        assert!(matches!(
            future.as_mut().poll(&mut Context::from_waker(&waker)),
            Poll::Pending
        ));
        assert_eq!(
            provider.calls.load(Ordering::SeqCst),
            2,
            "{name}: second pass entered"
        );
        {
            let partials = provider.partials.lock().unwrap();
            assert_eq!(partials.len(), 1);
            assert_eq!(
                partials[0].retirements(),
                0,
                "{name}: first output remains owned across suspension"
            );
        }
        drop(future);
        assert_eq!(
            provider.partials.lock().unwrap()[0].retirements(),
            1,
            "{name}: cancellation retires first output exactly once"
        );
        assert_eq!(
            provider.calls.load(Ordering::SeqCst),
            2,
            "{name}: no replay on cancellation"
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        input.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input.probe.retirements(), 0);
        drop(args);
        drop(input.handle);
        assert_eq!(input.probe.retirements(), 1);
    }
}
