use super::*;
use crate::builtins::common::test_support;
use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    unsupported_provider_operation, AccelDownloadFuture, AccelProvider, GpuTensorHandle,
    GpuTensorStorage, HostTensorView, NumericElementType, ProviderExecutionFuture,
    ProviderHilbertRequest, ThreadProviderGuard,
};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

struct NoHookProvider {
    delegate: InProcessProvider,
    hooks: AtomicUsize,
}

impl AccelProvider for NoHookProvider {
    fn device_id(&self) -> u32 {
        self.delegate.device_id()
    }
    fn device_info(&self) -> String {
        "hilbert no-hook fixture".into()
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.delegate.upload(host)
    }
    fn upload_numeric_raw(
        &self,
        host: &runmat_accelerate_api::HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.delegate.upload_numeric_raw(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.delegate.download(handle)
    }
    fn signal_hilbert<'a>(
        &'a self,
        _request: &'a ProviderHilbertRequest<'a>,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.hooks.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { panic!("unadmitted Hilbert hook must not execute") })
    }
}

#[test]
fn missing_hook_is_preflighted_before_any_provider_effect() {
    let _lock = test_support::accel_test_lock();
    let provider = Box::leak(Box::new(NoHookProvider {
        delegate: InProcessProvider::new(),
        hooks: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .upload(&HostTensorView {
            data: &[1.0, 0.0, -1.0, 0.0],
            shape: &[1, 4],
        })
        .unwrap();
    let output = hilbert_call(Value::GpuTensor(input.clone()), Vec::new()).unwrap();
    let complex = as_complex_tensor(output);
    assert_eq!(complex.shape(), &[1, 4]);
    assert!((complex.materialize_f64()[1].1 - 1.0).abs() < 1.0e-12);
    assert_eq!(provider.hooks.load(Ordering::SeqCst), 0);
    input.release().unwrap();
}

#[derive(Clone, Copy)]
enum SupportedMode {
    UnsupportedAfterEffect,
    UnsupportedUnphased,
    CancelAfterOutput,
    PendingAfterOutput,
    WrongShape,
    WrongStorage,
    ForeignOwner,
    InputAlias,
}

struct SupportedProvider {
    owner: TestAllocationOwner,
    foreign: TestAllocationOwner,
    mode: SupportedMode,
    cancellation: CancellationSource,
    hooks: AtomicUsize,
    downloads: AtomicUsize,
    produced: Mutex<Vec<RetirementProbe>>,
}

impl SupportedProvider {
    fn new(mode: SupportedMode) -> &'static Self {
        Box::leak(Box::new(Self {
            owner: TestAllocationOwner::new(),
            foreign: TestAllocationOwner::new(),
            mode,
            cancellation: CancellationSource::new(),
            hooks: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
            produced: Mutex::new(Vec::new()),
        }))
    }
}

impl AccelProvider for SupportedProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "hilbert supported fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("host upload is not part of this resident route")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { panic!("invoked Hilbert failures must not download") })
    }
    fn supports_signal_hilbert(&self, _: &ProviderHilbertRequest<'_>) -> bool {
        true
    }
    fn signal_hilbert<'a>(
        &'a self,
        request: &'a ProviderHilbertRequest<'a>,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.hooks.fetch_add(1, Ordering::SeqCst);
            if matches!(self.mode, SupportedMode::InputAlias) {
                return Ok(request.input.clone());
            }
            let output_owner = if matches!(self.mode, SupportedMode::ForeignOwner) {
                &self.foreign
            } else {
                &self.owner
            };
            let output_shape = if matches!(self.mode, SupportedMode::WrongShape) {
                vec![2, 2]
            } else {
                request.input.shape().to_vec()
            };
            let output_storage = if matches!(self.mode, SupportedMode::WrongStorage) {
                GpuTensorStorage::Real
            } else {
                GpuTensorStorage::ComplexInterleaved
            };
            let produced = output_owner.allocate_zeroed(
                output_shape,
                NumericElementType::F64,
                output_storage,
            )?;
            self.produced.lock().unwrap().push(produced.probe);
            match self.mode {
                SupportedMode::UnsupportedAfterEffect => {
                    produced
                        .handle
                        .release()
                        .expect("retire staged fixture output");
                    Err(runmat_accelerate_api::ProviderExecutionError::from(
                        unsupported_provider_operation("unsupported after Hilbert staging"),
                    )
                    .after_effect())
                }
                SupportedMode::UnsupportedUnphased => {
                    produced
                        .handle
                        .release()
                        .expect("retire staged fixture output");
                    Err(unsupported_provider_operation("unphased unsupported after staging").into())
                }
                SupportedMode::CancelAfterOutput => {
                    self.cancellation.cancel(CancellationReason::User);
                    Ok(produced.handle)
                }
                SupportedMode::PendingAfterOutput => {
                    std::future::pending::<()>().await;
                    Ok(produced.handle)
                }
                SupportedMode::WrongShape
                | SupportedMode::WrongStorage
                | SupportedMode::ForeignOwner
                | SupportedMode::InputAlias => Ok(produced.handle),
            }
        })
    }
}

#[test]
fn malformed_foreign_and_alias_outputs_are_rejected_without_retiring_input() {
    for mode in [
        SupportedMode::WrongShape,
        SupportedMode::WrongStorage,
        SupportedMode::ForeignOwner,
        SupportedMode::InputAlias,
    ] {
        let provider = SupportedProvider::new(mode);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_zeroed(vec![1, 4], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let error = hilbert_call(Value::GpuTensor(input.handle.clone()), Vec::new())
            .expect_err("provider output must match the declared Hilbert contract");
        assert_eq!(error.identifier(), HILBERT_ERROR_INTERNAL.identifier);
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        if !matches!(mode, SupportedMode::InputAlias) {
            assert_eq!(provider.produced.lock().unwrap()[0].retirements(), 1);
        }
        assert_eq!(input.probe.retirements(), 0);
        input.handle.release().unwrap();
    }
}

#[test]
fn invoked_unsupported_after_effect_is_terminal_without_host_replay() {
    for mode in [
        SupportedMode::UnsupportedAfterEffect,
        SupportedMode::UnsupportedUnphased,
    ] {
        let provider = SupportedProvider::new(mode);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_zeroed(vec![1, 4], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let error = hilbert_call(Value::GpuTensor(input.handle.clone()), Vec::new())
            .expect_err("post-effect Unsupported is not a fallback permit");
        assert_eq!(error.identifier(), HILBERT_ERROR_INTERNAL.identifier);
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(std::error::Error::source(&error).is_some());
        assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.produced.lock().unwrap()[0].retirements(), 1);
        assert_eq!(input.probe.retirements(), 0);
        input.handle.release().unwrap();
    }
}

#[test]
fn cancellation_after_hook_result_retires_only_new_output() {
    let provider = SupportedProvider::new(SupportedMode::CancelAfterOutput);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_zeroed(vec![1, 4], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let mut cooperation = SignalCooperation::with_budget(
        CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::NeverYield,
            provider.cancellation.token(),
            || Box::pin(async { Ok(()) }),
        ),
        BUILTIN_NAME,
    );
    let error = block_on(hilbert_gpu_tensor(
        input.handle.clone(),
        None,
        &mut cooperation,
    ))
    .expect_err("cancelled post-hook output must not publish");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.produced.lock().unwrap()[0].retirements(), 1);
    assert_eq!(input.probe.retirements(), 0);
    input.handle.release().unwrap();
}

#[test]
fn unpolled_and_pending_dropped_calls_do_not_publish_or_replay() {
    use std::future::Future;
    use std::task::{Context, Poll};

    let provider = SupportedProvider::new(SupportedMode::PendingAfterOutput);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_zeroed(vec![1, 4], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let unpolled = Box::pin(hilbert_call_async(Value::GpuTensor(input.handle.clone())));
    drop(unpolled);
    assert_eq!(provider.hooks.load(Ordering::SeqCst), 0);
    assert!(provider.produced.lock().unwrap().is_empty());

    let mut future = Box::pin(hilbert_call_async(Value::GpuTensor(input.handle.clone())));
    let waker = futures::task::noop_waker();
    assert!(matches!(
        future.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
    assert_eq!(provider.produced.lock().unwrap()[0].retirements(), 0);
    drop(future);
    assert_eq!(provider.produced.lock().unwrap()[0].retirements(), 1);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(input.probe.retirements(), 0);
    input.handle.release().unwrap();
}

async fn hilbert_call_async(value: Value) -> BuiltinResult<Value> {
    hilbert_builtin(
        None,
        value,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .await
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires an actual WGPU adapter; run explicitly for F32 and F64"]
fn actual_lazy_transpose_uses_checked_host_materialization() {
    use runmat_accelerate::backend::wgpu::provider::WgpuProvider;
    use runmat_accelerate_api::{HostNumericDataView, HostNumericTensorView, ProviderPrecision};

    let provider = Box::leak(Box::new(
        WgpuProvider::new(Default::default()).expect("actual WGPU adapter required"),
    ));
    match std::env::var("RUNMAT_WGPU_FORCE_PRECISION").as_deref() {
        Ok("f32") => assert_eq!(provider.precision(), ProviderPrecision::F32),
        Ok("f64") => assert_eq!(provider.precision(), ProviderPrecision::F64),
        _ => {}
    }
    let _selected = ThreadProviderGuard::set(Some(provider));
    let original_values = [1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0];
    let single_values = original_values.map(|value| value as f32);
    let input = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: match provider.precision() {
                ProviderPrecision::F32 => HostNumericDataView::F32(&single_values),
                ProviderPrecision::F64 => HostNumericDataView::F64(&original_values),
            },
            shape: &[4, 2],
            storage: GpuTensorStorage::Real,
        })
        .expect("upload original real matrix");
    let transposed = provider.transpose(&input).expect("lazy transpose view");
    assert!(runmat_accelerate_api::handle_is_transposed(&transposed));
    let request = ProviderHilbertRequest {
        input: &transposed,
        length: None,
        dim: 0,
    };
    assert!(!provider.supports_signal_hilbert(&request));
    let direct = block_on(provider.signal_hilbert(&request))
        .expect_err("direct hook cannot reinterpret a transpose as contiguous");
    assert!(direct.is_unsupported());
    assert!(!direct.is_after_effect());

    // Column-major transpose of the uploaded 4-by-2 matrix.
    let logical = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0];
    let host = Tensor::new(logical.to_vec(), vec![2, 4]).unwrap();
    let expected = as_complex_tensor(hilbert_call(Value::Tensor(host), Vec::new()).unwrap());
    let actual = as_complex_tensor(
        hilbert_call(Value::GpuTensor(transposed.clone()), Vec::new())
            .expect("runtime must use owner-aware host materialization"),
    );
    assert_eq!(actual.shape(), &[2, 4]);
    for (actual, expected) in actual
        .materialize_f64()
        .iter()
        .copied()
        .zip(expected.materialize_f64())
    {
        let tolerance = if provider.precision() == ProviderPrecision::F32 {
            2.0e-4
        } else {
            1.0e-10
        };
        assert!((actual.0 - expected.0).abs() <= tolerance);
        assert!((actual.1 - expected.1).abs() <= tolerance);
    }
    input.allocation_lease().validate_live().unwrap();
    transposed.allocation_lease().validate_live().unwrap();
}
