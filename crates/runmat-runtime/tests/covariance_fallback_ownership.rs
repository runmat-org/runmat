use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::allocation::{WeakAllocationLease, WeakGpuTensorHandle};
use runmat_accelerate_api::*;
use runmat_runtime::{call_builtin_async_with_outputs, GpuGatherRetry};
use runmat_value::{Tensor, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Clone, Copy)]
enum Mode {
    Alias,
    UploadFailure,
    UnsupportedWeight,
    Pending,
    PendingFallback,
}
struct Provider {
    inner: InProcessProvider,
    mode: Mode,
    alias: Mutex<Option<WeakGpuTensorHandle>>,
    uploads: AtomicUsize,
    downloads: AtomicUsize,
    calls: AtomicUsize,
    uploaded: Mutex<Vec<WeakAllocationLease>>,
}
impl Provider {
    fn new(mode: Mode) -> &'static Self {
        Box::leak(Box::new(Self {
            inner: Default::default(),
            mode,
            alias: Mutex::new(None),
            uploads: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
            calls: AtomicUsize::new(0),
            uploaded: Mutex::new(vec![]),
        }))
    }
    fn input(&self, data: &[f64], shape: &[usize]) -> GpuTensorHandle {
        self.inner.upload(&HostTensorView { data, shape }).unwrap()
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn upload(&self, host: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let attempt = self.uploads.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            Mode::Alias => {
                return self
                    .alias
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .acquire()?
                    .with_view(GpuTensorView::contiguous(host.shape.to_vec()))
                    .map_err(Into::into)
            }
            Mode::UploadFailure => {
                return Err(runmat_accelerate_api::ProviderExecutionError::from_source(
                    runmat_runtime::build_runtime_error("operational covariance weight upload")
                        .with_gpu_gather_retry(GpuGatherRetry::Requested)
                        .with_source(std::io::Error::new(
                            std::io::ErrorKind::BrokenPipe,
                            "weight transport failed",
                        ))
                        .build(),
                ))
            }
            Mode::UnsupportedWeight if attempt == 0 => {
                return Err(
                    unsupported_provider_operation("fixture does not upload weights").into(),
                )
            }
            _ => {}
        }
        let output = self.inner.upload_numeric_raw(host)?;
        self.uploaded
            .lock()
            .unwrap()
            .push(output.allocation_lease().downgrade());
        Ok(output)
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        let attempt = self.downloads.fetch_add(1, Ordering::SeqCst);
        if matches!(self.mode, Mode::PendingFallback) && attempt == 1 {
            return Box::pin(std::future::pending());
        }
        self.inner.download(input)
    }
    fn download_numeric<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        let attempt = self.downloads.fetch_add(1, Ordering::SeqCst);
        if matches!(self.mode, Mode::PendingFallback) && attempt == 1 {
            return Box::pin(std::future::pending());
        }
        self.inner.download_numeric(input)
    }
    fn covariance<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: Option<&'a GpuTensorHandle>,
        _: Option<&'a GpuTensorHandle>,
        _: &'a CovarianceOptions,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            if matches!(self.mode, Mode::Pending) {
                std::future::pending().await
            } else {
                Err(unsupported_provider_operation("fixture requires host covariance").into())
            }
        })
    }
}
fn matrix(provider: &Provider) -> GpuTensorHandle {
    provider.input(&[1., 2., 3., 4., 2., 4., 6., 8.], &[4, 2])
}
fn host_weights() -> Value {
    Value::Tensor(Tensor::new(vec![1., 2., 3., 4.], vec![4, 1]).unwrap())
}

#[test]
fn fallback_upload_rejects_aliases_of_second_input_and_weights() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for weight_alias in [false, true] {
        let provider = Provider::new(Mode::Alias);
        let _active = ThreadProviderGuard::set(Some(provider));
        let first = if weight_alias {
            matrix(provider)
        } else {
            provider.input(&[1., 2., 3., 4.], &[2, 2])
        };
        let other = if weight_alias {
            provider.input(&[1.; 4], &[4, 1])
        } else {
            provider.input(&[4., 3., 2., 1.], &[2, 2])
        };
        *provider.alias.lock().unwrap() = Some(other.downgrade());
        let first_weak = first.allocation_lease().downgrade();
        let other_weak = other.allocation_lease().downgrade();
        let error = block_on(call_builtin_async_with_outputs(
            "cov",
            &[
                Value::GpuTensor(first.clone()),
                Value::GpuTensor(other.clone()),
            ],
            1,
        ))
        .unwrap_err();
        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
        assert!(is_invalid_numeric_upload_result(&error));
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 2);
        assert!(first_weak.acquire().is_some());
        assert!(other_weak.acquire().is_some());
        assert!(block_on(provider.inner.download(&other)).is_ok());
        drop(error);
        drop((first, other));
        assert!(first_weak.acquire().is_none());
        assert!(other_weak.acquire().is_none());
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn operational_weight_upload_failure_is_terminal_even_with_nested_retry_request() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let provider = Provider::new(Mode::UploadFailure);
    let _active = ThreadProviderGuard::set(Some(provider));
    let input = matrix(provider);
    let error = block_on(call_builtin_async_with_outputs(
        "cov",
        &[Value::GpuTensor(input.clone()), host_weights()],
        1,
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:cov:Internal"));
    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
    let mut source: &(dyn std::error::Error + 'static) = &error;
    let mut retained_inner_request = false;
    loop {
        retained_inner_request |= source
            .downcast_ref::<runmat_runtime::RuntimeError>()
            .is_some_and(|nested| nested.gpu_gather_retry == GpuGatherRetry::Requested);
        if let Some(cause) = source.downcast_ref::<std::io::Error>() {
            assert_eq!(cause.kind(), std::io::ErrorKind::BrokenPipe);
            break;
        }
        source = source.source().expect("original transport cause");
    }
    assert!(
        retained_inner_request,
        "inner retry request remains in the cause chain"
    );
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert!(input.allocation_lease().validate_live().is_ok());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn typed_unsupported_weight_upload_uses_supported_host_compute_and_restore() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let provider = Provider::new(Mode::UnsupportedWeight);
    let _active = ThreadProviderGuard::set(Some(provider));
    let input = matrix(provider);
    let result = block_on(call_builtin_async_with_outputs(
        "cov",
        &[Value::GpuTensor(input.clone()), host_weights()],
        1,
    ))
    .unwrap();
    let Value::GpuTensor(output) = result.value() else {
        panic!("restored covariance")
    };
    let host = block_on(provider.inner.download(output)).unwrap();
    assert_eq!(host.shape, [2, 2]);
    assert_eq!(host.data.len(), 4);
    for (actual, expected) in host
        .data
        .iter()
        .zip([10. / 9., 20. / 9., 20. / 9., 40. / 9.])
    {
        assert!((actual - expected).abs() < 1e-12);
    }
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 2);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    drop(result);
    assert!(provider
        .uploaded
        .lock()
        .unwrap()
        .iter()
        .all(|lease| lease.acquire().is_none()));
}

#[test]
fn cancelling_actual_covariance_future_retires_owned_weight_upload_only() {
    use std::{
        future::Future,
        task::{Context, Poll},
    };
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let provider = Provider::new(Mode::Pending);
    let _active = ThreadProviderGuard::set(Some(provider));
    let input = matrix(provider);
    let args = vec![Value::GpuTensor(input.clone()), host_weights()];
    drop(call_builtin_async_with_outputs("cov", &args, 1));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    let mut call = Box::pin(call_builtin_async_with_outputs("cov", &args, 1));
    let waker = futures::task::noop_waker();
    assert!(matches!(
        call.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert!(provider.uploaded.lock().unwrap()[0].acquire().is_some());
    drop(call);
    assert!(provider.uploaded.lock().unwrap()[0].acquire().is_none());
    assert!(input.allocation_lease().validate_live().is_ok());
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}

#[test]
fn invalid_pair_geometry_and_empty_weight_cardinality_precede_all_hooks() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for (left_shape, right_shape) in [(vec![2, 2], vec![1, 4]), (vec![2, 2], vec![2, 2, 1])] {
        let provider = Provider::new(Mode::Pending);
        let _active = ThreadProviderGuard::set(Some(provider));
        let left = provider.input(&[1.; 4], &left_shape);
        let right = provider.input(&[1.; 4], &right_shape);
        let host_error = block_on(call_builtin_async_with_outputs(
            "cov",
            &[
                Value::Tensor(Tensor::new(vec![1.; 4], left_shape).unwrap()),
                Value::Tensor(Tensor::new(vec![1.; 4], right_shape).unwrap()),
                host_weights(),
            ],
            1,
        ))
        .unwrap_err();
        let resident_error = block_on(call_builtin_async_with_outputs(
            "cov",
            &[
                Value::GpuTensor(left),
                Value::GpuTensor(right),
                host_weights(),
            ],
            1,
        ))
        .unwrap_err();
        assert_eq!(resident_error.identifier(), host_error.identifier());
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    }
    let provider = Provider::new(Mode::Pending);
    let _active = ThreadProviderGuard::set(Some(provider));
    let left = provider.input(&[1.], &[1, 1]);
    let right = provider.input(&[2.], &[1, 1]);
    let weights = provider.input(&[], &[0, 1]);
    let error = block_on(call_builtin_async_with_outputs(
        "cov",
        &[
            Value::GpuTensor(left),
            Value::GpuTensor(right),
            Value::GpuTensor(weights),
        ],
        1,
    ))
    .unwrap_err();
    let host_error = block_on(call_builtin_async_with_outputs(
        "cov",
        &[
            Value::Tensor(Tensor::new(vec![1.], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![2.], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![], vec![0, 1]).unwrap()),
        ],
        1,
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:cov:InvalidArgument"));
    assert_eq!(error.identifier(), host_error.identifier());
    assert_eq!(error.message(), host_error.message());
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}

#[test]
fn invalid_weight_cardinality_keeps_priority_over_pair_shape_without_effects() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let provider = Provider::new(Mode::Pending);
    let _active = ThreadProviderGuard::set(Some(provider));
    let left = provider.input(&[1.; 6], &[3, 2]);
    let right = provider.input(&[1.; 3], &[3, 1]);
    let weights = Value::Tensor(Tensor::new(vec![1., 2.], vec![2, 1]).unwrap());
    let error = block_on(call_builtin_async_with_outputs(
        "cov",
        &[Value::GpuTensor(left), Value::GpuTensor(right), weights],
        1,
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:cov:WeightVectorLengthMismatch")
    );
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}

#[test]
fn cancelling_pending_fallback_download_preserves_all_original_leases() {
    use std::{
        future::Future,
        task::{Context, Poll},
    };
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let provider = Provider::new(Mode::PendingFallback);
    let _active = ThreadProviderGuard::set(Some(provider));
    let input = matrix(provider);
    let weights = provider.input(&[1., 2., 3., 4.], &[4, 1]);
    let input_weak = input.allocation_lease().downgrade();
    let weight_weak = weights.allocation_lease().downgrade();
    let args = vec![
        Value::GpuTensor(input.clone()),
        Value::GpuTensor(weights.clone()),
    ];
    let mut call = Box::pin(call_builtin_async_with_outputs("cov", &args, 1));
    let waker = futures::task::noop_waker();
    match call.as_mut().poll(&mut Context::from_waker(&waker)) {
        Poll::Pending => {}
        Poll::Ready(result) => panic!("expected pending fallback weight download, got {result:?}"),
    }
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 2);
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    drop(call);
    assert_eq!(
        block_on(provider.inner.download(&input)).unwrap().data,
        [1., 2., 3., 4., 2., 4., 6., 8.]
    );
    assert_eq!(
        block_on(provider.inner.download(&weights)).unwrap().data,
        [1., 2., 3., 4.]
    );
    assert!(input_weak.acquire().is_some());
    assert!(weight_weak.acquire().is_some());
    assert!(provider.uploaded.lock().unwrap().is_empty());
    drop(args);
    drop((input, weights));
    assert!(input_weak.acquire().is_none());
    assert!(weight_weak.acquire().is_none());
}
