use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::*;
use runmat_runtime::{call_builtin_async_with_outputs, GpuGatherRetry};
use runmat_value::{Tensor, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Clone, Copy, Debug)]
enum Mode {
    Success,
    Unsupported,
    Failure,
    Shape,
    Class,
    Complex,
    Integer,
    Logical,
    Owner,
    Alias,
    LaterFailure,
    LaterUnsupported,
    LaterAlias,
    LaterPending,
    RestoreUnsupported,
    RestoreFailure,
}
struct Provider {
    inner: InProcessProvider,
    foreign: InProcessProvider,
    mode: Mode,
    calls: AtomicUsize,
    downloads: AtomicUsize,
    uploads: AtomicUsize,
    alias: Mutex<Option<allocation::WeakGpuTensorHandle>>,
    produced: Mutex<Vec<allocation::WeakAllocationLease>>,
}
impl Provider {
    fn new(mode: Mode) -> &'static Self {
        Box::leak(Box::new(Self {
            inner: InProcessProvider::new(),
            foreign: InProcessProvider::new(),
            mode,
            calls: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
            uploads: AtomicUsize::new(0),
            alias: Mutex::new(None),
            produced: Mutex::new(vec![]),
        }))
    }
    async fn execute(
        &self,
        input: &GpuTensorHandle,
        axis: Option<usize>,
    ) -> anyhow::Result<GpuTensorHandle> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            Mode::Unsupported => return Err(unsupported_provider_operation("median fixture")),
            Mode::LaterUnsupported if call > 0 => {
                return Err(unsupported_provider_operation("later median fixture"))
            }
            Mode::Failure => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "median unsupported-looking operational failure",
                )
                .into())
            }
            Mode::LaterFailure if call > 0 => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "later operational failure",
                )
                .into())
            }
            Mode::LaterPending if call > 0 => return std::future::pending().await,
            _ => {}
        }
        let shape = match axis {
            Some(axis) => {
                let mut shape = input.shape().to_vec();
                shape[axis] = 1;
                shape
            }
            None => vec![1, 1],
        };
        if matches!(self.mode, Mode::Alias) || (matches!(self.mode, Mode::LaterAlias) && call > 0) {
            return self
                .alias
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .acquire()?
                .with_view(GpuTensorView::contiguous(shape))
                .map_err(Into::into);
        }
        let len = shape.iter().product();
        let output = match self.mode {
            Mode::Complex => self.inner.upload_numeric_host(&HostNumericTensorView {
                data: HostNumericDataView::F64(&vec![0.; len * 2]),
                shape: &shape,
                storage: GpuTensorStorage::ComplexInterleaved,
            })?,
            Mode::Integer => self.inner.upload_numeric_host(&HostNumericTensorView {
                data: HostNumericDataView::I32(&vec![0; len]),
                shape: &shape,
                storage: GpuTensorStorage::Real,
            })?,
            Mode::Class => self.inner.upload_numeric_host(&HostNumericTensorView {
                data: HostNumericDataView::F32(&vec![0.; len]),
                shape: &shape,
                storage: GpuTensorStorage::Real,
            })?,
            Mode::Logical => self
                .inner
                .upload_logical_raw(&HostLogicalTensorView {
                    data: &vec![0; len],
                    shape: &shape,
                })
                .expect("fixed logical fixture upload must succeed"),
            Mode::Owner => self.foreign.upload(&HostTensorView {
                data: &vec![0.; len],
                shape: &shape,
            })?,
            _ => match axis {
                Some(axis) => self.inner.reduce_median_dim(input, axis).await?,
                None => self.inner.reduce_median(input).await?,
            },
        };
        let output = if matches!(self.mode, Mode::Shape) {
            output.with_view(GpuTensorView::contiguous(vec![len, 1]))?
        } else {
            output
        };
        self.produced
            .lock()
            .unwrap()
            .push(output.allocation_lease().downgrade());
        Ok(output)
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
        self.uploads.fetch_add(1, Ordering::SeqCst);
        self.inner.upload(host)
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        if matches!(self.mode, Mode::RestoreUnsupported | Mode::RestoreFailure) {
            assert_eq!(host.shape, &[1, 3]);
            assert!(
                matches!(host.data, HostNumericDataView::F64(values) if values.len() == 3 && values.iter().all(|value| value.is_nan()))
            );
            return if matches!(self.mode, Mode::RestoreUnsupported) {
                Err(
                    unsupported_provider_operation("fixture cannot restore Double empty median")
                        .into(),
                )
            } else {
                Err(
                    std::io::Error::new(std::io::ErrorKind::BrokenPipe, "restoration failed")
                        .into(),
                )
            };
        }
        self.inner.upload_numeric_raw(host)
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(input)
    }
    fn download_numeric<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download_numeric(input)
    }
    fn reduce_median<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(self.execute(input, None))
    }
    fn reduce_median_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        axis: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(self.execute(input, Some(axis)))
    }
}

#[path = "median_provider_contract/preflight.rs"]
mod preflight;
fn input(provider: &Provider) -> GpuTensorHandle {
    provider
        .inner
        .upload(&HostTensorView {
            data: &[1., 3., 2., 8.],
            shape: &[2, 2],
        })
        .unwrap()
        .with_provenance(GpuHandleProvenance::Automatic)
        .unwrap()
}
fn axes() -> Value {
    Value::Tensor(Tensor::new(vec![1., 2.], vec![1, 2]).unwrap())
}
fn assert_original_error(error: &runmat_runtime::RuntimeError) {
    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
    let carrier = error
        .source
        .as_ref()
        .unwrap()
        .downcast_ref::<ProviderExecutionError>()
        .expect("typed execution failure");
    let mut current: &(dyn std::error::Error + 'static) = carrier;
    loop {
        if let Some(error) = current.downcast_ref::<std::io::Error>() {
            assert_eq!(error.kind(), std::io::ErrorKind::BrokenPipe);
            break;
        }
        current = current
            .source()
            .expect("retained original operational error");
    }
}

#[test]
fn median_only_typed_unsupported_falls_back_without_replaying_operational_errors() {
    for mode in [Mode::Success, Mode::Unsupported, Mode::Failure] {
        for all in [false, true] {
            let provider = Provider::new(mode);
            let _active = ThreadProviderGuard::set(Some(provider));
            let input = input(provider);
            let weak = input.allocation_lease().downgrade();
            let mut args = vec![Value::GpuTensor(input.clone())];
            if all {
                args.push(Value::String("all".into()));
            }
            let result = block_on(call_builtin_async_with_outputs("median", &args, 1));
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            match mode {
                Mode::Failure => {
                    assert_original_error(&result.unwrap_err());
                    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
                    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
                }
                _ => {
                    let result = result.unwrap();
                    let Value::GpuTensor(output) = result.value() else {
                        panic!("resident result")
                    };
                    assert!(!output.shares_allocation_with(&input));
                    let actual = block_on(provider.inner.download(output)).unwrap();
                    assert_eq!(actual.data, if all { vec![2.5] } else { vec![2., 5.] });
                    assert_eq!(
                        provider.downloads.load(Ordering::SeqCst),
                        usize::from(matches!(mode, Mode::Unsupported))
                    );
                    let output_weak = output.allocation_lease().downgrade();
                    drop(result);
                    assert!(output_weak.acquire().is_none());
                }
            }
            assert!(weak.acquire().is_some());
            drop(args);
            drop(input);
            assert!(weak.acquire().is_none());
        }
    }
}

#[test]
fn median_rejects_provider_metadata_and_original_aliases_without_gather() {
    for mode in [
        Mode::Shape,
        Mode::Class,
        Mode::Complex,
        Mode::Integer,
        Mode::Logical,
        Mode::Owner,
        Mode::Alias,
    ] {
        let provider = Provider::new(mode);
        let _active = ThreadProviderGuard::set(Some(provider));
        let input = input(provider);
        *provider.alias.lock().unwrap() = Some(input.downgrade());
        let weak = input.allocation_lease().downgrade();
        let error = block_on(call_builtin_async_with_outputs(
            "median",
            &[Value::GpuTensor(input.clone())],
            1,
        ))
        .unwrap_err();
        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never, "{mode:?}");
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        assert!(provider
            .produced
            .lock()
            .unwrap()
            .iter()
            .all(|weak| weak.acquire().is_none()));
        assert!(weak.acquire().is_some());
        drop(error);
        drop(input);
        assert!(weak.acquire().is_none());
    }
}

#[test]
fn median_intermediates_retire_on_later_failure_unsupported_alias_and_cancellation() {
    use std::{
        future::Future,
        task::{Context, Poll},
    };
    for mode in [
        Mode::LaterFailure,
        Mode::LaterUnsupported,
        Mode::LaterAlias,
        Mode::LaterPending,
    ] {
        let provider = Provider::new(mode);
        let _active = ThreadProviderGuard::set(Some(provider));
        let input = input(provider);
        *provider.alias.lock().unwrap() = Some(input.downgrade());
        let weak = input.allocation_lease().downgrade();
        let args = vec![Value::GpuTensor(input.clone()), axes()];
        drop(call_builtin_async_with_outputs("median", &args, 1));
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        if matches!(mode, Mode::LaterPending) {
            let mut future = Box::pin(call_builtin_async_with_outputs("median", &args, 1));
            let waker = futures::task::noop_waker();
            assert!(matches!(
                future.as_mut().poll(&mut Context::from_waker(&waker)),
                Poll::Pending
            ));
            assert!(provider.produced.lock().unwrap()[0].acquire().is_some());
            drop(future);
        } else {
            let result = block_on(call_builtin_async_with_outputs("median", &args, 1));
            match mode {
                Mode::LaterUnsupported => {
                    assert!(matches!(result.unwrap().value(), Value::GpuTensor(_)))
                }
                Mode::LaterFailure => assert_original_error(&result.unwrap_err()),
                Mode::LaterAlias => {
                    assert_eq!(result.unwrap_err().gpu_gather_retry, GpuGatherRetry::Never)
                }
                _ => unreachable!(),
            }
        }
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            provider.downloads.load(Ordering::SeqCst),
            usize::from(matches!(mode, Mode::LaterUnsupported))
        );
        assert!(provider
            .produced
            .lock()
            .unwrap()
            .iter()
            .all(|weak| weak.acquire().is_none()));
        assert!(weak.acquire().is_some());
        drop(args);
        drop(input);
        assert!(weak.acquire().is_none());
    }
}

#[test]
fn median_native_single_double_and_all_match_actual_host_oracles() {
    for single in [false, true] {
        let provider = Provider::new(Mode::Success);
        let _active = ThreadProviderGuard::set(Some(provider));
        // Column-major [1 2 100; 3 4 5]: global median 3.5, but the old
        // sequential median-of-column-medians incorrectly returned 3.
        let tensor = if single {
            Tensor::from_f32(vec![1., 3., 2., 4., 100., 5.], vec![2, 3]).unwrap()
        } else {
            Tensor::new(vec![1., 3., 2., 4., 100., 5.], vec![2, 3]).unwrap()
        };
        let source = if single {
            provider
                .inner
                .upload_numeric_raw(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&[1., 3., 2., 4., 100., 5.]),
                    shape: &[2, 3],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap()
        } else {
            provider
                .inner
                .upload(&HostTensorView {
                    data: &[1., 3., 2., 4., 100., 5.],
                    shape: &[2, 3],
                })
                .unwrap()
        }
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
        for (selector, expected, shape) in [
            (Value::String("all".into()), vec![3.5], vec![1, 1]),
            (Value::Num(1.), vec![2., 3., 52.5], vec![1, 3]),
            (Value::Num(2.), vec![2., 4.], vec![2, 1]),
        ] {
            let host = block_on(call_builtin_async_with_outputs(
                "median",
                &[Value::Tensor(tensor.clone()), selector.clone()],
                1,
            ))
            .unwrap();
            let host_values = match host.value() {
                Value::Num(value) => vec![*value],
                Value::Tensor(tensor) => tensor.materialize_f64(),
                other => panic!("host result: {other:?}"),
            };
            assert_eq!(host_values, expected);
            let result = block_on(call_builtin_async_with_outputs(
                "median",
                &[Value::GpuTensor(source.clone()), selector],
                1,
            ))
            .unwrap();
            let Value::GpuTensor(output) = result.value() else {
                panic!("resident native result")
            };
            assert_eq!(output.shape(), shape);
            assert_eq!(
                output.physical().element_type(),
                if single {
                    NumericElementType::F32
                } else {
                    NumericElementType::F64
                }
            );
            assert!(handle_is_explicit(output));
            let actual = block_on(provider.inner.download_numeric(output)).unwrap();
            match actual.data {
                HostNumericDataOwned::F32(values) => assert_eq!(
                    values.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                    expected
                ),
                HostNumericDataOwned::F64(values) => assert_eq!(values, expected),
                other => panic!("native storage changed: {other:?}"),
            }
        }
        assert_eq!(provider.calls.load(Ordering::SeqCst), 3);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn median_rank_boundary_is_typed_unsupported_after_owner_authentication() {
    let provider = InProcessProvider::new();
    let foreign = InProcessProvider::new();
    let owned = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1., 3.]),
            shape: &[2, 1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let error = block_on(provider.reduce_median_dim(&owned, 0)).unwrap_err();
    assert!(is_unsupported_provider_operation(&error));
    let foreign_error = block_on(foreign.reduce_median_dim(&owned, 0)).unwrap_err();
    assert!(!is_unsupported_provider_operation(&foreign_error));
    assert!(owned.allocation_lease().validate_live().is_ok());
}

#[test]
fn median_all_preserves_nan_empty_and_exact_integer_host_rules() {
    let mut with_nan = vec![1., 3., 2., 4., 100., 5., f64::NAN, 7.];
    let value = Value::Tensor(Tensor::new(with_nan.clone(), vec![2, 4]).unwrap());
    let included = block_on(call_builtin_async_with_outputs(
        "median",
        &[value.clone(), Value::String("all".into())],
        1,
    ))
    .unwrap();
    assert!(matches!(included.value(), Value::Num(value) if value.is_nan()));
    let omitted = block_on(call_builtin_async_with_outputs(
        "median",
        &[
            value,
            Value::String("all".into()),
            Value::String("omitnan".into()),
        ],
        1,
    ))
    .unwrap();
    with_nan.retain(|value| !value.is_nan());
    with_nan.sort_by(f64::total_cmp);
    assert!(matches!(omitted.value(), Value::Num(value) if *value == with_nan[3]));
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let exact = Value::Tensor(
        Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![u64::MAX - 5, u64::MAX - 1, u64::MAX - 3]),
            vec![1, 3],
        )
        .unwrap(),
    );
    let result = block_on(call_builtin_async_with_outputs(
        "median",
        &[exact, Value::String("all".into())],
        1,
    ))
    .unwrap();
    match result.value() {
        Value::Int(runmat_value::IntValue::U64(value)) => assert_eq!(*value, u64::MAX - 3),
        Value::Tensor(tensor) => assert!(
            matches!(tensor.integer_storage(), Some(runmat_value::IntegerStorage::U64(values)) if values == &vec![u64::MAX-3])
        ),
        other => panic!("exact integer result: {other:?}"),
    }
    let empty = Value::Tensor(Tensor::from_f32(vec![], vec![0, 3]).unwrap());
    let result = block_on(call_builtin_async_with_outputs(
        "median",
        &[empty, Value::String("all".into())],
        1,
    ))
    .unwrap();
    assert!(matches!(result.value(), Value::Num(value) if value.is_nan()));
}
