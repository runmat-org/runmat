//! Actual domain dispatch keeps terminal, typed provider causes.
#![cfg(not(target_arch = "wasm32"))]
use futures::executor::block_on;
use runmat_accelerate_api::test_support::TestAllocationOwner;
use runmat_accelerate_api::*;
use runmat_runtime::{call_builtin_async_with_outputs, GpuGatherRetry};
use runmat_value::Value;
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "provider_execution_sources/extrema_continuations.rs"]
mod extrema_continuations;

#[path = "provider_execution_sources/math_boundaries.rs"]
mod math_boundaries;

#[derive(Debug)]
struct DeviceFailure;
impl std::fmt::Display for DeviceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("injected transport failure")
    }
}
impl std::error::Error for DeviceFailure {}

struct Provider {
    owner: TestAllocationOwner,
    calls: AtomicUsize,
    nested: bool,
}
impl Provider {
    fn fail<T>(&self) -> anyhow::Result<T> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let cause = if self.nested {
            anyhow::Error::new(
                runmat_runtime::build_runtime_error("authoritative provider contract")
                    .with_identifier("RunMat:test:ProviderContract")
                    .with_builtin("provider-origin")
                    .with_source(DeviceFailure)
                    .build(),
            )
        } else {
            anyhow::Error::new(DeviceFailure)
        };
        Err(cause)
    }
}
macro_rules! dimension_hook {
    ($name:ident) => {
        fn $name<'a>(
            &'a self,
            _: &'a GpuTensorHandle,
            _: usize,
        ) -> AccelProviderFuture<'a, GpuTensorHandle> {
            Box::pin(async move { self.fail() })
        }
    };
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "terminal provider source fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("failure may not retry upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("failure may not gather")
    }
    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move { self.fail() })
    }
    fn reduce_sum_dim<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: usize,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fail().map_err(ProviderExecutionError::from) })
    }
    fn reduce_mean_dim<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: usize,
    ) -> ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async move {
            self.fail()
                .map_err(|source| ProviderExecutionError::from(source).with_unproven_effects())
        })
    }
    dimension_hook!(reduce_integer_mean_native_dim);
    fn reduce_min_dim<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ReduceDimResult> {
        Box::pin(async move { self.fail().map_err(Into::into) })
    }
    fn reduce_max_dim<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ReduceDimResult> {
        Box::pin(async move { self.fail().map_err(Into::into) })
    }
    fn elem_min<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fail().map_err(Into::into) })
    }
    fn elem_max<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fail().map_err(Into::into) })
    }
    fn scalar_min(&self, _: &GpuTensorHandle, _: f64) -> anyhow::Result<GpuTensorHandle> {
        self.fail()
    }
    fn scalar_max(&self, _: &GpuTensorHandle, _: f64) -> anyhow::Result<GpuTensorHandle> {
        self.fail()
    }
    fn fill_like(
        &self,
        _: &GpuTensorHandle,
        _: f64,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let result: anyhow::Result<GpuTensorHandle> = self.fail();
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
    }
    fn unary_cos<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fail() })
    }
    fn moving_window<'a>(
        &'a self,
        _: &'a ProviderMovingWindowRequest<'a>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fail() })
    }
    fn gradient_dim(
        &self,
        _: &GpuTensorHandle,
        _: usize,
        _: f64,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let result: anyhow::Result<_> = self.fail();
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
    }
    fn diff_dim(
        &self,
        _: &GpuTensorHandle,
        _: usize,
        _: usize,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let result: anyhow::Result<_> = self.fail();
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
    }
    fn cumtrapz_dim(
        &self,
        _: &GpuTensorHandle,
        _: usize,
        _: ProviderTrapezoidSpacing<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let result: anyhow::Result<_> = self.fail();
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
    }
}

fn extrema_args(form: usize, left: &GpuTensorHandle, right: &GpuTensorHandle) -> Vec<Value> {
    match form {
        0 => vec![Value::GpuTensor(left.clone())],
        1 => vec![
            Value::GpuTensor(left.clone()),
            Value::GpuTensor(right.clone()),
        ],
        2 => vec![Value::GpuTensor(left.clone()), Value::Num(2.0)],
        3 => vec![Value::Num(2.0), Value::GpuTensor(right.clone())],
        _ => unreachable!(),
    }
}

#[test]
fn public_extrema_dispatch_keeps_terminal_typed_failures_without_second_execution() {
    for nested in [false, true] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            calls: AtomicUsize::new(0),
            nested,
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        for name in ["min", "max"] {
            for form in 0..4 {
                let left = provider
                    .owner
                    .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
                    .unwrap();
                let right = provider
                    .owner
                    .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
                    .unwrap();
                let args = extrema_args(form, &left.handle, &right.handle);
                let before = provider.calls.load(Ordering::SeqCst);
                let error = block_on(runmat_runtime::call_builtin_async(name, &args)).unwrap_err();
                assert_eq!(
                    provider.calls.load(Ordering::SeqCst),
                    before + 1,
                    "{name} form {form}"
                );
                assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
                let mut source: &(dyn std::error::Error + 'static) = &error;
                while !source.is::<DeviceFailure>() {
                    source = source.source().expect("typed extrema cause retained");
                }
                if nested {
                    assert_eq!(error.identifier(), Some("RunMat:test:ProviderContract"));
                    assert_eq!(error.context.builtin.as_deref(), Some("provider-origin"));
                    assert_eq!(error.message(), "authoritative provider contract");
                }
                assert_eq!(left.probe.retirements(), 0);
                assert_eq!(right.probe.retirements(), 0);
                left.handle.allocation_lease().validate_live().unwrap();
                right.handle.allocation_lease().validate_live().unwrap();
                drop(error);
                drop(args);
                left.handle.release().unwrap();
                right.handle.release().unwrap();
                assert_eq!(left.probe.retirements(), 1);
                assert_eq!(right.probe.retirements(), 1);
            }
        }
    }
}

struct UnsupportedExtrema {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    calls: AtomicUsize,
    downloads: AtomicUsize,
}
impl UnsupportedExtrema {
    fn reject<T>(&self) -> anyhow::Result<T> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(unsupported_provider_operation(
            "fixture extrema capability unavailable",
        ))
    }
}
impl AccelProvider for UnsupportedExtrema {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "public unsupported extrema fixture".into()
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(handle)
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download_numeric(handle)
    }
    fn reduce_min_dim<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ReduceDimResult> {
        Box::pin(async move { self.reject().map_err(Into::into) })
    }
    fn reduce_max_dim<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ReduceDimResult> {
        Box::pin(async move { self.reject().map_err(Into::into) })
    }
    fn elem_min<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reject().map_err(Into::into) })
    }
    fn elem_max<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reject().map_err(Into::into) })
    }
    fn scalar_min(&self, _: &GpuTensorHandle, _: f64) -> anyhow::Result<GpuTensorHandle> {
        self.reject()
    }
    fn scalar_max(&self, _: &GpuTensorHandle, _: f64) -> anyhow::Result<GpuTensorHandle> {
        self.reject()
    }
    fn fill_like(
        &self,
        _: &GpuTensorHandle,
        _: f64,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(None)
    }
}

#[test]
fn public_extrema_dispatch_gathers_only_after_typed_unsupported() {
    let provider = Box::leak(Box::new(UnsupportedExtrema {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        calls: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    for name in ["min", "max"] {
        for form in 0..4 {
            let left = provider
                .upload(&HostTensorView {
                    data: &[3.0, -1.0],
                    shape: &[2, 1],
                })
                .unwrap();
            let right = provider
                .upload(&HostTensorView {
                    data: &[4.0, -2.0],
                    shape: &[2, 1],
                })
                .unwrap();
            let args = extrema_args(form, &left, &right);
            let calls = provider.calls.load(Ordering::SeqCst);
            let downloads = provider.downloads.load(Ordering::SeqCst);
            let result = block_on(runmat_runtime::call_builtin_async(name, &args)).unwrap();
            // Scalar max first tries fill_like, then scalar_max when the
            // provider explicitly rejects the fill capability. Operational
            // failures still stop at the first hook (covered separately).
            let expected_calls = if name == "max" && form >= 2 { 2 } else { 1 };
            assert_eq!(
                provider.calls.load(Ordering::SeqCst),
                calls + expected_calls,
                "{name} form {form}"
            );
            assert_eq!(
                provider.downloads.load(Ordering::SeqCst),
                downloads + if form == 1 { 2 } else { 1 }
            );
            if form == 0 {
                assert_eq!(result, Value::Num(if name == "min" { -1.0 } else { 3.0 }));
            } else {
                let expected = match (name, form) {
                    ("min", 1) => vec![3.0, -2.0],
                    ("max", 1) => vec![4.0, -1.0],
                    ("min", 2) => vec![2.0, -1.0],
                    ("max", 2) => vec![3.0, 2.0],
                    ("min", 3) => vec![2.0, -2.0],
                    ("max", 3) => vec![4.0, 2.0],
                    _ => unreachable!(),
                };
                let Value::Tensor(output) = result.value() else {
                    panic!("capability fallback returns host tensor")
                };
                assert_eq!(output.shape(), &[2, 1]);
                assert_eq!(output.materialize_f64(), expected);
            }
            drop(args);
            for input in [left, right] {
                input.allocation_lease().validate_live().unwrap();
                let physical = input.allocation_lease().downgrade();
                input.release().unwrap();
                assert!(physical.acquire().is_none());
            }
        }
    }
}

#[test]
fn native_download_boundaries_preserve_typed_failures_and_borrowed_inputs() {
    use runmat_runtime::builtins::common::gpu_helpers::{
        download_floating_projection_async, download_value_preserving_residency_async,
    };
    for nested in [false, true] {
        let provider = Provider {
            owner: TestAllocationOwner::new(),
            calls: AtomicUsize::new(0),
            nested,
        };
        let input = provider
            .owner
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let original = input.handle.wire_descriptor();
        let errors = [
            block_on(download_value_preserving_residency_async(
                &provider,
                &input.handle,
            ))
            .unwrap_err(),
            block_on(download_floating_projection_async(&provider, &input.handle)).unwrap_err(),
        ];
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        for error in errors {
            assert_eq!(error.gpu_gather_retry(), GpuGatherRetry::Never);
            let mut cursor: &(dyn std::error::Error + 'static) = &error;
            while !cursor.is::<DeviceFailure>() {
                cursor = cursor.source().expect("typed download cause preserved");
            }
            if nested {
                assert_eq!(error.identifier(), Some("RunMat:test:ProviderContract"));
                assert_eq!(error.context.builtin.as_deref(), Some("provider-origin"));
                assert_eq!(error.message(), "authoritative provider contract");
            } else {
                assert_eq!(error.identifier(), Some("RunMat:gpu:DownloadFailed"));
            }
            assert_eq!(input.handle.wire_descriptor(), original);
            assert_eq!(input.probe.retirements(), 0);
        }
        drop(input.handle);
        assert_eq!(input.probe.retirements(), 1);
    }
}

#[test]
fn actual_provider_domains_retain_typed_causes_without_retry() {
    runmat_gc::gc_test_context(|| {
        for nested in [false, true] {
            let provider = Box::leak(Box::new(Provider {
                owner: TestAllocationOwner::new(),
                calls: AtomicUsize::new(0),
                nested,
            }));
            let _provider = ThreadProviderGuard::set(Some(provider));
            for (name, integer) in [
                ("movsum", false),
                ("gradient", false),
                ("cumtrapz", false),
                ("diff", false),
                ("sum", false),
                ("mean", false),
                ("mean", true),
                ("pol2cart", false),
            ] {
                let input = provider
                    .owner
                    .allocate_zeroed(
                        vec![2, 2],
                        if integer {
                            NumericElementType::I64
                        } else {
                            NumericElementType::F64
                        },
                        GpuTensorStorage::Real,
                    )
                    .unwrap();
                let handle = input
                    .handle
                    .with_provenance(GpuHandleProvenance::Automatic)
                    .unwrap();
                let mut args = vec![Value::GpuTensor(handle)];
                match name {
                    "pol2cart" => args.push(args[0].clone()),
                    "movsum" => args.push(Value::Num(2.0)),
                    "mean" | "sum" if integer => {
                        args.push(Value::Num(1.0));
                        args.push(Value::from("native"));
                    }
                    "mean" | "sum" => args.push(Value::Num(1.0)),
                    _ => {}
                }
                let before = provider.calls.load(Ordering::SeqCst);
                let error = block_on(call_builtin_async_with_outputs(name, &args, 1)).unwrap_err();
                assert_eq!(
                    provider.calls.load(Ordering::SeqCst),
                    before + 1,
                    "{name}: actual selected hook"
                );
                assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never, "{name}");
                let mut cursor: &(dyn std::error::Error + 'static) = &error;
                loop {
                    if cursor.is::<DeviceFailure>() {
                        break;
                    }
                    cursor = cursor
                        .source()
                        .unwrap_or_else(|| panic!("{name}: typed cause missing: {error:?}"));
                }
                if nested {
                    assert_eq!(error.identifier(), Some("RunMat:test:ProviderContract"));
                    assert_eq!(error.message(), "authoritative provider contract");
                    assert_eq!(error.context.builtin.as_deref(), Some("provider-origin"));
                }
                assert_eq!(input.probe.retirements(), 0);
                drop(error);
                drop(args);
                drop(input.handle);
                assert_eq!(input.probe.retirements(), 1);
            }
        }
    });
}
