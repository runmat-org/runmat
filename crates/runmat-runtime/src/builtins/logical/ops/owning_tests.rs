use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{AccelDownloadFuture, AccelProviderFuture, HostTensorView};
use std::error::Error as _;
use std::sync::atomic::{AtomicUsize, Ordering};

struct FailedProbe {
    inner: InProcessProvider,
    downloads: AtomicUsize,
}

impl AccelProvider for FailedProbe {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "failed-logical-probe".into()
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::Relaxed);
        self.inner.download(handle)
    }
    fn logical_isnan(&self, _: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(std::io::Error::other("injected NaN probe execution failure").into())
    }
}

#[test]
fn logical_probe_failure_is_not_gathered_and_preserves_source_and_input() {
    let _state = test_support::accel_test_lock();
    let provider = Box::leak(Box::new(FailedProbe {
        inner: InProcessProvider::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let input = provider
        .upload(&HostTensorView {
            data: &[0.0, 2.0],
            shape: &[1, 2],
        })
        .unwrap();
    let error = block_on(logical_from_gpu(input.clone())).unwrap_err();
    assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
    let mut source: &(dyn std::error::Error + 'static) = &error;
    while source.downcast_ref::<std::io::Error>().is_none() {
        source = source
            .source()
            .expect("original operational source must survive");
    }
    assert_eq!(
        block_on(provider.inner.download(&input)).unwrap().data,
        vec![0.0, 2.0]
    );
}

#[test]
fn logical_single_and_empty_fallback_publish_real_logical_producers() {
    test_support::with_test_provider(|provider| {
        for (values, shape) in [(vec![0.0_f32, -3.0], vec![1, 2]), (vec![], vec![0, 3])] {
            let input = provider
                .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                    data: runmat_accelerate_api::HostNumericDataView::F32(&values),
                    shape: &shape,
                    storage: runmat_accelerate_api::GpuTensorStorage::Real,
                })
                .unwrap()
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .unwrap();
            let Value::GpuTensor(output) = block_on(logical_from_gpu(input.clone())).unwrap()
            else {
                panic!("explicit residency")
            };
            assert!(output.view().is_logical());
            assert!(!output.shares_allocation_with(&input));
            assert_eq!(output.shape(), shape);
            assert_eq!(
                block_on(provider.download(&output)).unwrap().data,
                values
                    .iter()
                    .map(|value| if *value == 0.0 { 0.0 } else { 1.0 })
                    .collect::<Vec<_>>()
            );
        }
    });
}

#[test]
fn logical_nan_probe_uses_logical_reduction_for_f32_and_rejects_nan() {
    test_support::with_test_provider(|provider| {
        let input = provider
            .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                data: runmat_accelerate_api::HostNumericDataView::F32(&[0.0, f32::NAN]),
                shape: &[1, 2],
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
            })
            .unwrap();
        assert_eq!(
            block_on(provider_input_contains_nan(provider, &input)).unwrap(),
            Some(true)
        );
        let error = block_on(logical_from_gpu(input)).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:logical:ConversionNotPossible")
        );
    });
}

#[test]
fn logical_nan_probe_preserves_three_dimensional_residency_and_nan_rejection() {
    test_support::with_test_provider(|provider| {
        for (values, contains_nan) in [
            ([0.0_f32, 1.0, 2.0, 3.0], false),
            ([0.0_f32, 1.0, f32::NAN, 3.0], true),
        ] {
            let input = provider
                .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                    data: runmat_accelerate_api::HostNumericDataView::F32(&values),
                    shape: &[2, 1, 2],
                    storage: runmat_accelerate_api::GpuTensorStorage::Real,
                })
                .unwrap()
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .unwrap();
            assert_eq!(
                block_on(provider_input_contains_nan(provider, &input)).unwrap(),
                Some(contains_nan)
            );
            match block_on(logical_from_gpu(input.clone())) {
                Err(error) if contains_nan => assert_eq!(
                    error.identifier(),
                    Some("RunMat:logical:ConversionNotPossible")
                ),
                Ok(Value::GpuTensor(output)) if !contains_nan => {
                    assert_eq!(output.shape(), [2, 1, 2]);
                    assert!(output.view().is_logical());
                    assert!(!output.shares_allocation_with(&input));
                }
                other => panic!("unexpected three-dimensional logical result: {other:?}"),
            }
        }
    });
}

struct BadReduction {
    inner: InProcessProvider,
    fail_reduction: bool,
    reductions: AtomicUsize,
    downloads: AtomicUsize,
}

impl AccelProvider for BadReduction {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "bad-logical-reduction".into()
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::Relaxed);
        self.inner.download(handle)
    }
    fn logical_isnan(&self, input: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        self.inner.logical_isnan(input)
    }
    fn reduce_any<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        assert!(!omit_nan);
        self.reductions.fetch_add(1, Ordering::Relaxed);
        Box::pin(async move {
            if self.fail_reduction {
                return Err(std::io::Error::other("injected logical reduction failure").into());
            }
            self.inner.upload(&HostTensorView {
                data: &[1.0],
                shape: &[1, 1],
            })
        })
    }
}

#[test]
fn logical_nan_probe_rejects_nonlogical_reduction_without_readback() {
    let _state = test_support::accel_test_lock();
    let provider = Box::leak(Box::new(BadReduction {
        inner: InProcessProvider::new(),
        fail_reduction: false,
        reductions: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
    }));
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let input = provider
        .upload(&HostTensorView {
            data: &[0.0, 2.0],
            shape: &[1, 2],
        })
        .unwrap();
    let error = block_on(provider_input_contains_nan(provider, &input)).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:logical:InternalError"));
    assert_eq!(provider.reductions.load(Ordering::Relaxed), 1);
    assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
}

#[test]
fn logical_nan_reduction_failure_preserves_cause_and_never_retries() {
    let _state = test_support::accel_test_lock();
    let provider = Box::leak(Box::new(BadReduction {
        inner: InProcessProvider::new(),
        fail_reduction: true,
        reductions: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
    }));
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let input = provider
        .upload(&HostTensorView {
            data: &[0.0, 2.0],
            shape: &[1, 2],
        })
        .unwrap();
    let error = block_on(logical_from_gpu(input)).unwrap_err();
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(provider.reductions.load(Ordering::Relaxed), 1);
    assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
    let cause = error
        .source()
        .unwrap()
        .downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
        .expect("provider execution cause survives");
    assert!(cause.is_after_effect());
    assert!(cause
        .source()
        .unwrap()
        .downcast_ref::<std::io::Error>()
        .is_some());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn canonical_wrapper_roots_managed_input_before_first_poll_and_retires_it_on_drop() {
    runmat_gc::gc_test_context(|| {
        let initial = runmat_gc::gc_allocate_rooted(Value::String("input".into()))
            .expect("allocate input target");
        let handle = initial.handle();
        let argument = Value::HandleObject(runmat_value::HandleRef {
            class_name: "LogicalInputFixture".into(),
            target: handle,
            valid: true,
        });
        let future =
            super::__rt_wrap_logical_builtin(crate::builtin::RuntimeBuiltinInvocation::new(
                std::slice::from_ref(&argument),
                None,
                Some(1),
            ));
        initial.unroot().expect("transfer allocation root");
        runmat_gc::gc_collect_major().expect("collect before first poll");
        assert_eq!(
            runmat_gc::gc_clone_value(&handle).expect("wrapper owns input"),
            Value::String("input".into())
        );
        drop(future);
        runmat_gc::gc_collect_major().expect("collect after never-polled future drop");
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn canonical_wrapper_retires_managed_input_after_conversion_failure() {
    runmat_gc::gc_test_context(|| {
        let initial = runmat_gc::gc_allocate_rooted(Value::String("input".into()))
            .expect("allocate input target");
        let handle = initial.handle();
        let argument = Value::HandleObject(runmat_value::HandleRef {
            class_name: "LogicalInputFixture".into(),
            target: handle,
            valid: true,
        });
        let future =
            super::__rt_wrap_logical_builtin(crate::builtin::RuntimeBuiltinInvocation::new(
                std::slice::from_ref(&argument),
                None,
                Some(1),
            ));
        initial.unroot().expect("transfer allocation root");
        let error = block_on(future).expect_err("handle input is not logical");
        assert_eq!(
            error.identifier(),
            Some("RunMat:logical:ConversionNotPossible")
        );
        runmat_gc::gc_collect_major().expect("collect after error");
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}
