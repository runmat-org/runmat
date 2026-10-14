//! Real native allocations through public dispatch, independent of provider preference.
use runmat_accelerate_api::allocation::WeakAllocationLease;
use runmat_accelerate_api::*;
use runmat_value::{Tensor, Value};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Mutex,
};

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

struct Provider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    hooks: AtomicUsize,
    downloads: AtomicUsize,
    uploads: AtomicUsize,
    kernel: Mutex<Option<(NumericElementType, WeakAllocationLease)>>,
    pending: AtomicBool,
}
impl Provider {
    fn new() -> &'static Self {
        Box::leak(Box::new(Self {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            hooks: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
            uploads: AtomicUsize::new(0),
            kernel: Mutex::new(None),
            pending: AtomicBool::new(false),
        }))
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "native imfilter reference owner".into()
    }
    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("typed upload required")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("typed download required")
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        self.inner.upload_numeric_raw(host)
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download_numeric(handle)
    }
    fn imfilter<'a>(
        &'a self,
        image: &'a GpuTensorHandle,
        kernel: &'a GpuTensorHandle,
        options: &'a ImfilterOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.hooks.fetch_add(1, Ordering::SeqCst);
        *self.kernel.lock().unwrap() = Some((
            kernel.physical().element_type(),
            kernel.allocation_lease().downgrade(),
        ));
        if self.pending.load(Ordering::SeqCst) {
            Box::pin(std::future::pending())
        } else {
            self.inner.imfilter(image, kernel, options)
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn cancelling_single_filter_retires_prepared_kernel_not_original_double_h() {
    use std::future::Future;
    let provider = Provider::new();
    let _selected = ThreadProviderGuard::set(Some(provider));
    provider.pending.store(true, Ordering::SeqCst);
    let image = provider
        .inner
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 2.0, 3.0]),
            shape: &[3, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap()
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    let kernel = provider
        .inner
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[1.0, 1.0, 1.0]),
            shape: &[3, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let arguments = [
        Value::GpuTensor(image.clone()),
        Value::GpuTensor(kernel.clone()),
    ];
    let mut call = Box::pin(runmat_runtime::call_builtin_async("imfilter", &arguments));
    assert!(call
        .as_mut()
        .poll(&mut std::task::Context::from_waker(
            futures::task::noop_waker_ref()
        ))
        .is_pending());
    assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    assert!(provider
        .kernel
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .1
        .acquire()
        .is_some());
    drop(call);
    let prepared = provider.kernel.lock().unwrap();
    let (element, weak) = prepared.as_ref().unwrap();
    assert_eq!(*element, NumericElementType::F32);
    assert!(weak.acquire().is_none());
    image.allocation_lease().validate_live().unwrap();
    kernel.allocation_lease().validate_live().unwrap();
    assert_eq!(kernel.physical().element_type(), NumericElementType::F64);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn image_native_class_and_accumulation_do_not_follow_provider_preference() {
    for single in [false, true] {
        for explicit in [false, true] {
            for resident_kernel in [false, true] {
                let provider = Provider::new();
                let _selected = ThreadProviderGuard::set(Some(provider));
                assert_eq!(provider.precision(), ProviderPrecision::F64);
                let image_values = [16_777_216.0, 1.0, -16_777_216.0];
                let single_values = image_values.map(|value| value as f32);
                let image = provider
                    .inner
                    .upload_numeric_raw(&HostNumericTensorView {
                        data: if single {
                            HostNumericDataView::F32(&single_values)
                        } else {
                            HostNumericDataView::F64(&image_values)
                        },
                        shape: &[3, 1],
                        storage: GpuTensorStorage::Real,
                    })
                    .unwrap()
                    .with_provenance(if explicit {
                        GpuHandleProvenance::Explicit
                    } else {
                        GpuHandleProvenance::Automatic
                    })
                    .unwrap();
                let weights = vec![1.0 + 2.0_f64.powi(-24), 1.0, 1.0];
                let kernel_owner = resident_kernel.then(|| {
                    provider
                        .inner
                        .upload_numeric_raw(&HostNumericTensorView {
                            data: HostNumericDataView::F64(&weights),
                            shape: &[3, 1],
                            storage: GpuTensorStorage::Real,
                        })
                        .unwrap()
                });
                let kernel_value = match kernel_owner.as_ref() {
                    Some(handle) => Value::GpuTensor(handle.clone()),
                    None => Value::Tensor(Tensor::new(weights, vec![3, 1]).unwrap()),
                };
                let output = futures::executor::block_on(runmat_runtime::call_builtin_async(
                    "imfilter",
                    &[Value::GpuTensor(image.clone()), kernel_value],
                ))
                .unwrap();
                let Value::GpuTensor(output) = output.value() else {
                    panic!("supported native route must execute")
                };
                assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
                assert_eq!(
                    provider.downloads.load(Ordering::SeqCst),
                    usize::from(resident_kernel)
                );
                assert_eq!(
                    output.physical().element_type(),
                    if single {
                        NumericElementType::F32
                    } else {
                        NumericElementType::F64
                    }
                );
                assert_eq!(handle_provenance(output), handle_provenance(&image));
                assert_eq!(output.shape(), &[3, 1]);
                let expected_kernel = if single && explicit {
                    NumericElementType::F32
                } else {
                    NumericElementType::F64
                };
                let guard = provider.kernel.lock().unwrap();
                let (actual_kernel, weak) = guard.as_ref().unwrap();
                assert_eq!(*actual_kernel, expected_kernel);
                let reused = resident_kernel && expected_kernel == NumericElementType::F64;
                assert_eq!(
                    weak.acquire().is_some(),
                    reused,
                    "only original H may remain owned"
                );
                drop(guard);
                assert_eq!(
                    provider.uploads.load(Ordering::SeqCst),
                    usize::from(!reused)
                );
                let host =
                    futures::executor::block_on(provider.inner.download_numeric(output)).unwrap();
                let values = match host.data {
                    HostNumericDataOwned::F32(values) => {
                        values.into_iter().map(f64::from).collect::<Vec<_>>()
                    }
                    HostNumericDataOwned::F64(values) => values,
                    _ => panic!("native floating result"),
                };
                assert_eq!(values[1], if single && explicit { 0.0 } else { 2.0 });
                image.allocation_lease().validate_live().unwrap();
                if let Some(kernel) = kernel_owner {
                    kernel.allocation_lease().validate_live().unwrap();
                    assert!(!output.shares_allocation_with(&kernel));
                }
                assert!(!output.shares_allocation_with(&image));
            }
        }
    }
}
