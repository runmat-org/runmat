//! Actual public dispatch with real leases; no duplicate builtin registration.
use futures::{task::noop_waker_ref, FutureExt};
use runmat_accelerate_api::allocation::WeakGpuTensorHandle;
use runmat_accelerate_api::*;
use runmat_value::{Tensor, Value};
use std::task::{Context, Poll};
use std::{
    error::Error,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
};

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[derive(Clone, Copy)]
enum Mode {
    Unsupported,
    Operational,
    UploadOperational,
    UploadUnsupported,
    UploadAlias,
    RestoreAlias,
    ResultAlias,
    Pending,
}
#[derive(Debug)]
struct Operational;
impl std::fmt::Display for Operational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("imfilter not supported by provider")
    }
}
impl Error for Operational {}

struct Provider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    mode: Mode,
    hooks: AtomicUsize,
    uploads: AtomicUsize,
    downloads: AtomicUsize,
    alias: Mutex<Option<GpuTensorHandle>>,
    scratch: Mutex<Option<WeakGpuTensorHandle>>,
}
impl Provider {
    fn new(mode: Mode) -> &'static Self {
        Box::leak(Box::new(Self {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            mode,
            hooks: AtomicUsize::new(0),
            uploads: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
            alias: Mutex::new(None),
            scratch: Mutex::new(None),
        }))
    }
    fn input(&self, data: &[f64], shape: &[usize], explicit: bool) -> GpuTensorHandle {
        self.inner
            .upload(&HostTensorView { data, shape })
            .unwrap()
            .with_provenance(if explicit {
                GpuHandleProvenance::Explicit
            } else {
                GpuHandleProvenance::Automatic
            })
            .unwrap()
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "imfilter owning fallback fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("kernel preparation must use checked native upload")
    }
    fn upload_numeric_raw(&self, view: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            Mode::UploadOperational => Err(anyhow::Error::new(Operational)),
            Mode::UploadUnsupported => Err(unsupported_provider_operation(
                "native kernel upload unavailable",
            )),
            Mode::UploadAlias | Mode::RestoreAlias => {
                Ok(self.alias.lock().unwrap().as_ref().unwrap().clone())
            }
            _ => self.inner.upload_numeric_raw(view),
        }
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
    fn imfilter<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        kernel: &'a GpuTensorHandle,
        _: &'a ImfilterOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.hooks.fetch_add(1, Ordering::SeqCst);
            *self.scratch.lock().unwrap() = Some(kernel.downgrade());
            match self.mode {
                Mode::Unsupported | Mode::RestoreAlias => Err(unsupported_provider_operation(
                    "imfilter not supported by provider",
                )),
                Mode::ResultAlias => Ok(self.alias.lock().unwrap().as_ref().unwrap().clone()),
                Mode::Pending => futures::future::pending().await,
                _ => Err(anyhow::Error::new(Operational)),
            }
        })
    }
}
fn host_kernel() -> Value {
    Value::Tensor(Tensor::new(vec![1.], vec![1, 1]).unwrap())
}
fn assert_operational(error: &runmat_runtime::RuntimeError) {
    assert_eq!(
        error.gpu_gather_retry,
        runmat_runtime::GpuGatherRetry::Never
    );
    let mut cursor: &(dyn Error + 'static) = error;
    while !cursor.is::<Operational>() {
        cursor = cursor.source().expect("exact operational cause retained");
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn operational_upload_and_hook_failures_never_gather_or_replay() {
    for mode in [Mode::Operational, Mode::UploadOperational] {
        let provider = Provider::new(mode);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider.input(&[1., 2., 3., 4.], &[2, 2], false);
        let error = futures::executor::block_on(runmat_runtime::call_builtin_async(
            "imfilter",
            &[Value::GpuTensor(input.clone()), host_kernel()],
        ))
        .unwrap_err();
        assert_operational(&error);
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
        assert_eq!(
            provider.hooks.load(Ordering::SeqCst),
            usize::from(matches!(mode, Mode::Operational))
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        if let Some(scratch) = provider.scratch.lock().unwrap().as_ref() {
            assert!(scratch.acquire().is_err());
        }
        input.allocation_lease().validate_live().unwrap();
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn typed_unsupported_preserves_supported_explicit_resident_fallback() {
    let provider = Provider::new(Mode::Unsupported);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider.input(&[1., 2., 3., 4.], &[2, 2], true);
    let output = futures::executor::block_on(runmat_runtime::call_builtin_async(
        "imfilter",
        &[Value::GpuTensor(input.clone()), host_kernel()],
    ))
    .unwrap();
    let Value::GpuTensor(output) = output else {
        panic!("explicit residency must be retained")
    };
    assert!(!output.shares_allocation_with(&input));
    assert_eq!(output.shape(), &[2, 2]);
    assert_eq!(handle_precision(&output), Some(ProviderPrecision::F64));
    assert_eq!(
        futures::executor::block_on(provider.inner.download(&output))
            .unwrap()
            .data,
        [1., 2., 3., 4.]
    );
    assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    input.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn typed_upload_unsupported_can_fallback_before_kernel_execution() {
    let provider = Provider::new(Mode::UploadUnsupported);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider.input(&[1., 2., 3., 4.], &[2, 2], false);
    let result = futures::executor::block_on(runmat_runtime::call_builtin_async(
        "imfilter",
        &[Value::GpuTensor(input.clone()), host_kernel()],
    ))
    .unwrap();
    let Value::Tensor(result) = result else {
        panic!("automatic placement permits host result")
    };
    assert_eq!(result.materialize_f64(), [1., 2., 3., 4.]);
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.hooks.load(Ordering::SeqCst), 0);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    input.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn invalid_aliases_reject_only_the_returned_owner_reference() {
    for mode in [Mode::UploadAlias, Mode::ResultAlias, Mode::RestoreAlias] {
        for alias_kernel in [false, true] {
            let provider = Provider::new(mode);
            let _selected = ThreadProviderGuard::set(Some(provider));
            let input = provider.input(&[1., 2., 3., 4.], &[2, 2], true);
            let kernel = if matches!(mode, Mode::UploadAlias) && alias_kernel {
                let foreign = Box::leak(Box::new(
                    runmat_accelerate::simple_provider::InProcessProvider::new(),
                ));
                // SAFETY: leaked process-lifetime owner has a fresh device id.
                unsafe { register_device_provider(foreign) };
                foreign
                    .upload(&HostTensorView {
                        data: &[1.; 4],
                        shape: &[2, 2],
                    })
                    .unwrap()
            } else {
                provider.input(&[1.; 4], &[2, 2], false)
            };
            *provider.alias.lock().unwrap() = Some(if alias_kernel {
                kernel.clone()
            } else {
                input.clone()
            });
            // Upload alias uses a host H to force upload; output alias uses real
            // resident H so both original allocations are protected.
            let h = if matches!(mode, Mode::UploadAlias) && !alias_kernel {
                Value::Tensor(Tensor::new(vec![1.; 4], vec![2, 2]).unwrap())
            } else {
                Value::GpuTensor(kernel.clone())
            };
            let result = futures::executor::block_on(runmat_runtime::call_builtin_async(
                "imfilter",
                &[Value::GpuTensor(input.clone()), h],
            ));
            assert_eq!(
                result.unwrap_err().gpu_gather_retry,
                runmat_runtime::GpuGatherRetry::Never
            );
            input.allocation_lease().validate_live().unwrap();
            kernel.allocation_lease().validate_live().unwrap();
            provider.alias.lock().unwrap().take();
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn cancelling_pending_hook_retires_uploaded_kernel_not_borrowed_image() {
    let provider = Provider::new(Mode::Pending);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider.input(&[1., 2., 3., 4.], &[2, 2], false);
    let args = [Value::GpuTensor(input.clone()), host_kernel()];
    let mut pending = runmat_runtime::call_builtin_async("imfilter", &args).boxed_local();
    assert!(matches!(
        pending.poll_unpin(&mut Context::from_waker(noop_waker_ref())),
        Poll::Pending
    ));
    let scratch = provider.scratch.lock().unwrap().take().unwrap();
    assert!(scratch.acquire().is_ok());
    drop(pending);
    assert!(scratch.acquire().is_err());
    input.allocation_lease().validate_live().unwrap();
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}
