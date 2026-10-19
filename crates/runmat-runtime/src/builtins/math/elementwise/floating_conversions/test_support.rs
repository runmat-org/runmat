use crate::builtins::common::test_support;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    allocation::WeakAllocationLease, AccelDownloadFuture, AccelIntegerDownloadFuture,
    AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle, HostIntegerTensorView,
    HostNumericTensorView, HostTensorView, ProviderExecutionFuture, ProviderPrecision,
    ThreadProviderGuard,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Clone, Copy)]
pub(super) enum HookBehavior {
    Success,
    Unsupported,
    Failure,
    Malformed,
    PendingDirectHook,
    PendingTemplateDownload,
}

pub(super) struct ConversionHookProvider {
    inner: InProcessProvider,
    behavior: HookBehavior,
    malformed_output: Mutex<Option<WeakAllocationLease>>,
    created_output: Mutex<Option<WeakAllocationLease>>,
    hook_calls: AtomicUsize,
    numeric_download_calls: AtomicUsize,
}

impl ConversionHookProvider {
    fn new(behavior: HookBehavior) -> Self {
        Self {
            inner: InProcessProvider::new(),
            behavior,
            malformed_output: Mutex::new(None),
            created_output: Mutex::new(None),
            hook_calls: AtomicUsize::new(0),
            numeric_download_calls: AtomicUsize::new(0),
        }
    }

    fn invoke<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        single: bool,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.hook_calls.fetch_add(1, Ordering::SeqCst);
            match self.behavior {
                HookBehavior::Success => self.inner.unary_double(input).await,
                HookBehavior::PendingTemplateDownload => {
                    // A valid same-class hook may return an input alias. This
                    // fixture deliberately produces a genuinely distinct
                    // allocation so abandonment must retire an intermediate.
                    let source = self.inner.download(input).await?;
                    let data = if single {
                        runmat_accelerate_api::HostNumericDataOwned::F32(
                            source.data.iter().map(|x| *x as f32).collect(),
                        )
                    } else {
                        runmat_accelerate_api::HostNumericDataOwned::F64(source.data)
                    };
                    let host = runmat_accelerate_api::HostNumericTensorOwned {
                        data,
                        shape: source.shape,
                        storage: source.storage,
                    };
                    let output = self.inner.upload_numeric_host(&host.as_view())?;
                    assert!(!output.shares_allocation_with(input));
                    *self.created_output.lock().unwrap() =
                        Some(output.allocation_lease().downgrade());
                    Ok(output)
                }
                HookBehavior::Unsupported => {
                    Err(runmat_accelerate_api::unsupported_provider_operation(
                        "conversion hook unavailable",
                    )
                    .into())
                }
                HookBehavior::Failure => {
                    Err(anyhow::anyhow!("injected conversion kernel failure").into())
                }
                HookBehavior::PendingDirectHook => futures::future::pending().await,
                HookBehavior::Malformed => {
                    let output = self.inner.upload(&HostTensorView {
                        data: &[1.0],
                        shape: &[1, 1],
                    })?;
                    *self.malformed_output.lock().expect("malformed output lock") =
                        Some(output.allocation_lease().downgrade());
                    Ok(output)
                }
            }
        })
    }

    pub(super) fn take_malformed_output(&self) -> WeakAllocationLease {
        self.malformed_output
            .lock()
            .expect("malformed output lock")
            .take()
            .expect("provider recorded malformed output")
    }

    pub(super) fn take_created_output(&self) -> WeakAllocationLease {
        self.created_output.lock().unwrap().take().unwrap()
    }

    pub(super) fn hook_calls(&self) -> usize {
        self.hook_calls.load(Ordering::SeqCst)
    }

    pub(super) fn numeric_download_calls(&self) -> usize {
        self.numeric_download_calls.load(Ordering::SeqCst)
    }
}

impl AccelProvider for ConversionHookProvider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }

    fn device_info(&self) -> String {
        self.inner.device_info()
    }

    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
    }

    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.inner.upload_numeric_raw(host)
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.numeric_download_calls.fetch_add(1, Ordering::SeqCst);
        if matches!(self.behavior, HookBehavior::PendingTemplateDownload) {
            return Box::pin(futures::future::pending());
        }
        self.inner.download_numeric(handle)
    }

    fn upload_integer(&self, host: &HostIntegerTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload_integer(host)
    }

    fn download_integer<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelIntegerDownloadFuture<'a> {
        self.inner.download_integer(handle)
    }

    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }

    fn unary_double<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.invoke(input, false)
    }

    fn unary_single<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if matches!(self.behavior, HookBehavior::Success) {
                self.hook_calls.fetch_add(1, Ordering::SeqCst);
                self.inner.unary_single(input).await
            } else {
                self.invoke(input, true).await
            }
        })
    }
}

/// Release all caller-owned aliases together, then check the physical lease,
/// not just a view identity. The observer never retains an allocation.
pub(super) fn release_final_handles(handles: impl IntoIterator<Item = GpuTensorHandle>) {
    let handles: Vec<_> = handles.into_iter().collect();
    let physical: Vec<_> = handles
        .iter()
        .map(|handle| handle.allocation_lease().downgrade())
        .collect();
    for handle in handles {
        handle.release().expect("release owned fixture handle");
    }
    for weak in physical {
        assert!(
            weak.acquire().is_none(),
            "physical lease retained after all fixture owners left"
        );
    }
}

pub(super) fn with_conversion_hook_provider<R>(
    behavior: HookBehavior,
    test: impl FnOnce(&'static ConversionHookProvider) -> R,
) -> R {
    let _state = test_support::accel_test_lock();
    let provider = Box::leak(Box::new(ConversionHookProvider::new(behavior)));
    // SAFETY: the leaked provider has process lifetime and owns a fresh device
    // identifier allocated by its in-process backend.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    test(provider)
}
