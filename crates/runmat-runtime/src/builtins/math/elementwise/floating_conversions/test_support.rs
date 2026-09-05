use crate::builtins::common::test_support;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelIntegerDownloadFuture, AccelNumericDownloadFuture, AccelProvider,
    AccelProviderFuture, GpuTensorHandle, HostIntegerTensorView, HostNumericTensorView,
    HostTensorView, ProviderPrecision, ThreadProviderGuard,
};
use std::sync::Mutex;

#[derive(Clone, Copy)]
pub(super) enum HookBehavior {
    Success,
    Unsupported,
    Failure,
    Malformed,
}

pub(super) struct ConversionHookProvider {
    inner: InProcessProvider,
    behavior: HookBehavior,
    malformed_output: Mutex<Option<GpuTensorHandle>>,
}

impl ConversionHookProvider {
    fn new(behavior: HookBehavior) -> Self {
        Self {
            inner: InProcessProvider::new(),
            behavior,
            malformed_output: Mutex::new(None),
        }
    }

    fn invoke<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        single: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            match self.behavior {
                HookBehavior::Success if single => self.inner.unary_single(input).await,
                HookBehavior::Success => self.inner.unary_double(input).await,
                HookBehavior::Unsupported => {
                    Err(runmat_accelerate_api::unsupported_provider_operation(
                        "conversion hook unavailable",
                    ))
                }
                HookBehavior::Failure => anyhow::bail!("injected conversion kernel failure"),
                HookBehavior::Malformed => {
                    let output = self.inner.upload(&HostTensorView {
                        data: &[1.0],
                        shape: &[1, 1],
                    })?;
                    *self.malformed_output.lock().expect("malformed output lock") =
                        Some(output.clone());
                    Ok(output)
                }
            }
        })
    }

    pub(super) fn take_malformed_output(&self) -> GpuTensorHandle {
        self.malformed_output
            .lock()
            .expect("malformed output lock")
            .take()
            .expect("provider recorded malformed output")
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

    fn upload_numeric(&self, host: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload_numeric(host)
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
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

    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        self.inner.free(handle)
    }

    fn unary_double<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.invoke(input, false)
    }

    fn unary_single<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.invoke(input, true)
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
