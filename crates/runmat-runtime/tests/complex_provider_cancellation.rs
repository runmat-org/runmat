use futures::FutureExt;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, GpuTensorHandle, HostTensorView,
    ProviderPrecision, ThreadProviderGuard,
};
use runmat_value::Value;
use std::sync::OnceLock;

struct PendingProvider(InProcessProvider);

fn pending(handles: &[&GpuTensorHandle]) -> AccelProviderFuture<'static, GpuTensorHandle> {
    for handle in handles {
        runmat_accelerate_api::set_handle_logical(handle, true);
        runmat_accelerate_api::record_handle_transpose(handle, 91, 17);
    }
    Box::pin(futures::future::pending())
}

impl AccelProvider for PendingProvider {
    fn device_id(&self) -> u32 {
        self.0.device_id()
    }
    fn device_info(&self) -> String {
        self.0.device_info()
    }
    fn precision(&self) -> ProviderPrecision {
        self.0.precision()
    }
    fn upload(&self, view: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.0.upload(view)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.0.download(handle)
    }
    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        self.0.free(handle)
    }
    fn unary_conj<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        pending(&[handle])
    }
    fn unary_real<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        pending(&[handle])
    }
    fn unary_imag<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        pending(&[handle])
    }
    fn complex_from_real<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        pending(&[handle])
    }
    fn complex_from_real_imag<'a>(
        &'a self,
        real: &'a GpuTensorHandle,
        imaginary: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        pending(&[real, imaginary])
    }
}

#[test]
fn abandoned_component_and_construction_calls_restore_each_borrowed_input() {
    static PROVIDER: OnceLock<PendingProvider> = OnceLock::new();
    let provider = PROVIDER.get_or_init(|| PendingProvider(InProcessProvider::new()));
    let _current = ThreadProviderGuard::set(Some(provider));
    for (name, count) in [
        ("conj", 1),
        ("real", 1),
        ("imag", 1),
        ("complex", 1),
        ("complex", 2),
    ] {
        let handles: Vec<_> = (0..count)
            .map(|_| {
                provider
                    .upload(&HostTensorView {
                        data: &[1.0, 2.0],
                        shape: &[1, 2],
                    })
                    .unwrap()
            })
            .collect();
        let args: Vec<_> = handles.iter().cloned().map(Value::GpuTensor).collect();
        let mut call = Box::pin(runmat_runtime::call_builtin_async(name, &args));
        assert!(
            call.as_mut().now_or_never().is_none(),
            "{name} must reach pending provider"
        );
        assert!(handles.iter().all(runmat_accelerate_api::handle_is_logical));
        drop(call);
        for handle in &handles {
            assert!(!runmat_accelerate_api::handle_is_logical(handle), "{name}");
            assert!(
                runmat_accelerate_api::handle_transpose_info(handle).is_none(),
                "{name}"
            );
            let value = futures::executor::block_on(provider.download(handle)).unwrap();
            assert_eq!(
                value.data,
                vec![1.0, 2.0],
                "borrowed allocation must remain live"
            );
            provider.free(handle).unwrap();
        }
    }
}
