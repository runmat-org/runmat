//! Fault-injected provider results exercise the complete builtin boundary.
use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, HostTensorView};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};
struct Provider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    result: Mutex<Option<GpuTensorHandle>>,
    frees: AtomicUsize,
}
impl AccelProvider for Provider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        if let Some(handle) = self.result.lock().unwrap().as_ref() {
            return Ok(handle.clone());
        }
        self.inner.upload(host)
    }
    fn download<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }
    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        self.frees.fetch_add(1, Ordering::SeqCst);
        self.inner.free(handle)
    }
    fn device_info(&self) -> String {
        "toeplitz test owner".into()
    }
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
}
#[test]
fn rejected_uploads_preserve_both_operands_and_release_fresh_results() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(Provider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        result: Mutex::new(None),
        frees: AtomicUsize::new(0),
    }));
    // The unique in-process device id and leaked lifetime satisfy ownership registration.
    unsafe {
        runmat_accelerate_api::register_device_provider(provider);
    }
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let c = provider
        .upload(&HostTensorView {
            data: &[1., 2.],
            shape: &[2, 1],
        })
        .unwrap();
    let r = provider
        .upload(&HostTensorView {
            data: &[1., 3.],
            shape: &[1, 2],
        })
        .unwrap();
    let malformed = provider
        .upload(&HostTensorView {
            data: &[0.],
            shape: &[1, 1],
        })
        .unwrap();
    for (result, expected_frees) in [(&c, 0), (&r, 0), (&malformed, 1)] {
        *provider.result.lock().unwrap() = Some(result.clone());
        let error = block_on(toeplitz_builtin(vec![
            Value::GpuTensor(c.clone()),
            Value::GpuTensor(r.clone()),
        ]))
        .unwrap_err();
        assert_eq!(error.identifier(), ERROR_INTERNAL.identifier);
        assert_eq!(provider.frees.load(Ordering::SeqCst), expected_frees);
        assert!(block_on(provider.download(&c)).is_ok());
        assert!(block_on(provider.download(&r)).is_ok());
    }
    provider.free(&c).unwrap();
    provider.free(&r).unwrap();
}
