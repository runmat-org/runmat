//! Applies to bounds and both top-k identities: all use restore_pair at their
//! only resident output boundary. This is fault-injected ownership evidence,
//! not hardware timing or GPU-kernel evidence.
use super::*;
use runmat_accelerate_api::{AccelProvider, HostNumericTensorView, HostTensorView};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

struct Provider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    uploads: AtomicUsize,
    frees: AtomicUsize,
    first: Mutex<Option<GpuTensorHandle>>,
    alias_second: bool,
}
impl AccelProvider for Provider {
    fn upload(&self, view: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(view)
    }
    fn upload_numeric_raw(&self, view: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
        if self.uploads.fetch_add(1, Ordering::SeqCst) == 1 {
            if self.alias_second {
                return Ok(self.first.lock().unwrap().clone().unwrap());
            }
            anyhow::bail!("injected second-output upload failure");
        }
        let output = self.inner.upload_numeric(view)?;
        *self.first.lock().unwrap() = Some(output.clone());
        Ok(output)
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
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "reduction output restoration test owner".into()
    }
}
fn provider(alias_second: bool) -> &'static Provider {
    let provider = Box::leak(Box::new(Provider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        uploads: AtomicUsize::new(0),
        frees: AtomicUsize::new(0),
        first: Mutex::new(None),
        alias_second,
    }));
    // Unique device and process-long allocation satisfy registration lifetime.
    unsafe {
        runmat_accelerate_api::register_device_provider(provider);
    }
    provider
}
#[test]
fn rejected_second_result_releases_first_without_freeing_source() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    for alias in [false, true] {
        let provider = provider(alias);
        let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let mut source = provider
            .upload(&HostTensorView {
                data: &[1., 2.],
                shape: &[2, 1],
            })
            .unwrap();
        runmat_accelerate_api::mark_handle_explicit(&mut source);
        let before = gpu_helpers::snapshot_handle_metadata(&source);
        assert!(restore_pair(
            &source,
            Value::Num(2.),
            Value::Num(1.),
            2,
            "maxk",
            &[&source]
        )
        .is_err());
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 2);
        assert_eq!(provider.frees.load(Ordering::SeqCst), 1);
        assert_eq!(gpu_helpers::snapshot_handle_metadata(&source), before);
        assert!(futures::executor::block_on(provider.download(&source)).is_ok());
        provider.free(&source).unwrap();
    }
}
#[test]
fn discarded_and_unrequested_results_are_not_uploaded() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    for retained in [0, 1] {
        let provider = provider(false);
        let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let source = provider
            .upload(&HostTensorView {
                data: &[1., 2.],
                shape: &[2, 1],
            })
            .unwrap();
        let (first, second) = restore_pair(
            &source,
            Value::Num(1.),
            Value::Num(2.),
            retained,
            "bounds",
            &[&source],
        )
        .unwrap();
        assert_eq!(provider.uploads.load(Ordering::SeqCst), retained);
        assert!(!matches!(second, Value::GpuTensor(_)));
        if let Value::GpuTensor(handle) = first {
            provider.free(&handle).unwrap();
        }
        provider.free(&source).unwrap();
    }
}
