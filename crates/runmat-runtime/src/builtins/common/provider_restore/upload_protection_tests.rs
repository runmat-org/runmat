use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, HostNumericTensorView, HostTensorView, ThreadProviderGuard,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

struct Provider {
    inner: InProcessProvider,
    response: Mutex<Option<GpuTensorHandle>>,
    frees: AtomicUsize,
}
impl Provider {
    fn new() -> &'static Self {
        Box::leak(Box::new(Self {
            inner: InProcessProvider::new(),
            response: Mutex::new(None),
            frees: AtomicUsize::new(0),
        }))
    }
    fn source(&self) -> GpuTensorHandle {
        self.inner
            .upload(&HostTensorView {
                data: &[0.5],
                shape: &[1, 1],
            })
            .unwrap()
    }
}
impl AccelProvider for Provider {
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        Ok(self
            .response
            .lock()
            .unwrap()
            .as_ref()
            .expect("configured upload")
            .clone())
    }
    fn upload_numeric(&self, _: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
        Ok(self
            .response
            .lock()
            .unwrap()
            .as_ref()
            .expect("configured upload")
            .clone())
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
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
        self.inner.device_info()
    }
}

#[test]
fn protected_restore_rejects_malformed_borrowed_alias_before_low_level_cleanup() {
    let _serial = test_support::accel_test_lock();
    let provider = Provider::new();
    let _selected = ThreadProviderGuard::set(Some(provider));
    let source = provider.source();
    let mut malformed = source.clone();
    malformed.descriptor.storage = None;
    *provider.response.lock().unwrap() = Some(malformed);
    for value in [
        Value::Num(0.25),
        Value::Complex(0.25, 0.5),
        Value::Tensor(Tensor::new(vec![0.25], vec![1, 1]).unwrap()),
        Value::ComplexTensor(ComplexTensor::new(vec![(0.25, 0.5)], vec![1, 1]).unwrap()),
    ] {
        assert!(upload_value_protected(
            provider,
            value,
            "restore-test",
            std::slice::from_ref(&source)
        )
        .is_err());
        assert_eq!(provider.frees.load(Ordering::SeqCst), 0);
        assert_eq!(
            block_on(provider.inner.download(&source)).unwrap().data,
            vec![0.5]
        );
    }
    provider.inner.free(&source).unwrap();
}

#[test]
fn restore_like_always_protects_its_borrowed_prototype() {
    let _serial = test_support::accel_test_lock();
    let provider = Provider::new();
    let _selected = ThreadProviderGuard::set(Some(provider));
    let prototype = provider.source();
    let mut malformed = prototype.clone();
    malformed.descriptor.storage = None;
    *provider.response.lock().unwrap() = Some(malformed);
    assert!(upload_value_like(provider, Value::Num(0.0), "restore-test", &prototype).is_err());
    assert_eq!(provider.frees.load(Ordering::SeqCst), 0);
    assert!(block_on(provider.inner.download(&prototype)).is_ok());
    provider.inner.free(&prototype).unwrap();
}

#[test]
fn restore_rejection_never_frees_through_unrelated_selected_provider() {
    let _serial = test_support::accel_test_lock();
    let invoked = Provider::new();
    let unrelated = Provider::new();
    let output = invoked.source();
    *invoked.response.lock().unwrap() = Some(output);
    let _selected = ThreadProviderGuard::set(Some(unrelated));
    // Upload has a valid physical descriptor but no registered exact owner.
    // The final ownership check rejects it and retains invoked-provider cleanup.
    assert!(upload_value_protected(invoked, Value::Num(0.25), "restore-test", &[]).is_err());
    assert_eq!(invoked.frees.load(Ordering::SeqCst), 1);
    assert_eq!(unrelated.frees.load(Ordering::SeqCst), 0);
}
