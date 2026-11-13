use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, HostTensorView, ThreadProviderGuard,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    OnceLock,
};

struct Probe {
    inner: InProcessProvider,
    downloads: AtomicUsize,
}
impl AccelProvider for Probe {
    fn upload(&self, value: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(value)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(handle)
    }
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
}

#[test]
fn atan2_rejects_invalid_pair_before_resident_download() {
    let _serial = test_support::accel_test_lock();
    static PROVIDER: OnceLock<Probe> = OnceLock::new();
    let provider = PROVIDER.get_or_init(|| Probe {
        inner: InProcessProvider::new(),
        downloads: AtomicUsize::new(0),
    });
    provider.downloads.store(0, Ordering::SeqCst);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let source = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[1, 2],
        })
        .unwrap();
    for invalid in [
        Value::String("invalid".into()),
        Value::Complex(0.0, 0.0),
        Value::Tensor(Tensor::new(vec![0.0; 3], vec![1, 3]).unwrap()),
    ] {
        for reversed in [false, true] {
            let mut args = [Value::GpuTensor(source.clone()), invalid.clone()];
            if reversed {
                args.swap(0, 1);
            }
            assert!(block_on(crate::dispatcher::call_builtin_async("atan2", &args)).is_err());
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        }
    }
    // Live owners always provide complete immutable storage facts. An invalid
    // view is rejected at admission, before it can become a runtime argument.
    assert!(source
        .with_view(source.view().clone().with_shape(vec![1, 3]))
        .is_err());
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert!(block_on(provider.inner.download(&source)).is_ok());
    source.release().unwrap();
}
