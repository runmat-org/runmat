use super::*;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle,
    HostNumericTensorView, HostTensorView,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    downloads: AtomicUsize,
}

impl CountingProvider {
    fn new() -> &'static Self {
        Box::leak(Box::new(Self {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            downloads: AtomicUsize::new(0),
        }))
    }
}

impl AccelProvider for CountingProvider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(handle)
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
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download_numeric(handle)
    }
}

#[test]
fn invalid_host_option_value_precedes_resident_predictor_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountingProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
    let x = crate::builtins::common::gpu_helpers::upload_tensor(provider, &x).unwrap();
    let error = block_on(fitctree_builtin(
        None,
        Value::GpuTensor(x),
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1),
            Value::from("MinLeafSize"),
            Value::from("invalid"),
        ]),
    ))
    .expect_err("invalid structural control");
    assert!(error.message().contains("MinLeafSize"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}

#[test]
fn incompatible_response_geometry_precedes_resident_predictor_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountingProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
    let x = crate::builtins::common::gpu_helpers::upload_tensor(provider, &x).unwrap();
    let error = block_on(fitctree_builtin(
        None,
        Value::GpuTensor(x),
        crate::builtins::common::test_support::builtin_arguments(vec![tensor_value(
            vec![0.0, 1.0, 0.0],
            3,
            1,
        )]),
    ))
    .expect_err("response row mismatch");
    assert!(error.message().contains("length(Y)"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}
