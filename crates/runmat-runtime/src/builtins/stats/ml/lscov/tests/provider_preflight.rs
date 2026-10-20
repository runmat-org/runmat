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
fn invalid_algorithm_precedes_resident_design_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountingProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let a = Tensor::new(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], vec![3, 2]).unwrap();
    let a = crate::builtins::common::gpu_helpers::upload_tensor(provider, &a).unwrap();
    let b = tensor(vec![1.0, 3.0, 5.0], 3, 1);
    let error = block_on(lscov_builtin(
        Value::GpuTensor(a),
        b,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("invalid")]),
    ))
    .expect_err("invalid algorithm");
    assert!(error.message().contains("algorithm"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}

#[test]
fn impossible_response_geometry_precedes_resident_design_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountingProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let a = Tensor::new(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], vec![3, 2]).unwrap();
    let a = crate::builtins::common::gpu_helpers::upload_tensor(provider, &a).unwrap();
    let b = tensor(vec![1.0, 3.0], 2, 1);
    let error = block_on(lscov_builtin(
        Value::GpuTensor(a),
        b,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("response row mismatch");
    assert!(error.message().contains("observation"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}

#[test]
fn invalid_host_weights_precede_resident_design_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountingProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let a = Tensor::new(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], vec![3, 2]).unwrap();
    let a = crate::builtins::common::gpu_helpers::upload_tensor(provider, &a).unwrap();
    let b = tensor(vec![1.0, 3.0, 5.0], 3, 1);
    let invalid_weights = tensor(vec![0.0, 0.0, 0.0], 3, 1);
    let error = block_on(lscov_builtin(
        Value::GpuTensor(a),
        b,
        crate::builtins::common::test_support::builtin_arguments(vec![invalid_weights]),
    ))
    .expect_err("all-zero weights");
    assert!(error.message().contains("positive"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}

#[test]
fn nonvector_response_with_matching_count_rejects_wrong_rows_before_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountingProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let a = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![4, 1]).unwrap();
    let a = crate::builtins::common::gpu_helpers::upload_tensor(provider, &a).unwrap();
    let b = tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let error = block_on(lscov_builtin(
        Value::GpuTensor(a),
        b,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("nonvector row mismatch");
    assert!(error.message().contains("observation"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}
