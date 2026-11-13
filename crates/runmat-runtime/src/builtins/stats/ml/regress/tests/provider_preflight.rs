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

fn resident_design(provider: &'static CountingProvider) -> Value {
    let x = Tensor::new(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], vec![3, 2]).unwrap();
    let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &x).unwrap();
    Value::GpuTensor(handle)
}

#[test]
fn known_invalid_alpha_precedes_resident_design_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountingProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let y = tensor(vec![1.0, 3.0, 5.0], 3, 1);
    let error = block_on(regress_builtin(
        y,
        resident_design(provider),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ))
    .expect_err("alpha outside open interval");
    assert!(error.message().contains("alpha"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}

#[test]
fn known_invalid_response_geometry_precedes_resident_design_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountingProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let y = tensor(vec![1.0, 3.0], 2, 1);
    let error = block_on(regress_builtin(
        y,
        resident_design(provider),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("response/design mismatch");
    assert!(error.message().contains("length(y)"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}

#[test]
fn integer_alpha_extension_gate_precedes_resident_design_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountingProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let y = tensor(vec![1.0, 3.0, 5.0], 3, 1);
    let alpha =
        Value::Tensor(Tensor::new_integer(IntegerStorage::I64(vec![1]), vec![1, 1]).unwrap());
    let error = block_on(regress_builtin(
        y,
        resident_design(provider),
        crate::builtins::common::test_support::builtin_arguments(vec![alpha]),
    ))
    .expect_err("integer alpha is extension-only");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:RegressIntegerAlphaExtension")
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}

#[test]
fn empty_observations_precede_resident_alpha_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountingProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let alpha = Tensor::new(vec![0.05], vec![1, 1]).unwrap();
    let alpha = crate::builtins::common::gpu_helpers::upload_tensor(provider, &alpha).unwrap();
    let response = tensor(Vec::new(), 0, 1);
    let design = tensor(Vec::new(), 0, 2);
    let error = block_on(regress_builtin(
        response,
        design,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(alpha)]),
    ))
    .expect_err("empty observations");
    assert!(error.message().contains("complete observation"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}
