//! Registered-owner transfer and borrowed-lease witnesses for deconv.

use std::sync::atomic::{AtomicUsize, Ordering};

use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelIntegerDownloadFuture, AccelNumericDownloadFuture, AccelProvider,
    GpuTensorHandle, HostNumericTensorView, HostTensorView, ThreadProviderGuard,
};
use runmat_value::{IntValue, IntegerStorage, Tensor, Value};

use super::deconv_builtin;

struct CountedOwner {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    integer_downloads: AtomicUsize,
    other_downloads: AtomicUsize,
}

impl CountedOwner {
    fn new() -> &'static Self {
        Box::leak(Box::new(Self {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            integer_downloads: AtomicUsize::new(0),
            other_downloads: AtomicUsize::new(0),
        }))
    }

    fn wide_input(&self, values: Vec<u64>) -> GpuTensorHandle {
        let shape = vec![1, values.len()];
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(values), shape).expect("typed integer fixture");
        crate::builtins::common::gpu_helpers::upload_tensor(self, &tensor)
            .expect("upload native integer fixture")
    }
}

impl AccelProvider for CountedOwner {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }

    fn device_info(&self) -> String {
        self.inner.device_info()
    }

    fn upload(&self, host: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.inner.upload_numeric_raw(host)
    }

    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.other_downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(handle)
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.integer_downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download_numeric(handle)
    }

    fn download_integer<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelIntegerDownloadFuture<'a> {
        self.other_downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download_integer(handle)
    }
}

#[test]
fn wide_integer_operand_reads_once_and_preserves_borrowed_input() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountedOwner::new();
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let input = provider.wide_input(vec![1u64 << 54, 0]);
    let denominator = provider.wide_input(vec![1]);
    let result = block_on(deconv_builtin(
        None,
        Value::GpuTensor(input.clone()),
        Value::GpuTensor(denominator.clone()),
    ))
    .expect("exact wide deconvolution");
    assert!(matches!(result, Value::GpuTensor(_)));
    assert_eq!(provider.integer_downloads.load(Ordering::SeqCst), 2);
    assert_eq!(provider.other_downloads.load(Ordering::SeqCst), 0);
    assert!(input.allocation_lease().validate_live().is_ok());
    assert!(denominator.allocation_lease().validate_live().is_ok());
}

#[test]
fn invalid_host_peer_rejects_before_any_resident_readback() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountedOwner::new();
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let numerator = provider.wide_input(vec![1u64 << 54, 0]);
    let denominator = Tensor::new(vec![0.0, 0.0], vec![1, 2]).expect("zero denominator");
    let zero_error = block_on(deconv_builtin(
        None,
        Value::GpuTensor(numerator.clone()),
        Value::Tensor(denominator),
    ))
    .expect_err("known zero denominator");
    assert_eq!(
        zero_error.identifier(),
        super::DECONV_ERROR_DENOMINATOR_INVALID.identifier
    );
    assert_eq!(provider.integer_downloads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.other_downloads.load(Ordering::SeqCst), 0);
    assert!(numerator.allocation_lease().validate_live().is_ok());

    let denominator = provider.wide_input(vec![1]);
    let inexact_error = block_on(deconv_builtin(
        None,
        Value::Int(IntValue::U64((1u64 << 53) + 1)),
        Value::GpuTensor(denominator.clone()),
    ))
    .expect_err("known inexact host numerator");
    assert_eq!(
        inexact_error.identifier(),
        super::DECONV_ERROR_INVALID_INPUT.identifier
    );
    assert_eq!(provider.integer_downloads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.other_downloads.load(Ordering::SeqCst), 0);
    assert!(denominator.allocation_lease().validate_live().is_ok());
}
