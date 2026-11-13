//! Registered-owner witnesses for coefficient admission before any transfer.
use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView, NumericElementType,
    ThreadProviderGuard,
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Default)]
struct CountedRejectDownloads {
    owner: TestAllocationOwner,
    downloads: AtomicUsize,
}

impl AccelProvider for CountedRejectDownloads {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "reference-line admission fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("reference-line admission must not upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Err(std::io::Error::other("reference-line rejected download").into()) })
    }
    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Err(std::io::Error::other("reference-line rejected download").into()) })
    }
}

#[test]
fn invalid_later_host_coefficient_and_style_precede_resident_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(CountedRejectDownloads::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let sample = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let resident = || Value::GpuTensor(sample.handle.clone());
    let bad_calls = [
        vec![resident(), Value::String("bad intercept".into())],
        vec![resident(), Value::Num(f64::INFINITY)],
        vec![resident(), Value::Num(0.0), Value::String("Color".into())],
    ];
    for args in bad_calls {
        let err = block_on(run_reference_lines(
            None,
            crate::builtins::common::test_support::builtin_arguments(args),
            ReferenceLineOwner::Refline,
        ))
        .expect_err("host-invalid call");
        assert_eq!(
            err.identifier(),
            runmat_builtins::REFLINE_ERROR_INVALID_ARGUMENT.identifier
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        sample.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(sample.probe.retirements(), 0);
    }
}

#[test]
fn resident_download_failure_preserves_nested_cause_and_borrowed_lease() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(CountedRejectDownloads::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let sample = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let err = block_on(run_reference_lines(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::GpuTensor(sample.handle.clone()),
            Value::Num(0.0),
        ]),
        ReferenceLineOwner::Refline,
    ))
    .expect_err("rejected download");
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    let cause = err
        .source
        .as_ref()
        .and_then(|source| source.downcast_ref::<RuntimeError>())
        .expect("nested gather cause");
    assert_eq!(cause.identifier(), Some("RunMat:gather:DownloadFailed"));
    assert_eq!(err.gpu_gather_retry, cause.gpu_gather_retry);
    assert_eq!(err.span, cause.span);
    sample.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(sample.probe.retirements(), 0);
}
