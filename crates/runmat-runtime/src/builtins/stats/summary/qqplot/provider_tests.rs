//! Registered owner witnesses for Q-Q plot admission and gather errors.
use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView, NumericElementType,
    ThreadProviderGuard,
};
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Default)]
struct RejectDownloads {
    owner: TestAllocationOwner,
    downloads: AtomicUsize,
    pending: AtomicBool,
}

impl AccelProvider for RejectDownloads {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "qqplot registered-owner preflight witness".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("Q-Q plot does not upload during input admission")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        if self.pending.load(Ordering::SeqCst) {
            return Box::pin(futures::future::pending());
        }
        Box::pin(async { Err(std::io::Error::other("qqplot fixture rejected download").into()) })
    }
    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        if self.pending.load(Ordering::SeqCst) {
            return Box::pin(futures::future::pending());
        }
        Box::pin(async { Err(std::io::Error::other("qqplot fixture rejected download").into()) })
    }
}

#[test]
fn unpolled_and_abandoned_gather_keep_caller_lease_live() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloads::default()));
    provider.pending.store(true, Ordering::SeqCst);
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let sample = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let value = || Value::GpuTensor(sample.handle.clone());

    drop(qqplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![value()]),
    ));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    sample.handle.allocation_lease().validate_live().unwrap();

    let mut pending = Box::pin(qqplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![value()]),
    ));
    let waker = futures::task::noop_waker();
    let mut context = std::task::Context::from_waker(&waker);
    assert!(matches!(
        pending.as_mut().poll(&mut context),
        std::task::Poll::Pending
    ));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    drop(pending);
    sample.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(sample.probe.retirements(), 0);
}

#[test]
fn known_invalid_host_roles_reject_without_resident_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloads::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let sample = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let resident = || Value::GpuTensor(sample.handle.clone());
    assert!(runmat_accelerate_api::same_provider_instance(
        runmat_accelerate_api::provider_for_handle(&sample.handle).unwrap(),
        provider,
    ));
    for args in [
        vec![resident(), Value::from("not a sample")],
        vec![resident(), Value::Num(1.0), Value::Complex(1.0, 1.0)],
        vec![
            resident(),
            Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
            Value::Num(-1.0),
        ],
        vec![
            resident(),
            Value::Num(1.0),
            Value::Num(50.0),
            Value::Num(2.0),
        ],
    ] {
        let error = block_on(qqplot_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect_err("known-invalid call");
        assert_eq!(error.identifier(), QQPLOT_ERROR_INVALID_ARGUMENT.identifier);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        sample.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(sample.probe.retirements(), 0);
    }
    let wrong_shape = provider
        .owner
        .allocate_zeroed(
            vec![2, 2, 2],
            NumericElementType::F64,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let error = block_on(qqplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
            wrong_shape.handle.clone(),
        )]),
    ))
    .expect_err("known rank-three matrix is invalid");
    assert!(error.message().contains("vector or 2-D matrix"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    wrong_shape
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
}

#[test]
fn provider_failure_preserves_cause_and_borrowed_source() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloads::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let sample = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let error = block_on(qqplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
            sample.handle.clone(),
        )]),
    ))
    .expect_err("provider rejects gather");
    assert_eq!(error.identifier(), QQPLOT_ERROR_INVALID_ARGUMENT.identifier);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    let gather = error
        .source
        .as_ref()
        .and_then(|source| source.downcast_ref::<RuntimeError>())
        .expect("nested gather RuntimeError");
    assert_eq!(gather.identifier(), Some("RunMat:gather:DownloadFailed"));
    assert_eq!(error.span, gather.span);
    assert_eq!(error.gpu_gather_retry, gather.gpu_gather_retry);
    let mut source: &(dyn std::error::Error + 'static) = &error;
    while !source.is::<std::io::Error>() {
        source = source.source().expect("provider cause retained");
    }
    assert_eq!(source.to_string(), "qqplot fixture rejected download");
    sample.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(sample.probe.retirements(), 0);
}
