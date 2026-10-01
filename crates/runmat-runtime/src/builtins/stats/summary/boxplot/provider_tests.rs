//! Registered-provider witnesses for boxplot admission and borrowed inputs.
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
        "boxplot registered-owner fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("boxplot must not upload during admission")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        if self.pending.load(Ordering::SeqCst) {
            return Box::pin(futures::future::pending());
        }
        Box::pin(async { Err(std::io::Error::other("boxplot fixture rejected download").into()) })
    }
    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        if self.pending.load(Ordering::SeqCst) {
            return Box::pin(futures::future::pending());
        }
        Box::pin(async { Err(std::io::Error::other("boxplot fixture rejected download").into()) })
    }
}

fn sample() -> Value {
    Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap())
}

#[test]
fn host_invalid_controls_and_geometry_reject_before_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloads::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let option = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let calls = [
        vec![
            Value::GpuTensor(resident.handle.clone()),
            Value::String("Whisker".into()),
            Value::Num(-1.0),
        ],
        vec![
            sample(),
            Value::String("Whisker".into()),
            Value::GpuTensor(option.handle.clone()),
            Value::String("Jitter".into()),
            Value::Num(-1.0),
        ],
        vec![
            sample(),
            Value::String("Whisker".into()),
            Value::GpuTensor(option.handle.clone()),
            Value::String("UnknownOption".into()),
            Value::Num(1.0),
        ],
        vec![
            Value::String("invalid sample".into()),
            Value::GpuTensor(resident.handle.clone()),
        ],
    ];
    for args in calls {
        let err = block_on(boxplot_builtin(args)).expect_err("known invalid host input");
        assert_eq!(err.identifier(), BOXPLOT_ERROR_INVALID_ARGUMENT.identifier);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        resident.handle.allocation_lease().validate_live().unwrap();
        option.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(resident.probe.retirements(), 0);
        assert_eq!(option.probe.retirements(), 0);
    }
}

#[test]
fn gather_failure_retains_cause_and_caller_lease() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloads::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let err = block_on(boxplot_builtin(vec![Value::GpuTensor(
        resident.handle.clone(),
    )]))
    .expect_err("rejected download");
    assert_eq!(err.identifier(), BOXPLOT_ERROR_INVALID_ARGUMENT.identifier);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    let gather = err
        .source
        .as_ref()
        .and_then(|source| source.downcast_ref::<RuntimeError>())
        .expect("nested gather error");
    assert_eq!(gather.identifier(), Some("RunMat:gather:DownloadFailed"));
    assert_eq!(err.span, gather.span);
    assert_eq!(err.gpu_gather_retry, gather.gpu_gather_retry);
    let mut source: &(dyn std::error::Error + 'static) = &err;
    while !source.is::<std::io::Error>() {
        source = source.source().expect("provider cause retained");
    }
    assert_eq!(source.to_string(), "boxplot fixture rejected download");
    resident.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(resident.probe.retirements(), 0);
}

#[test]
fn unpolled_and_abandoned_pending_future_keep_borrowed_input_live() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloads::default()));
    provider.pending.store(true, Ordering::SeqCst);
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let value = || Value::GpuTensor(resident.handle.clone());
    drop(boxplot_builtin(vec![value()]));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    resident.handle.allocation_lease().validate_live().unwrap();

    let mut pending = Box::pin(boxplot_builtin(vec![value()]));
    let waker = futures::task::noop_waker();
    let mut context = std::task::Context::from_waker(&waker);
    assert!(matches!(
        pending.as_mut().poll(&mut context),
        std::task::Poll::Pending
    ));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    drop(pending);
    resident.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(resident.probe.retirements(), 0);
}
