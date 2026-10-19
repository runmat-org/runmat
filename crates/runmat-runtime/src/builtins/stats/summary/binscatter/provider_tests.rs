//! Registered-owner admission, failure provenance, and borrowed lease witnesses.
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
        "binscatter registered-owner fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("binscatter must not upload during admission")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        if self.pending.load(Ordering::SeqCst) {
            return Box::pin(futures::future::pending());
        }
        Box::pin(async {
            Err(std::io::Error::other("binscatter fixture rejected download").into())
        })
    }
    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        if self.pending.load(Ordering::SeqCst) {
            return Box::pin(futures::future::pending());
        }
        Box::pin(async {
            Err(std::io::Error::other("binscatter fixture rejected download").into())
        })
    }
}

#[test]
fn unpolled_and_abandoned_gather_keep_caller_lease_live() {
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
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
    let host = || Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap());

    drop(binscatter_builtin(vec![value(), host()]));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    sample.handle.allocation_lease().validate_live().unwrap();

    let mut pending = Box::pin(binscatter_builtin(vec![value(), host()]));
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
fn known_invalid_controls_and_geometry_preflight_without_download() {
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloads::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let sample = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let resident = || Value::GpuTensor(sample.handle.clone());
    let host = || Value::Tensor(Tensor::new(vec![0.0, 0.2, 0.8, 1.0], vec![4, 1]).unwrap());
    let calls = [
        vec![
            resident(),
            host(),
            Value::String("NumBins".into()),
            Value::Num(0.0),
        ],
        vec![
            resident(),
            host(),
            Value::String("XLimits".into()),
            Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap()),
        ],
        vec![
            resident(),
            host(),
            Value::String("NoSuchOption".into()),
            Value::Num(1.0),
        ],
        vec![
            resident(),
            host(),
            Value::String("DisplayStyle".into()),
            Value::String("points".into()),
        ],
        vec![
            resident(),
            host(),
            Value::String("NumBins".into()),
            resident(),
            Value::String("FaceAlpha".into()),
            Value::Num(2.0),
        ],
    ];
    for args in calls {
        let err = block_on(binscatter_builtin(args)).expect_err("known invalid host controls");
        assert_eq!(
            err.identifier(),
            BINSCATTER_ERROR_INVALID_ARGUMENT.identifier
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        sample.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(sample.probe.retirements(), 0);
    }
    let table = crate::builtins::table::table_from_columns(
        vec!["x".into(), "y".into()],
        vec![host(), host()],
    )
    .unwrap();
    let selector = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let err = block_on(binscatter_builtin(vec![
        table,
        Value::GpuTensor(selector.handle.clone()),
        Value::String("absent".into()),
    ]))
    .expect_err("invalid host table selector before resident index download");
    assert!(err.message().contains("not found"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    sample.handle.allocation_lease().validate_live().unwrap();
    selector.handle.allocation_lease().validate_live().unwrap();
    let invalid_selected = crate::builtins::table::table_from_columns(
        vec!["x".into(), "y".into()],
        vec![Value::String("text".into()), Value::Num(1.0)],
    )
    .unwrap();
    let err = block_on(binscatter_builtin(vec![
        invalid_selected,
        Value::String("x".into()),
        Value::GpuTensor(selector.handle.clone()),
    ]))
    .expect_err("known selected text column before resident y selector");
    assert!(err.message().contains("x must be numeric"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    selector.handle.allocation_lease().validate_live().unwrap();
    let matrix = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let err = block_on(binscatter_builtin(vec![
        Value::GpuTensor(matrix.handle.clone()),
        host(),
    ]))
    .expect_err("known matrix invalid");
    assert!(err.message().contains("vector"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    matrix.handle.allocation_lease().validate_live().unwrap();
}

#[test]
fn gather_failure_retains_original_cause_and_borrowed_input() {
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloads::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let sample = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let host = Value::Tensor(Tensor::new(vec![0.0, 0.2, 0.8, 1.0], vec![4, 1]).unwrap());
    let err = block_on(binscatter_builtin(vec![
        Value::GpuTensor(sample.handle.clone()),
        host,
    ]))
    .expect_err("rejected download");
    assert_eq!(
        err.identifier(),
        BINSCATTER_ERROR_INVALID_ARGUMENT.identifier
    );
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
    assert_eq!(source.to_string(), "binscatter fixture rejected download");
    sample.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(sample.probe.retirements(), 0);
}
