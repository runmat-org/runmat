use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView, NumericElementType,
    ThreadProviderGuard,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct PendingDataProvider {
    owner: TestAllocationOwner,
    downloads: AtomicUsize,
}

impl AccelProvider for PendingDataProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "fprintf admission fixture".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { futures::future::pending().await })
    }

    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { futures::future::pending().await })
    }
}

#[test]
fn known_read_only_target_fails_before_pending_data_download() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fprintf_read_only_preflight");
    test_support::fs::write(&path, "unchanged").expect("fixture");
    let opened = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("r"),
    ])
    .expect("read-only fopen");
    let fid = opened.as_open().expect("file form").fid as i32;
    let provider = Box::leak(Box::new(PendingDataProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("resident value");
    let args = [
        Value::Num(fid as f64),
        Value::from("%d"),
        Value::GpuTensor(resident.handle.clone()),
    ];
    let mut call = Box::pin(evaluate(&args));
    let error = match futures::poll!(call.as_mut()) {
        std::task::Poll::Ready(Err(error)) => error,
        other => panic!("read-only target must fail before data download: {other:?}"),
    };
    assert_eq!(error.identifier(), Some("RunMat:fprintf:InvalidIdentifier"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(resident.probe.retirements(), 0);
    assert_eq!(
        test_support::fs::read_to_string(&path).unwrap(),
        "unchanged"
    );
    drop(call);
    run_fclose(&[Value::Num(fid as f64)]).expect("close read-only file");
    test_support::fs::remove_file(path).unwrap();
}

#[test]
fn known_malformed_format_fails_before_pending_data_download() {
    let provider = Box::leak(Box::new(PendingDataProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("resident value");
    let args = [Value::from("%q"), Value::GpuTensor(resident.handle.clone())];
    let mut call = Box::pin(evaluate(&args));
    let error = match futures::poll!(call.as_mut()) {
        std::task::Poll::Ready(Err(error)) => error,
        other => panic!("bad format must fail before data download: {other:?}"),
    };
    assert_eq!(error.identifier(), Some("RunMat:fprintf:InvalidFormat"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(resident.probe.retirements(), 0);
}

#[test]
fn later_resident_format_gate_precedes_any_download() {
    let provider = Box::leak(Box::new(PendingDataProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("resident format candidate");
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let args = [
        Value::Num(1.0),
        Value::Num(5.0),
        Value::GpuTensor(resident.handle.clone()),
    ];
    let mut call = Box::pin(evaluate(&args));
    let error = match futures::poll!(call.as_mut()) {
        std::task::Poll::Ready(Err(error)) => error,
        other => panic!("later format gate must fail before download: {other:?}"),
    };
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FprintfResidentFormatExtension")
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(resident.probe.retirements(), 0);
}
