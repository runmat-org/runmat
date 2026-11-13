use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView, NumericElementType,
    ThreadProviderGuard,
};
use std::sync::atomic::{AtomicUsize, Ordering};

fn poll_once<F: std::future::Future>(future: std::pin::Pin<&mut F>) -> std::task::Poll<F::Output> {
    let mut context = std::task::Context::from_waker(futures::task::noop_waker_ref());
    std::future::Future::poll(future, &mut context)
}

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
    let error = match poll_once(call.as_mut()) {
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
fn read_only_target_precedes_pending_resident_format_download() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fprintf_resident_format_read_only");
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
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("resident format");
    let args = [
        Value::Num(fid as f64),
        Value::GpuTensor(resident.handle.clone()),
        Value::Num(7.0),
    ];
    let mut call = Box::pin(evaluate(&args));
    let error = match poll_once(call.as_mut()) {
        std::task::Poll::Ready(Err(error)) => error,
        other => panic!("known invalid target must precede format download: {other:?}"),
    };
    assert_eq!(error.identifier(), Some("RunMat:fprintf:InvalidIdentifier"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(resident.probe.retirements(), 0);
    drop(call);
    run_fclose(&[Value::Num(fid as f64)]).expect("close");
    assert_eq!(
        test_support::fs::read_to_string(&path).unwrap(),
        "unchanged"
    );
    test_support::fs::remove_file(path).unwrap();
}

#[test]
fn later_stream_label_data_cannot_override_known_target_before_resident_format() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fprintf_later_stream_data");
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
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("resident format");
    let args = [
        Value::Num(fid as f64),
        Value::GpuTensor(resident.handle.clone()),
        Value::from("stderr"),
    ];
    let mut call = Box::pin(evaluate(&args));
    let error = match poll_once(call.as_mut()) {
        std::task::Poll::Ready(Err(error)) => error,
        other => panic!("later data cannot defer known target error: {other:?}"),
    };
    assert_eq!(error.identifier(), Some("RunMat:fprintf:InvalidIdentifier"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(resident.probe.retirements(), 0);
    drop(call);
    run_fclose(&[Value::Num(fid as f64)]).expect("close");
    assert_eq!(
        test_support::fs::read_to_string(&path).unwrap(),
        "unchanged"
    );
    test_support::fs::remove_file(path).unwrap();
}

#[test]
fn known_invalid_character_matrix_format_precedes_resident_data() {
    let provider = Box::leak(Box::new(PendingDataProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("resident data");
    let matrix = runmat_value::CharArray::from_unicode_scalars(vec!['%', 'd', '%', 'd'], 2, 2)
        .expect("character matrix");
    let args = [
        Value::CharArray(matrix),
        Value::GpuTensor(resident.handle.clone()),
    ];
    let mut call = Box::pin(evaluate(&args));
    let error = match poll_once(call.as_mut()) {
        std::task::Poll::Ready(Err(error)) => error,
        other => panic!("invalid format geometry must precede data download: {other:?}"),
    };
    assert_eq!(error.identifier(), FPRINTF_ERROR_FORMAT.identifier);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(resident.probe.retirements(), 0);
}

#[test]
fn unambiguous_invalid_fid_never_falls_back_to_stdout_or_downloads_data() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    crate::console::reset_thread_buffer();
    let provider = Box::leak(Box::new(PendingDataProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("resident data");
    for invalid in [
        Value::Num(f64::NAN),
        Value::Num(f64::INFINITY),
        Value::Num(2.5),
    ] {
        let args = [
            invalid,
            Value::from("%d"),
            Value::GpuTensor(resident.handle.clone()),
        ];
        let mut call = Box::pin(evaluate(&args));
        let error = match poll_once(call.as_mut()) {
            std::task::Poll::Ready(Err(error)) => error,
            other => panic!("invalid fid must precede resident data: {other:?}"),
        };
        assert_eq!(error.identifier(), Some("RunMat:fprintf:InvalidInput"));
        drop(call);
    }
    let exact_overflow = Value::Int(runmat_value::IntValue::U64(u64::MAX));
    let args = [exact_overflow, Value::from("%d")];
    let error = evaluate(&args);
    let error = futures::executor::block_on(error).expect_err("wide fid invalid");
    assert_eq!(error.identifier(), Some("RunMat:fprintf:InvalidInput"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(resident.probe.retirements(), 0);
    assert!(crate::console::take_thread_buffer().is_empty());
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
    let error = match poll_once(call.as_mut()) {
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
    let error = match poll_once(call.as_mut()) {
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
