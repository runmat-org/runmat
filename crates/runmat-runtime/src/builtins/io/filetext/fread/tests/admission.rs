use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView, NumericElementType,
    ThreadProviderGuard,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct PendingControlProvider {
    owner: TestAllocationOwner,
    downloads: AtomicUsize,
}

impl AccelProvider for PendingControlProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "fread admission fixture".into()
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

fn poll_once<F: std::future::Future>(future: std::pin::Pin<&mut F>) -> std::task::Poll<F::Output> {
    let mut context = std::task::Context::from_waker(futures::task::noop_waker_ref());
    std::future::Future::poll(future, &mut context)
}

#[test]
fn cancelled_pending_resident_control_drops_download_and_restores_session_fid() {
    use crate::context::RuntimeContext;
    use crate::execution::RuntimeExecutionService;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::rc::Rc;

    let _registry = registry_guard();
    let _accel = test_support::accel_test_lock();
    registry::reset_for_tests();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let path = unique_path("fread_cancelled_control");
    test_support::fs::write(&path, "abc").expect("source fixture");
    let cancellation = CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(RuntimeExecutionService::new()),
        cancellation.token(),
    );
    let opened = futures::executor::block_on(fopen::evaluate_in(
        Some(&runtime),
        &[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("r"),
        ],
    ))
    .expect("session open");
    let fid = opened.as_open().expect("open form").fid as i32;
    let provider = Box::leak(Box::new(PendingControlProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("resident size");
    let fid_value = Value::Num(fid as f64);
    let controls = [Value::GpuTensor(resident.handle.clone())];
    let mut call = Box::pin(evaluate_in(Some(&runtime), &fid_value, &controls));
    assert!(poll_once(call.as_mut()).is_pending());
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    cancellation.cancel(CancellationReason::User);
    let error = futures::executor::block_on(call).expect_err("cancelled gather");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(
        resident.probe.retirements(),
        0,
        "borrowed resident input stays live"
    );
    assert!(registry::acquire_file_in(fid, Some(&runtime)).is_ok());
    futures::executor::block_on(registry::close_async_in(fid, Some(&runtime)))
        .expect("close after cancelled gather");
    test_support::fs::remove_file(path).expect("cleanup");
    registry::reset_for_tests();
}

#[test]
fn host_controls_are_prepared_as_typed_values_once() {
    let fid = Value::Num(3.0);
    let arguments = [
        Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()),
        Value::from("*uint16"),
        Value::Num(1.0),
        Value::from("ieee-be"),
    ];
    let prepared = PreparedFreadCall::new(
        &fid,
        crate::call::FunctionArgumentView::from_slice(&arguments),
    )
    .expect("host admission");
    assert_eq!(
        futures::executor::block_on(prepared.resolve_fid()).unwrap(),
        3
    );
    let controls = futures::executor::block_on(prepared.resolve_controls("native")).unwrap();
    assert!(matches!(
        controls.size,
        SizeSpec::Matrix {
            rows: 2,
            cols: Some(3)
        }
    ));
    assert!(matches!(
        controls.precision.output,
        OutputKind::Numeric(NumericDType::U16)
    ));
    assert_eq!(controls.skip, 1);
    assert!(matches!(controls.machine, MachineFormat::BigEndian));
}

#[test]
fn malformed_host_controls_precede_resident_identifier_download() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let provider = Box::leak(Box::new(PendingControlProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let allocation = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("resident identifier");
    let fid = Value::GpuTensor(allocation.handle.clone());
    let invalid_calls = [
        (vec![Value::Num(-1.0)], "RunMat:fread:InvalidInput"),
        (
            vec![Value::Num(1.0), Value::from("bad_precision")],
            "RunMat:fread:InvalidOption",
        ),
        (
            vec![Value::Num(1.0), Value::from("uint8"), Value::Num(-1.0)],
            "RunMat:fread:InvalidOption",
        ),
        (
            vec![
                Value::Num(1.0),
                Value::from("uint8"),
                Value::Num(0.0),
                Value::from("vaxd"),
            ],
            "RunMat:fread:InvalidOption",
        ),
    ];
    for (arguments, identifier) in invalid_calls {
        let mut call = Box::pin(evaluate(&fid, &arguments));
        let error = match poll_once(call.as_mut()) {
            std::task::Poll::Ready(Err(error)) => error,
            other => {
                panic!("known-invalid control must reject before provider download: {other:?}")
            }
        };
        assert_eq!(error.identifier(), Some(identifier));
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        drop(call);
    }
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn write_only_target_precedes_registered_resident_size_download() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let path = unique_path("fread_write_only_counted");
    let opened = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("w"),
    ])
    .expect("write-only fopen");
    let fid = opened.as_open().expect("file form").fid;
    let provider = Box::leak(Box::new(PendingControlProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let allocation = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("resident size");
    let arguments = [Value::GpuTensor(allocation.handle.clone())];
    let fid_value = Value::Num(fid);
    let mut call = Box::pin(evaluate(&fid_value, &arguments));
    let error = match poll_once(call.as_mut()) {
        std::task::Poll::Ready(Err(error)) => error,
        other => panic!("write-only target must reject before download: {other:?}"),
    };
    assert_eq!(error.identifier(), Some("RunMat:fread:InvalidIdentifier"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(call);
    run_fclose(&[Value::Num(fid)]).expect("close");
    test_support::fs::remove_file(path).expect("cleanup");
}

#[test]
fn write_only_file_rejects_resident_size_before_download() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fread_write_only_preflight");
    let opened = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("w"),
    ])
    .expect("write-only fopen");
    let fid = opened.as_open().expect("file form").fid;
    let resident = TestAllocationOwner::new()
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("resident size");
    let _no_provider = runmat_accelerate_api::ThreadProviderGuard::set(None);
    let error = run_evaluate(
        &Value::Num(fid),
        &[Value::GpuTensor(resident.handle.clone())],
    )
    .expect_err("write-only target precedes resident size");
    assert_eq!(error.identifier(), Some("RunMat:fread:InvalidIdentifier"));
    assert_eq!(resident.probe.retirements(), 0);
    run_fclose(&[Value::Num(fid)]).expect("close");
    test_support::fs::remove_file(path).expect("cleanup");
}
