use super::super::entrypoints::fillmissing_builtin;
use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostNumericTensorView, HostTensorView,
    NumericElementType, ThreadProviderGuard,
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Default)]
struct RejectingTransfers {
    owner: TestAllocationOwner,
    uploads: AtomicUsize,
    downloads: AtomicUsize,
}

impl AccelProvider for RejectingTransfers {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "fillmissing preflight transfer witness".into()
    }

    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        anyhow::bail!("invalid fillmissing call attempted upload")
    }

    fn upload_numeric_raw(
        &self,
        _: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        Err(anyhow::anyhow!("invalid fillmissing call attempted numeric upload").into())
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("invalid fillmissing call attempted download") })
    }

    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("invalid fillmissing call attempted numeric download") })
    }
}

#[test]
fn registered_resident_invalid_controls_precede_all_provider_transfers() {
    let _guard = test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectingTransfers::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _thread = ThreadProviderGuard::set(Some(provider));
    let allocation = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let selected = runmat_accelerate_api::provider_for_handle(&allocation.handle).unwrap();
    assert!(runmat_accelerate_api::same_provider_instance(
        selected, provider
    ));
    for rest in [
        vec![Value::from("constant"), Value::from("not-numeric")],
        vec![Value::from("previous"), Value::Num(0.0)],
        vec![Value::from("bogus")],
    ] {
        let error = block_on(fillmissing_builtin(
            None,
            Value::GpuTensor(allocation.handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
        .expect_err("host-visible invalid call");
        assert_eq!(error.identifier(), Some("RunMat:missing:InvalidArgument"));
    }
    let complex = provider
        .owner
        .allocate_zeroed(
            vec![2, 1],
            NumericElementType::F64,
            GpuTensorStorage::ComplexInterleaved,
        )
        .unwrap();
    let error = block_on(fillmissing_builtin(
        None,
        Value::GpuTensor(complex.handle.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("constant"),
            Value::Num(0.0),
        ]),
    ))
    .expect_err("known complex representation must be rejected before download");
    assert_eq!(error.identifier(), Some("RunMat:missing:UnsupportedType"));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(allocation.probe.retirements(), 0);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
    assert_eq!(complex.probe.retirements(), 0);
    complex.handle.release().unwrap();
    assert_eq!(complex.probe.retirements(), 1);
}

#[test]
fn integer_noop_still_validates_dimension_and_direct_scalar_tensor_dimension() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let integer = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, u64::MAX]), vec![2, 1]).unwrap(),
    );
    let error = block_on(fillmissing_builtin(
        None,
        integer,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("constant"),
            Value::Num(0.0),
            Value::Num(0.0),
        ]),
    ))
    .expect_err("zero axis is invalid even if no values are missing");
    assert_eq!(error.identifier(), Some("RunMat:missing:InvalidArgument"));

    let input = tensor(vec![1.0, f64::NAN, 3.0], vec![1, 3]);
    let axis =
        Value::Tensor(Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1]).unwrap());
    let result = block_on(fillmissing_builtin(
        None,
        input,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("linear"), axis]),
    ))
    .unwrap();
    assert!(
        matches!(result, Value::Tensor(output) if output.materialize_f64() == vec![1.0, 2.0, 3.0])
    );
}
