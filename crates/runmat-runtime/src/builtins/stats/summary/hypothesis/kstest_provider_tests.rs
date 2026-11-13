//! Exact registered-owner witnesses for KS admission and future lifetime.
use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuHandleProvenance, GpuTensorHandle, GpuTensorStorage, HostTensorView,
    NumericElementType, ThreadProviderGuard,
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy)]
enum DownloadMode {
    Fail,
    Pending,
}

struct WitnessProvider {
    owner: TestAllocationOwner,
    downloads: AtomicUsize,
    mode: DownloadMode,
}

impl WitnessProvider {
    fn new(mode: DownloadMode) -> Self {
        Self {
            owner: TestAllocationOwner::new(),
            downloads: AtomicUsize::new(0),
            mode,
        }
    }
}

impl AccelProvider for WitnessProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "kstest registered owner witness".into()
    }

    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("kstest admission must not upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            DownloadMode::Fail => Box::pin(async {
                Err(std::io::Error::other("kstest fixture rejected download").into())
            }),
            DownloadMode::Pending => Box::pin(futures::future::pending()),
        }
    }

    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            DownloadMode::Fail => Box::pin(async {
                Err(std::io::Error::other("kstest fixture rejected download").into())
            }),
            DownloadMode::Pending => Box::pin(futures::future::pending()),
        }
    }
}

fn assert_owner_cause(error: &RuntimeError) {
    assert_eq!(error.identifier(), KSTEST_ERROR_INVALID_ARGUMENT.identifier);
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let gather = error
        .source
        .as_ref()
        .and_then(|source| source.downcast_ref::<RuntimeError>())
        .expect("typed gather error remains nested");
    assert_eq!(gather.identifier(), Some("RunMat:gather:DownloadFailed"));
    assert_eq!(error.span, gather.span);
    assert_eq!(error.gpu_gather_retry, gather.gpu_gather_retry);
    let mut source: &(dyn std::error::Error + 'static) = error;
    while !source.is::<std::io::Error>() {
        source = source.source().expect("original provider failure retained");
    }
    assert_eq!(source.to_string(), "kstest fixture rejected download");
}

#[test]
fn registered_owner_observes_no_download_for_known_invalid_calls() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(WitnessProvider::new(DownloadMode::Fail)));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let allocation = provider
        .owner
        .allocate_zeroed(vec![3, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let resident = || Value::GpuTensor(allocation.handle.clone());
    assert!(runmat_accelerate_api::same_provider_instance(
        runmat_accelerate_api::provider_for_handle(&allocation.handle).unwrap(),
        provider
    ));

    {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let valid_cdf = provider
            .owner
            .allocate_zeroed(vec![3, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let error = block_on(kstest_builtin(
            resident(),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("CDF"),
                Value::GpuTensor(valid_cdf.handle.clone()),
                Value::from("Tail"),
                Value::from("sideways"),
            ]),
        ))
        .unwrap_err();
        assert!(error.message().contains("unsupported Tail"), "{error}");
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        valid_cdf.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(valid_cdf.probe.retirements(), 0);

        let error = block_on(kstest_builtin(
            resident(),
            crate::builtins::common::test_support::builtin_arguments(vec![
                resident(),
                Value::Num(0.05),
            ]),
        ))
        .unwrap_err();
        assert!(error.message().contains("option name"), "{error}");
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);

        let invalid_resident_cdf = provider
            .owner
            .allocate_zeroed(vec![2, 3], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let error = block_on(kstest_builtin(
            resident(),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("CDF"),
                Value::GpuTensor(invalid_resident_cdf.handle.clone()),
            ]),
        ))
        .unwrap_err();
        assert!(error.message().contains("n-by-2"), "{error}");
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        invalid_resident_cdf
            .handle
            .allocation_lease()
            .validate_live()
            .unwrap();
        assert_eq!(invalid_resident_cdf.probe.retirements(), 0);

        let complex_cdf = provider
            .owner
            .allocate_zeroed(
                vec![2, 2],
                NumericElementType::F64,
                GpuTensorStorage::ComplexInterleaved,
            )
            .unwrap();
        let error = block_on(kstest_builtin(
            resident(),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("CDF"),
                Value::GpuTensor(complex_cdf.handle.clone()),
            ]),
        ))
        .unwrap_err();
        assert!(error.message().contains("real numeric matrix"), "{error}");
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        complex_cdf
            .handle
            .allocation_lease()
            .validate_live()
            .unwrap();
        assert_eq!(complex_cdf.probe.retirements(), 0);

        let complex_alpha = provider
            .owner
            .allocate_zeroed(
                vec![1, 1],
                NumericElementType::F64,
                GpuTensorStorage::ComplexInterleaved,
            )
            .unwrap();
        let error = block_on(kstest_builtin(
            resident(),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("Alpha"),
                Value::GpuTensor(complex_alpha.handle.clone()),
            ]),
        ))
        .unwrap_err();
        assert!(error.message().contains("real numeric"), "{error}");
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        complex_alpha
            .handle
            .allocation_lease()
            .validate_live()
            .unwrap();
        assert_eq!(complex_alpha.probe.retirements(), 0);

        let invalid_cdf = Value::Tensor(Tensor::new(vec![0.0; 6], vec![2, 3]).unwrap());
        let error = block_on(kstest_builtin(
            resident(),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("CDF"),
                invalid_cdf,
            ]),
        ))
        .unwrap_err();
        assert!(error.message().contains("n-by-2"), "{error}");
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);

        let matrix = provider
            .owner
            .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let error = block_on(kstest_builtin(
            Value::GpuTensor(matrix.handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap_err();
        assert!(error.message().contains("must be a vector"), "{error}");
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        matrix.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(matrix.probe.retirements(), 0);

        let _outputs = crate::output_count::push_output_count(Some(5)).expect("output-count scope");
        let error = block_on(kstest_builtin(
            resident(),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap_err();
        assert!(
            error.message().contains("too many output arguments"),
            "{error}"
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    }

    {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer = Value::Tensor(
            Tensor::new_integer(
                runmat_value::IntegerStorage::U8(vec![0, 1, 0, 1]),
                vec![2, 2],
            )
            .unwrap(),
        );
        let error = block_on(kstest_builtin(
            resident(),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("CDF"),
                integer,
            ]),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            KSTEST_INTEGER_CDF_EXTENSION.error_identifier
        );

        let explicit = allocation
            .handle
            .clone()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        let error = block_on(kstest_builtin(
            Value::GpuTensor(explicit),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            KSTEST_RESIDENT_INPUT_EXTENSION.error_identifier
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    }
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn provider_failures_retain_sources_and_borrowed_allocations() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(WitnessProvider::new(DownloadMode::Fail)));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let allocation = provider
        .owner
        .allocate_zeroed(vec![3, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let resident = || Value::GpuTensor(allocation.handle.clone());

    let error = block_on(kstest_builtin(
        resident(),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert_owner_cause(&error);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    assert_eq!(allocation.probe.retirements(), 0);

    let sample = Value::Tensor(Tensor::new(vec![-1.0, 0.0, 1.0], vec![3, 1]).unwrap());
    let cdf = provider
        .owner
        .allocate_zeroed(vec![3, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let error = block_on(kstest_builtin(
        sample,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("CDF"),
            Value::GpuTensor(cdf.handle.clone()),
        ]),
    ))
    .unwrap_err();
    assert_owner_cause(&error);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 2);
    cdf.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(cdf.probe.retirements(), 0);
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn unpolled_and_pending_call_drop_preserve_borrowed_owner() {
    use std::future::Future;
    use std::task::{Context, Poll};

    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(WitnessProvider::new(DownloadMode::Pending)));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let allocation = provider
        .owner
        .allocate_zeroed(vec![3, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let resident = || Value::GpuTensor(allocation.handle.clone());

    let unpolled = kstest_builtin(
        resident(),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    );
    drop(unpolled);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();

    let mut pending = Box::pin(kstest_builtin(
        resident(),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ));
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    assert!(matches!(pending.as_mut().poll(&mut context), Poll::Pending));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    drop(pending);
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    assert_eq!(allocation.probe.retirements(), 0);
}
