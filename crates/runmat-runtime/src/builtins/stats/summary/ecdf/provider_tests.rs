//! Registered-owner witnesses for empirical statistics and graphics admission.
use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle,
    GpuTensorStorage, HostNumericDataOwned, HostNumericDataView, HostNumericTensorOwned,
    HostNumericTensorView, HostTensorView, NumericElementType, ThreadProviderGuard,
};
#[cfg(not(feature = "plot-core"))]
use runmat_value::LogicalArray;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

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
        "empirical distribution owner witness".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("admission must not upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            DownloadMode::Fail => Box::pin(async {
                Err(std::io::Error::other("empirical fixture rejected download").into())
            }),
            DownloadMode::Pending => Box::pin(futures::future::pending()),
        }
    }
    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            DownloadMode::Fail => Box::pin(async {
                Err(std::io::Error::other("empirical fixture rejected download").into())
            }),
            DownloadMode::Pending => Box::pin(futures::future::pending()),
        }
    }
}

fn assert_original_cause(error: &RuntimeError, identifier: &str) {
    assert_eq!(error.identifier(), Some(identifier));
    let gather = error
        .source
        .as_ref()
        .and_then(|source| source.downcast_ref::<RuntimeError>())
        .expect("typed gather error retained");
    assert_eq!(gather.identifier(), Some("RunMat:gather:DownloadFailed"));
    assert_eq!(error.span, gather.span);
    assert_eq!(error.gpu_gather_retry, gather.gpu_gather_retry);
    let mut source: &(dyn std::error::Error + 'static) = error;
    while !source.is::<std::io::Error>() {
        source = source.source().expect("provider source retained");
    }
    assert_eq!(source.to_string(), "empirical fixture rejected download");
}

#[test]
fn known_invalid_ecdf_controls_and_cdfplot_shape_precede_downloads() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(WitnessProvider::new(DownloadMode::Fail)));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let allocation = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let resident = || Value::GpuTensor(allocation.handle.clone());
    assert!(runmat_accelerate_api::same_provider_instance(
        runmat_accelerate_api::provider_for_handle(&allocation.handle).unwrap(),
        provider
    ));
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);

    let error = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            resident(),
            Value::from("Function"),
            Value::from("sideways"),
        ]),
    ))
    .unwrap_err();
    assert!(error.message().contains("unsupported Function"), "{error}");
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);

    let error = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            resident(),
            Value::from("Frequency"),
            resident(),
            Value::from("Bounds"),
            Value::from("sideways"),
        ]),
    ))
    .unwrap_err();
    assert!(error.message().contains("unsupported Bounds"), "{error}");
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);

    let error = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            resident(),
            Value::from("Frequency"),
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![2, 1]).unwrap()),
        ]),
    ))
    .unwrap_err();
    assert!(
        error.message().contains("same number of elements"),
        "{error}"
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);

    let error = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            resident(),
            resident(),
            Value::Num(1.0),
        ]),
    ))
    .unwrap_err();
    assert!(
        error.message().contains("option names must be text"),
        "{error}"
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);

    let malformed = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let error = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
            malformed.handle.clone(),
        )]),
    ))
    .unwrap_err();
    assert!(error.message().contains("interval-censored"), "{error}");
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    let error = block_on(cdfplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
            malformed.handle.clone(),
        )]),
    ))
    .unwrap_err();
    assert!(error.message().contains("expected a vector"), "{error}");
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    malformed.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(malformed.probe.retirements(), 0);
    let complex = provider
        .owner
        .allocate_zeroed(
            vec![4, 1],
            NumericElementType::F64,
            GpuTensorStorage::ComplexInterleaved,
        )
        .unwrap();
    let error = block_on(cdfplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
            complex.handle.clone(),
        )]),
    ))
    .unwrap_err();
    assert!(error.message().contains("expected real numeric"), "{error}");
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    complex.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(complex.probe.retirements(), 0);
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn gather_errors_preserve_owner_cause_and_borrowed_lease() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(WitnessProvider::new(DownloadMode::Fail)));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let allocation = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let resident = || Value::GpuTensor(allocation.handle.clone());

    let error = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![resident()]),
    ))
    .unwrap_err();
    assert_original_cause(&error, "RunMat:ecdf:InvalidArgument");
    #[cfg(feature = "plot-core")]
    {
        let error = block_on(cdfplot_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![resident()]),
        ))
        .unwrap_err();
        assert_original_cause(&error, "RunMat:cdfplot:InvalidArgument");
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 2);
    }
    #[cfg(not(feature = "plot-core"))]
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn abandoned_unpolled_and_pending_calls_retain_borrowed_input() {
    use std::future::Future;
    use std::task::Context;

    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(WitnessProvider::new(DownloadMode::Pending)));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let allocation = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();

    let unpolled = ecdf_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::GpuTensor(allocation.handle.clone())],
    ));
    drop(unpolled);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    let mut pending = Box::pin(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
            allocation.handle.clone(),
        )]),
    ));
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    assert!(pending.as_mut().poll(&mut context).is_pending());
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    drop(pending);
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    assert_eq!(allocation.probe.retirements(), 0);
}

#[cfg(not(feature = "plot-core"))]
#[test]
fn unavailable_ecdf_statement_form_does_not_download_resident_inputs() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(WitnessProvider::new(DownloadMode::Fail)));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let allocation = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let invalid = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::GpuTensor(allocation.handle.clone()),
            Value::from("Alpha"),
            Value::Num(2.0),
        ]),
    ))
    .expect_err("known invalid host control must precede missing presentation");
    assert_eq!(invalid.identifier(), Some("RunMat:ecdf:InvalidArgument"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    let invalid_sample = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("not numeric"),
            Value::from("Frequency"),
            Value::GpuTensor(allocation.handle.clone()),
        ]),
    ))
    .expect_err("known invalid host sample must precede missing presentation");
    assert_eq!(
        invalid_sample.identifier(),
        Some("RunMat:ecdf:InvalidArgument")
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    for (data, shape) in [(vec![], vec![0, 1]), (vec![1; 6], vec![2, 3])] {
        let logical = Value::LogicalArray(LogicalArray::new(data, shape).unwrap());
        let error = block_on(ecdf_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                logical,
                Value::from("Frequency"),
                Value::GpuTensor(allocation.handle.clone()),
            ]),
        ))
        .expect_err("known invalid logical shape must precede missing presentation");
        assert_eq!(error.identifier(), Some("RunMat:ecdf:InvalidArgument"));
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    }
    let error = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
            allocation.handle.clone(),
        )]),
    ))
    .expect_err("plotting is unavailable without plot-core");
    assert!(error
        .source
        .as_deref()
        .and_then(|source| source.downcast_ref::<crate::plotting_hooks::PlottingUnavailable>())
        .is_some());
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    assert_eq!(allocation.probe.retirements(), 0);
}

struct SiblingAliasProvider {
    owner: TestAllocationOwner,
    retained: Mutex<Vec<GpuTensorHandle>>,
    borrowed_alias: Mutex<Option<GpuTensorHandle>>,
    probes: Mutex<Vec<RetirementProbe>>,
    uploads: AtomicUsize,
}

impl AccelProvider for SiblingAliasProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "ecdf sibling alias witness".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("typed upload required")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("output staging must not download") })
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            let shape = handle.shape().to_vec();
            let count = shape.iter().product();
            Ok(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(vec![1.0; count]),
                shape,
                storage: GpuTensorStorage::Real,
            })
        })
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let index = self.uploads.fetch_add(1, Ordering::SeqCst);
        let mut retained = self.retained.lock().unwrap();
        if index == 1 {
            return Ok(self
                .borrowed_alias
                .lock()
                .unwrap()
                .as_ref()
                .unwrap_or_else(|| retained.first().expect("first staged output"))
                .clone());
        }
        let HostNumericDataView::F64(data) = host.data else {
            panic!("expected double outputs")
        };
        let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(data.to_vec()),
            shape: host.shape.to_vec(),
            storage: host.storage,
        })?;
        self.probes.lock().unwrap().push(allocation.probe);
        retained.push(allocation.handle.clone());
        Ok(allocation.handle)
    }
}

#[test]
fn second_output_alias_is_rejected_without_retiring_borrowed_source() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(SiblingAliasProvider {
        owner: TestAllocationOwner::new(),
        retained: Mutex::new(Vec::new()),
        borrowed_alias: Mutex::new(None),
        probes: Mutex::new(Vec::new()),
        uploads: AtomicUsize::new(0),
    }));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let source = provider
        .owner
        .allocate_zeroed(vec![3, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let outputs = vec![
        Value::Tensor(Tensor::new(vec![0.0, 0.5, 1.0], vec![3, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap()),
    ];
    let error = restore_ecdf_outputs(outputs, Some(&source.handle), &[]).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:ecdf:Internal"));
    assert!(error.message().contains("alias"), "{error}");
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 2);
    source.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(source.probe.retirements(), 0);
    let probes = provider.probes.lock().unwrap().clone();
    assert_eq!(probes.len(), 1);
    provider.retained.lock().unwrap().clear();
    assert_eq!(probes[0].retirements(), 1);
}

#[test]
fn resident_frequency_alias_on_second_output_preserves_borrowed_control_and_cleans_first() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(SiblingAliasProvider {
        owner: TestAllocationOwner::new(),
        retained: Mutex::new(Vec::new()),
        borrowed_alias: Mutex::new(None),
        probes: Mutex::new(Vec::new()),
        uploads: AtomicUsize::new(0),
    }));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let sample = provider
        .owner
        .allocate_zeroed(vec![3, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let frequency = provider
        .owner
        .allocate_zeroed(vec![3, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    *provider.borrowed_alias.lock().unwrap() = Some(frequency.handle.clone());
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::GpuTensor(sample.handle.clone()),
            Value::from("Frequency"),
            Value::GpuTensor(frequency.handle.clone()),
        ]),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:ecdf:Internal"));
    assert!(error.message().contains("alias"), "{error}");
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 2);
    sample.handle.allocation_lease().validate_live().unwrap();
    frequency.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(sample.probe.retirements(), 0);
    assert_eq!(frequency.probe.retirements(), 0);
    let probes = provider.probes.lock().unwrap().clone();
    assert_eq!(probes.len(), 1);
    provider.retained.lock().unwrap().clear();
    provider.borrowed_alias.lock().unwrap().take();
    assert_eq!(probes[0].retirements(), 1);
}
