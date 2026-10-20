use super::admission::OptionInput;
use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelIntegerDownloadFuture, AccelNumericDownloadFuture, AccelProvider,
    GpuTensorHandle, GpuTensorStorage, HostIntegerTensorView, HostNumericTensorView,
    HostTensorView, NumericElementType,
};
use runmat_value::Tensor;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[test]
fn borrowed_handle_capacity_failure_keeps_operational_source() {
    use std::error::Error;

    let mut handles = Vec::<u8>::new();
    let error = reserve_borrowed(
        &mut handles,
        usize::MAX,
        "kmeans: borrowed operands exceed available memory",
    )
    .expect_err("capacity overflow must reject before allocation");
    assert_eq!(error.identifier(), Some("RunMat:kmeans:Internal"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(error
        .source()
        .expect("original reservation failure")
        .is::<std::collections::TryReserveError>());
}

enum UploadBehavior {
    Alias(GpuTensorHandle),
    FailSecond,
    Unsupported,
}

struct RestoreProvider {
    owner: TestAllocationOwner,
    behavior: Mutex<UploadBehavior>,
    uploads: AtomicUsize,
    staged: Mutex<Option<RetirementProbe>>,
}

impl RestoreProvider {
    fn new(behavior: UploadBehavior, owner: TestAllocationOwner) -> &'static Self {
        Box::leak(Box::new(Self {
            owner,
            behavior: Mutex::new(behavior),
            uploads: AtomicUsize::new(0),
            staged: Mutex::new(None),
        }))
    }
}

impl AccelProvider for RestoreProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "kmeans restoration fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("typed upload required")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("restoration cannot download") })
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let attempt = self.uploads.fetch_add(1, Ordering::SeqCst);
        match &*self.behavior.lock().unwrap() {
            UploadBehavior::Alias(handle) => Ok(handle.clone()),
            UploadBehavior::Unsupported => Err(
                runmat_accelerate_api::unsupported_provider_operation("fixture has no f64 upload")
                    .into(),
            ),
            UploadBehavior::FailSecond if attempt > 0 => {
                Err(anyhow::anyhow!("second output failed").into())
            }
            UploadBehavior::FailSecond => {
                let result = self.owner.allocate_zeroed(
                    host.shape.to_vec(),
                    host.data.element_type(),
                    host.storage,
                )?;
                *self.staged.lock().unwrap() = Some(result.probe);
                Ok(result.handle)
            }
        }
    }
}

struct CountingProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    downloads: AtomicUsize,
}

impl CountingProvider {
    fn new() -> &'static Self {
        Box::leak(Box::new(Self {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            downloads: AtomicUsize::new(0),
        }))
    }
}

impl AccelProvider for CountingProvider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.inner.upload_numeric_raw(host)
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download_numeric(handle)
    }
    fn upload_integer(&self, host: &HostIntegerTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload_integer(host)
    }
    fn download_integer<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelIntegerDownloadFuture<'a> {
        self.inner.download_integer(handle)
    }
}

#[test]
fn resident_integer_x_and_start_each_download_once() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountingProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let x = Tensor::new_integer(
        runmat_value::IntegerStorage::U64(vec![0, 1, 9, 10]),
        vec![4, 1],
    )
    .unwrap();
    let start =
        Tensor::new_integer(runmat_value::IntegerStorage::U64(vec![0, 10]), vec![2, 1]).unwrap();
    let x = crate::builtins::common::gpu_helpers::upload_tensor(provider, &x).unwrap();
    let start = crate::builtins::common::gpu_helpers::upload_tensor(provider, &start).unwrap();
    block_on(kmeans_builtin(
        Value::GpuTensor(x),
        Value::Num(2.0),
        vec![Value::from("Start"), Value::GpuTensor(start)],
    ))
    .expect("resident typed integer clustering");
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 2);
}

#[test]
fn invalid_text_option_with_resident_x_does_not_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountingProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let x = Tensor::new(vec![0.0, 1.0, 9.0, 10.0], vec![4, 1]).unwrap();
    let x = crate::builtins::common::gpu_helpers::upload_tensor(provider, &x).unwrap();
    let err = block_on(kmeans_builtin(
        Value::GpuTensor(x),
        Value::Num(2.0),
        vec![Value::from("Distance"), Value::Num(1.0)],
    ))
    .unwrap_err();
    assert!(err.message().contains("Distance must be text"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}

#[test]
fn invalid_host_numeric_start_rejects_before_resident_x_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountingProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let x = Tensor::new(vec![0.0, 1.0, 9.0, 10.0], vec![4, 1]).unwrap();
    let x = crate::builtins::common::gpu_helpers::upload_tensor(provider, &x).unwrap();
    for invalid_value in [f64::NAN, f64::INFINITY] {
        let start = Tensor::new(vec![0.0, invalid_value], vec![2, 1]).unwrap();
        let error = block_on(kmeans_builtin(
            Value::GpuTensor(x.clone()),
            Value::Num(2.0),
            vec![Value::from("Start"), Value::Tensor(start)],
        ))
        .expect_err("known invalid Start values must reject without provider access");
        assert!(error
            .message()
            .contains("numeric Start values must be finite"));
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn registered_known_start_shape_rejects_before_resident_x_download_and_keeps_lease() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountingProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let x = Tensor::new(vec![0.0, 1.0, 9.0, 10.0, 0.0, 1.0, 9.0, 10.0], vec![4, 2]).unwrap();
    let x = crate::builtins::common::gpu_helpers::upload_tensor(provider, &x).unwrap();
    for (k, start) in [
        (
            Value::Num(2.0),
            Tensor::new(vec![0.0; 6], vec![2, 3]).unwrap(),
        ),
        (
            Value::Num(3.0),
            Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap(),
        ),
    ] {
        let error = block_on(kmeans_builtin_registered(
            None,
            Value::GpuTensor(x.clone()),
            k,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("Start"),
                Value::Tensor(start),
            ]),
        ))
        .expect_err("known k-by-p mismatch must reject before download");
        assert!(error.message().contains("numeric Start must be k-by-p"));
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        assert!(x.allocation_lease().validate_live().is_ok());
    }
}

#[test]
fn effective_start_shape_uses_last_override_and_vector_feature_geometry() {
    let x = Tensor::new(vec![0.0, 1.0, 9.0, 10.0], vec![1, 4]).unwrap();
    let options = admission::admit_option_values(
        admission::parse_option_headers(crate::builtins::common::test_support::builtin_arguments(
            vec![
                Value::from("Start"),
                Value::Tensor(Tensor::new(vec![0.0; 6], vec![2, 3]).unwrap()),
                Value::from("Start"),
                Value::Tensor(Tensor::new(vec![0.0, 10.0], vec![2, 1]).unwrap()),
            ],
        ))
        .unwrap(),
    )
    .unwrap();
    geometry::preflight_effective_start(&Value::Tensor(x), &Value::Num(2.0), &options)
        .expect("the last vector-shaped Start overrides the earlier matrix");
}

#[test]
fn ignored_resident_statset_field_does_not_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountingProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let ignored = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let ignored = crate::builtins::common::gpu_helpers::upload_tensor(provider, &ignored).unwrap();
    let mut statset = runmat_value::UnboundedStructValueBuilder::new();
    statset.insert("LegacyField", Value::GpuTensor(ignored));
    block_on(kmeans_builtin(
        Value::Tensor(Tensor::new(vec![0.0, 1.0, 9.0, 10.0], vec![4, 1]).unwrap()),
        Value::Num(2.0),
        vec![
            Value::from("Start"),
            Value::Tensor(Tensor::new(vec![0.0, 10.0], vec![2, 1]).unwrap()),
            Value::from("Options"),
            Value::Struct(statset.finish()),
        ],
    ))
    .expect("ignored statset field cannot access provider");
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}

#[test]
fn typed_headers_preserve_order_and_skip_unrecognized_statset_payloads() {
    let owner = TestAllocationOwner::new();
    let ignored = owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let mut statset = runmat_value::UnboundedStructValueBuilder::new();
    statset.insert("LegacyField", Value::GpuTensor(ignored.handle.clone()));
    let headers = admission::parse_option_headers(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Distance"),
            Value::from("cityblock"),
            Value::from("Options"),
            Value::Struct(statset.finish()),
            Value::from("Distance"),
            Value::from("cosine"),
        ]),
    )
    .unwrap();
    let options = admission::admit_option_values(headers).unwrap();
    assert_eq!(options.len(), 3);
    assert!(matches!(
        options[0],
        OptionInput::Distance(Distance::Cityblock)
    ));
    assert!(matches!(options[1], OptionInput::Statset(ref fields) if fields.is_empty()));
    assert!(matches!(
        options[2],
        OptionInput::Distance(Distance::Cosine)
    ));
    assert_eq!(ignored.probe.retirements(), 0);
}

#[test]
fn restoration_rejects_alias_to_non_x_borrowed_start() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let owner = TestAllocationOwner::new();
    let source = owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let start = owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let provider = RestoreProvider::new(UploadBehavior::Alias(start.handle.clone()), owner);
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let mut borrowed = BorrowedOperands::new();
    borrowed
        .protect_value(&Value::GpuTensor(start.handle.clone()))
        .unwrap();
    borrowed
        .record_consumed(&Value::GpuTensor(start.handle.clone()))
        .unwrap();
    let result = restore_kmeans_value(
        Value::Tensor(Tensor::new(vec![1.0; 4], vec![4, 1]).unwrap()),
        Some(&source.handle),
        &borrowed,
    );
    assert!(result
        .unwrap_err()
        .message()
        .contains("aliasing a borrowed input"));
    assert_eq!(start.probe.retirements(), 0);
    assert_eq!(source.probe.retirements(), 0);
}

#[test]
fn ignored_nested_resident_metadata_is_protected_without_inheriting_explicit_intent() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let owner = TestAllocationOwner::new();
    let source = owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let ignored = owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let empty_stream = owner
        .allocate_zeroed(vec![0, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let ignored_explicit = ignored
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .unwrap();
    let provider = RestoreProvider::new(UploadBehavior::Alias(ignored_explicit.clone()), owner);
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let cell = runmat_value::CellArray::new(vec![Value::GpuTensor(ignored_explicit.clone())], 1, 1)
        .unwrap();
    let mut statset = runmat_value::UnboundedStructValueBuilder::new();
    statset.insert("IgnoredField", Value::Cell(cell));
    statset.insert("Streams", Value::GpuTensor(empty_stream.handle.clone()));
    let mut borrowed = BorrowedOperands::new();
    borrowed
        .protect_value(&Value::Struct(statset.finish()))
        .unwrap();
    let result = restore_kmeans_value(
        Value::Tensor(Tensor::new(vec![1.0; 4], vec![4, 1]).unwrap()),
        Some(&source.handle),
        &borrowed,
    );
    assert!(result
        .unwrap_err()
        .message()
        .contains("aliasing a borrowed input"));
    assert_eq!(ignored.probe.retirements(), 0);
    assert_eq!(empty_stream.probe.retirements(), 0);
    assert_eq!(source.probe.retirements(), 0);

    *provider.behavior.lock().unwrap() = UploadBehavior::Alias(empty_stream.handle.clone());
    let empty_alias = restore_kmeans_value(
        Value::Tensor(Tensor::new(vec![1.0; 4], vec![4, 1]).unwrap()),
        Some(&source.handle),
        &borrowed,
    );
    assert!(empty_alias
        .unwrap_err()
        .message()
        .contains("aliasing a borrowed input"));
    assert_eq!(empty_stream.probe.retirements(), 0);

    *provider.behavior.lock().unwrap() = UploadBehavior::Unsupported;
    let host = Value::Tensor(Tensor::new(vec![1.0; 4], vec![4, 1]).unwrap());
    let automatic = source
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        .unwrap();
    let result = restore_kmeans_value(host.clone(), Some(&automatic), &borrowed)
        .expect("ignored explicit metadata cannot force output placement");
    assert_eq!(result, host);
}

#[test]
fn consumed_explicit_scalar_control_prevents_implicit_host_output_fallback() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let owner = TestAllocationOwner::new();
    let source = owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let control = owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let explicit_control = control
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .unwrap();
    let provider = RestoreProvider::new(UploadBehavior::Unsupported, owner);
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let host = Value::Tensor(Tensor::new(vec![1.0; 4], vec![4, 1]).unwrap());

    let ignored_control = BorrowedOperands::new();
    let automatic = source
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        .unwrap();
    let fallback = restore_kmeans_value(host.clone(), Some(&automatic), &ignored_control)
        .expect("unsupported automatic upload permits host fallback");
    assert_eq!(fallback, host);

    let mut consumed_control = BorrowedOperands::new();
    consumed_control
        .protect_value(&Value::GpuTensor(explicit_control.clone()))
        .unwrap();
    consumed_control
        .record_consumed(&Value::GpuTensor(explicit_control))
        .unwrap();
    let error = restore_kmeans_value(host, Some(&source.handle), &consumed_control).unwrap_err();
    assert!(error.message().contains("failed to restore GPU result"));
    assert!(std::error::Error::source(&error).is_some());
    assert_eq!(source.probe.retirements(), 0);
    assert_eq!(control.probe.retirements(), 0);
}

#[test]
fn later_upload_failure_retires_first_staged_output_and_preserves_cause() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let owner = TestAllocationOwner::new();
    let source = owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let provider = RestoreProvider::new(UploadBehavior::FailSecond, owner);
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let values = Value::output_list(vec![
        Value::Tensor(Tensor::new(vec![1.0; 4], vec![4, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![2.0; 2], vec![2, 1]).unwrap()),
    ]);
    let result = restore_kmeans_value(values, Some(&source.handle), &BorrowedOperands::new());
    let error = result.unwrap_err();
    assert!(error.message().contains("second output failed"));
    assert!(std::error::Error::source(&error).is_some());
    assert_eq!(
        provider
            .staged
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .retirements(),
        1
    );
    assert_eq!(source.probe.retirements(), 0);
}

#[test]
fn invalid_text_control_precedes_any_resident_download() {
    let owner = TestAllocationOwner::new();
    let resident = owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let headers = admission::parse_option_headers(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Distance"),
            Value::GpuTensor(resident.handle.clone()),
        ]),
    )
    .unwrap();
    assert!(admission::admit_option_values(headers).is_err());
    assert_eq!(resident.probe.retirements(), 0);
}

#[test]
fn host_wide_start_is_checked_before_any_download() {
    let start = Value::Tensor(
        Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![(1_u64 << 53) + 1]),
            vec![1, 1],
        )
        .unwrap(),
    );
    let headers = admission::parse_option_headers(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("Start"), start]),
    )
    .unwrap();
    assert!(admission::admit_option_values(headers).is_err());
    let aligned = Value::Tensor(
        Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![1_u64 << 54]),
            vec![1, 1],
        )
        .unwrap(),
    );
    let headers =
        admission::parse_option_headers(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::from("Start"), aligned],
        ))
        .unwrap();
    assert!(admission::admit_option_values(headers).is_ok());
}
