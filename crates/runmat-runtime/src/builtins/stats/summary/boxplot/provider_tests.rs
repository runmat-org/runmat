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

fn categorical_group(codes: Vec<f64>, with_categories: bool) -> Value {
    let mut object =
        runmat_value::ObjectInstance::new(runmat_types::standard::CATEGORICAL.to_string());
    let len = codes.len();
    object.replace_property_unbounded(
        "Codes".to_string(),
        Value::Tensor(Tensor::new(codes, vec![len, 1]).unwrap()),
    );
    if with_categories {
        object.replace_property_unbounded(
            "Categories".to_string(),
            Value::StringArray(
                runmat_value::StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap(),
            ),
        );
    }
    Value::Object(object)
}

#[test]
fn host_invalid_controls_and_geometry_reject_before_download() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
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
    let short_group = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let logical_scalar = provider
        .owner
        .allocate_logical(vec![1], vec![1, 1])
        .unwrap();
    let logical_short = provider
        .owner
        .allocate_logical(vec![0, 1], vec![2, 1])
        .unwrap();
    let calls = [
        vec![
            Value::GpuTensor(resident.handle.clone()),
            Value::String("Whisker".into()),
            Value::Num(-1.0),
        ],
        vec![
            Value::GpuTensor(resident.handle.clone()),
            Value::String("OutlierSize".into()),
            Value::Num(f64::MAX),
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
            sample(),
            Value::String("Notch".into()),
            Value::GpuTensor(option.handle.clone()),
        ],
        vec![
            sample(),
            Value::String("FullFactors".into()),
            Value::GpuTensor(option.handle.clone()),
        ],
        vec![
            sample(),
            Value::String("ColorGroup".into()),
            Value::Cell(
                CellArray::new(vec![Value::GpuTensor(option.handle.clone())], 1, 1).unwrap(),
            ),
            Value::String("Jitter".into()),
            Value::Num(-1.0),
        ],
        vec![
            Value::String("invalid sample".into()),
            Value::GpuTensor(resident.handle.clone()),
        ],
        vec![
            Value::GpuTensor(resident.handle.clone()),
            Value::Cell(
                CellArray::new(
                    vec![Value::String("A".into()), Value::String("B".into())],
                    1,
                    2,
                )
                .unwrap(),
            ),
        ],
        vec![
            Value::GpuTensor(short_group.handle.clone()),
            Value::StringArray(
                runmat_value::StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap(),
            ),
            Value::from("FactorSeparator"),
            Value::Num(2.0),
        ],
        vec![
            Value::GpuTensor(short_group.handle.clone()),
            Value::StringArray(
                runmat_value::StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap(),
            ),
            Value::from("FactorSeparator"),
            Value::Num(0.0),
        ],
        vec![
            Value::GpuTensor(short_group.handle.clone()),
            Value::StringArray(
                runmat_value::StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap(),
            ),
            Value::from("FactorGap"),
            Value::Num(-1.0),
        ],
        vec![
            Value::GpuTensor(short_group.handle.clone()),
            Value::StringArray(
                runmat_value::StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap(),
            ),
            Value::from("FactorGap"),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        ],
        vec![
            Value::GpuTensor(short_group.handle.clone()),
            Value::StringArray(
                runmat_value::StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap(),
            ),
            Value::from("FactorDirection"),
            Value::StringArray(
                runmat_value::StringArray::new(vec!["data".into(), "list".into()], vec![1, 2])
                    .unwrap(),
            ),
        ],
        vec![
            Value::GpuTensor(resident.handle.clone()),
            categorical_group(vec![1.0, 2.0, 1.0, 2.0, 1.0], true),
        ],
        vec![
            Value::GpuTensor(resident.handle.clone()),
            Value::String("ColorGroup".into()),
            Value::Complex(1.0, 1.0),
        ],
        vec![
            sample(),
            Value::Cell(
                CellArray::new(
                    vec![
                        Value::GpuTensor(option.handle.clone()),
                        Value::Complex(1.0, 1.0),
                    ],
                    1,
                    2,
                )
                .unwrap(),
            ),
        ],
        vec![
            Value::GpuTensor(resident.handle.clone()),
            Value::Cell(
                CellArray::new(vec![Value::GpuTensor(short_group.handle.clone())], 1, 1).unwrap(),
            ),
        ],
        vec![
            Value::GpuTensor(resident.handle.clone()),
            Value::Cell(
                CellArray::new(vec![Value::GpuTensor(logical_short.handle.clone())], 1, 1).unwrap(),
            ),
        ],
        vec![
            Value::GpuTensor(resident.handle.clone()),
            Value::Cell(
                CellArray::new(
                    vec![
                        categorical_group(vec![1.0, 2.0, 1.0], true),
                        Value::GpuTensor(resident.handle.clone()),
                    ],
                    1,
                    2,
                )
                .unwrap(),
            ),
        ],
        vec![
            sample(),
            Value::String("ColorGroup".into()),
            Value::Cell(
                CellArray::new(
                    vec![
                        Value::GpuTensor(option.handle.clone()),
                        Value::Complex(1.0, 1.0),
                    ],
                    1,
                    2,
                )
                .unwrap(),
            ),
        ],
    ];
    for args in calls {
        let err = block_on(boxplot_builtin(args)).expect_err("known invalid host input");
        assert_eq!(err.identifier(), BOXPLOT_ERROR_INVALID_ARGUMENT.identifier);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        resident.handle.allocation_lease().validate_live().unwrap();
        option.handle.allocation_lease().validate_live().unwrap();
        short_group
            .handle
            .allocation_lease()
            .validate_live()
            .unwrap();
        logical_scalar
            .handle
            .allocation_lease()
            .validate_live()
            .unwrap();
        logical_short
            .handle
            .allocation_lease()
            .validate_live()
            .unwrap();
        assert_eq!(resident.probe.retirements(), 0);
        assert_eq!(option.probe.retirements(), 0);
        assert_eq!(short_group.probe.retirements(), 0);
        assert_eq!(logical_scalar.probe.retirements(), 0);
        assert_eq!(logical_short.probe.retirements(), 0);
    }

    let scalar_labels = Value::Cell(
        CellArray::new(
            vec![
                Value::GpuTensor(option.handle.clone()),
                Value::GpuTensor(logical_scalar.handle.clone()),
                Value::String("c".into()),
                Value::String("d".into()),
            ],
            4,
            1,
        )
        .unwrap(),
    );
    block_on(admission::admit_host_visible(
        &sample(),
        Some(&scalar_labels),
        &mut cooperation::BoxplotCooperation::new(None),
    ))
    .expect("a resident scalar among four scalar labels remains admissible");
    let row_factor =
        Value::Cell(CellArray::new(vec![Value::GpuTensor(resident.handle.clone())], 1, 1).unwrap());
    block_on(admission::admit_host_visible(
        &Value::Tensor(Tensor::new(vec![1.0; 8], vec![4, 2]).unwrap()),
        Some(&row_factor),
        &mut cooperation::BoxplotCooperation::new(None),
    ))
    .expect("a four-row factor can group a four-by-two matrix");
    let categorical_factor = Value::Cell(
        CellArray::new(
            vec![
                categorical_group(vec![1.0, 2.0, 1.0, 2.0], true),
                Value::GpuTensor(resident.handle.clone()),
            ],
            1,
            2,
        )
        .unwrap(),
    );
    block_on(admission::admit_host_visible(
        &sample(),
        Some(&categorical_factor),
        &mut cooperation::BoxplotCooperation::new(None),
    ))
    .expect("valid host categorical factors remain admissible beside resident factors");
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}

#[test]
fn malformed_host_categorical_keeps_table_domain_cause_without_download() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloads::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let error = block_on(boxplot_builtin(vec![
        Value::GpuTensor(resident.handle.clone()),
        categorical_group(vec![1.0, 2.0, 1.0, 2.0], false),
    ]))
    .expect_err("missing categorical categories reject at admission");
    assert_eq!(
        error.identifier(),
        BOXPLOT_ERROR_INVALID_ARGUMENT.identifier
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    let source = error
        .source
        .as_ref()
        .and_then(|source| source.downcast_ref::<RuntimeError>())
        .expect("original table-domain categorical error remains source");
    assert_eq!(error.span, source.span);
    assert_eq!(error.gpu_gather_retry, source.gpu_gather_retry);
    resident.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(resident.probe.retirements(), 0);
}

#[test]
fn nested_resident_group_is_compatibility_gated_before_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloads::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let group = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let nested =
        Value::Cell(CellArray::new(vec![Value::GpuTensor(group.handle.clone())], 1, 1).unwrap());
    let err = block_on(boxplot_builtin(vec![sample(), nested]))
        .expect_err("nested resident group is a RunMat-only extension");
    assert_eq!(
        err.identifier(),
        BOXPLOT_GPU_INPUT_EXTENSION.error_identifier
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    group.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(group.probe.retirements(), 0);
}

#[test]
fn gather_failure_retains_cause_and_caller_lease() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
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
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
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

#[test]
fn registered_cancelled_nested_host_preflight_never_downloads_borrowed_sample() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloads::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(
            vec![128, 1],
            NumericElementType::F64,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let labels = Value::Cell(
        CellArray::new(
            vec![Value::StringArray(
                runmat_value::StringArray::new(
                    (0..128).map(|index| format!("g{index}")).collect(),
                    vec![128, 1],
                )
                .unwrap(),
            )],
            1,
            1,
        )
        .unwrap(),
    );
    let source = runmat_execution::cancellation::CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    source.cancel(runmat_execution::CancellationReason::User);
    let error = block_on(boxplot_builtin_registered(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::GpuTensor(resident.handle.clone()),
            labels,
        ]),
    ))
    .expect_err("cancelled nested group admission must stop before resident transfer");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    resident.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(resident.probe.retirements(), 0);
}
