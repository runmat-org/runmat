use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView, NumericElementType,
    ThreadProviderGuard,
};
use runmat_value::{CellArray, LogicalArray, StringArray};
use std::sync::atomic::{AtomicUsize, Ordering};

struct RejectingDownloadProvider {
    owner: TestAllocationOwner,
    downloads: AtomicUsize,
}

#[tokio::test]
async fn discarded_classify_admits_before_provider_access() {
    let provider = Box::leak(Box::new(RejectingDownloadProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let resident = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let (_, training, group) = valid_floating_inputs();

    let err = classify_builtin(
        Value::GpuTensor(resident.handle.clone()),
        training.clone(),
        group.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("bogus")]),
    )
    .await
    .expect_err("discarding outputs must not bypass type admission");
    assert!(err.message().contains("type"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);

    let err = classify_builtin(
        Value::GpuTensor(resident.handle.clone()),
        training.clone(),
        group.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("linear"),
            tensor(vec![f64::NAN, 1.0], vec![1, 2]),
        ]),
    )
    .await
    .expect_err("discarding outputs must not bypass host prior admission");
    assert!(err.message().contains("prior probabilities"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);

    let result = classify_builtin(
        Value::GpuTensor(resident.handle.clone()),
        training,
        group,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .await
    .expect("valid discarded call needs no download or prediction buffers");
    assert!(matches!(result, Value::OutputList(outputs) if outputs.is_empty()));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(resident.probe.retirements(), 0);
    drop(resident.handle);
    assert_eq!(resident.probe.retirements(), 1);
}

impl AccelProvider for RejectingDownloadProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "classify gather fixture".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("classify fixture download failure") })
    }

    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("classify fixture native download failure") })
    }
}

#[tokio::test]
async fn classify_preflight_and_gather_error_preserve_provider_contract() {
    let provider = Box::leak(Box::new(RejectingDownloadProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let resident = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let (_, training, group) = valid_floating_inputs();
    let mut prior = runmat_value::UnboundedStructValueBuilder::new();
    prior.insert("group", group.clone());
    let err = classify_builtin(
        Value::GpuTensor(resident.handle.clone()),
        training.clone(),
        group.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("linear"),
            Value::Struct(prior.finish()),
        ]),
    )
    .await
    .expect_err("missing prior probability must reject before download");
    assert!(err
        .message()
        .contains("prior struct must contain prob field"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);

    let invalid_group = Value::Cell(
        CellArray::new(
            vec![
                Value::Num(1.0),
                Value::Num(1.0),
                Value::Num(2.0),
                Value::Num(2.0),
            ],
            4,
            1,
        )
        .unwrap(),
    );
    let err = classify_builtin(
        Value::GpuTensor(resident.handle.clone()),
        training.clone(),
        invalid_group,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .await
    .expect_err("host cell labels must reject before resident sample download");
    assert!(err.message().contains("group"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);

    for probabilities in [
        tensor(vec![f64::NAN, 1.0], vec![1, 2]),
        tensor(vec![-1.0, 2.0], vec![1, 2]),
        tensor(vec![0.0, 0.0], vec![1, 2]),
    ] {
        let err = classify_builtin(
            Value::GpuTensor(resident.handle.clone()),
            training.clone(),
            group.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("linear"),
                probabilities,
            ]),
        )
        .await
        .expect_err("invalid host prior must reject before resident sample download");
        assert!(err.message().contains("prior probabilities"));
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    }

    let mut mismatched = runmat_value::UnboundedStructValueBuilder::new();
    mismatched.insert("group", tensor(vec![1.0, 2.0], vec![1, 2]));
    mismatched.insert("prob", tensor(vec![1.0], vec![1, 1]));
    let err = classify_builtin(
        Value::GpuTensor(resident.handle.clone()),
        training.clone(),
        group.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("linear"),
            Value::Struct(mismatched.finish()),
        ]),
    )
    .await
    .expect_err("known prior group/prob length mismatch must reject before download");
    assert!(err.message().contains("group and prob lengths"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);

    let resident_training = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    for invalid_group in [
        Value::LogicalArray(LogicalArray::new(vec![0, 0, 1, 1], vec![2, 2]).unwrap()),
        Value::StringArray(
            StringArray::new(
                vec!["left".into(), "left".into(), "right".into(), "right".into()],
                vec![2, 2],
            )
            .unwrap(),
        ),
        Value::Cell(
            CellArray::new(
                vec![
                    Value::from("left"),
                    Value::from("left"),
                    Value::from("right"),
                    Value::from("right"),
                ],
                2,
                2,
            )
            .unwrap(),
        ),
    ] {
        let err = classify_builtin(
            Value::GpuTensor(resident.handle.clone()),
            training.clone(),
            invalid_group,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .await
        .expect_err("matrix-shaped group must reject before sample download");
        assert!(err.message().contains("group must be a vector"));
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    }
    for (role, control) in [
        (
            "type",
            vec![Value::StringArray(
                StringArray::new(vec!["linear".into(), "quadratic".into()], vec![1, 2]).unwrap(),
            )],
        ),
        (
            "prior",
            vec![
                Value::from("linear"),
                Value::StringArray(
                    StringArray::new(vec!["empirical".into(), "empirical".into()], vec![1, 2])
                        .unwrap(),
                ),
            ],
        ),
    ] {
        let err = classify_builtin(
            Value::GpuTensor(resident.handle.clone()),
            training.clone(),
            group.clone(),
            crate::builtins::common::test_support::builtin_arguments(control),
        )
        .await
        .expect_err("nonscalar text control must reject before sample download");
        assert!(err.message().contains(role));
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    }
    for invalid_sample in [
        Value::String("bad".into()),
        tensor(vec![f64::NAN, 1.0], vec![2, 1]),
        Value::Int(runmat_value::IntValue::I64((1_i64 << 53) + 1)),
    ] {
        classify_builtin(
            invalid_sample,
            Value::GpuTensor(resident_training.handle.clone()),
            group.clone(),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .await
        .expect_err("known-invalid sample must reject before resident training download");
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    }
    for invalid_training in [
        Value::String("bad".into()),
        tensor(vec![0.0, f64::INFINITY, 2.0, 3.0], vec![4, 1]),
        Value::Int(runmat_value::IntValue::I64((1_i64 << 53) + 1)),
    ] {
        classify_builtin(
            Value::GpuTensor(resident.handle.clone()),
            invalid_training,
            group.clone(),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .await
        .expect_err("known-invalid training must reject before resident sample download");
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    }
    classify_builtin(
        Value::GpuTensor(resident.handle.clone()),
        training.clone(),
        tensor(vec![1.0; 4], vec![4, 1]),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .await
    .expect_err("one known class must reject before resident sample download");
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(resident_training.probe.retirements(), 0);

    let valid_prior = tensor(vec![0.25, 0.75], vec![1, 2]);
    let err = classify_builtin(
        Value::GpuTensor(resident.handle.clone()),
        training.clone(),
        group.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("linear"),
            valid_prior,
        ]),
    )
    .await
    .expect_err("valid host prior must not mask the deferred sample provider error");
    assert!(err.message().contains("classify fixture"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);

    let err = classify_builtin(
        Value::GpuTensor(resident.handle.clone()),
        training.clone(),
        group.clone(),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .await
    .expect_err("provider must reject resident sample download");
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 2);
    assert_eq!(err.context.builtin.as_deref(), Some("classify"));
    assert!(err.source.is_some());
    assert!(err.message().contains("classify fixture"));
    assert_eq!(resident.probe.retirements(), 0);

    let complex = provider
        .owner
        .allocate_zeroed(
            vec![2, 1],
            NumericElementType::F64,
            GpuTensorStorage::ComplexInterleaved,
        )
        .unwrap();
    let err = classify_builtin(
        Value::GpuTensor(complex.handle.clone()),
        training,
        group,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .await
    .expect_err("complex predictors must reject before download");
    assert!(err.message().contains("2-D numeric matrix"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 2);
    assert_eq!(complex.probe.retirements(), 0);
    drop(complex.handle);
    drop(resident_training.handle);
    drop(resident.handle);
    assert_eq!(complex.probe.retirements(), 1);
    assert_eq!(resident_training.probe.retirements(), 1);
    assert_eq!(resident.probe.retirements(), 1);
}
