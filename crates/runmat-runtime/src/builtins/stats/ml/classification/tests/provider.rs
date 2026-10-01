use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView, NumericElementType,
    ThreadProviderGuard,
};
use runmat_value::{CellArray, StructValue};
use std::sync::atomic::{AtomicUsize, Ordering};

struct RejectingDownloadProvider {
    owner: TestAllocationOwner,
    downloads: AtomicUsize,
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
    let mut prior = StructValue::new();
    prior.insert("group", group.clone());
    let err = classify_builtin(
        Value::GpuTensor(resident.handle.clone()),
        training.clone(),
        group.clone(),
        vec![Value::from("linear"), Value::Struct(prior)],
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
        Vec::new(),
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
            vec![Value::from("linear"), probabilities],
        )
        .await
        .expect_err("invalid host prior must reject before resident sample download");
        assert!(err.message().contains("prior probabilities"));
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    }

    let mut mismatched = StructValue::new();
    mismatched.insert("group", tensor(vec![1.0, 2.0], vec![1, 2]));
    mismatched.insert("prob", tensor(vec![1.0], vec![1, 1]));
    let err = classify_builtin(
        Value::GpuTensor(resident.handle.clone()),
        training.clone(),
        group.clone(),
        vec![Value::from("linear"), Value::Struct(mismatched)],
    )
    .await
    .expect_err("known prior group/prob length mismatch must reject before download");
    assert!(err.message().contains("group and prob lengths"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);

    let valid_prior = tensor(vec![0.25, 0.75], vec![1, 2]);
    let err = classify_builtin(
        Value::GpuTensor(resident.handle.clone()),
        training.clone(),
        group.clone(),
        vec![Value::from("linear"), valid_prior],
    )
    .await
    .expect_err("valid host prior must not mask the deferred sample provider error");
    assert!(err.message().contains("classify fixture"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);

    let err = classify_builtin(
        Value::GpuTensor(resident.handle.clone()),
        training.clone(),
        group.clone(),
        Vec::new(),
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
        Vec::new(),
    )
    .await
    .expect_err("complex predictors must reject before download");
    assert!(err.message().contains("2-D numeric matrix"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 2);
    assert_eq!(complex.probe.retirements(), 0);
    drop(complex.handle);
    drop(resident.handle);
    assert_eq!(complex.probe.retirements(), 1);
    assert_eq!(resident.probe.retirements(), 1);
}
