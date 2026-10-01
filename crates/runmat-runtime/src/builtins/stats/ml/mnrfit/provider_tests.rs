use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView, NumericElementType,
    ThreadProviderGuard,
};
use runmat_value::StructValue;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

struct FailingDownloadProvider {
    owner: TestAllocationOwner,
    downloads: AtomicUsize,
}

impl AccelProvider for FailingDownloadProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "mnrfit gather fixture".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, AtomicOrdering::SeqCst);
        Box::pin(async { anyhow::bail!("mnrfit fixture download failure") })
    }

    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, AtomicOrdering::SeqCst);
        Box::pin(async { anyhow::bail!("mnrfit fixture native download failure") })
    }
}

#[tokio::test]
async fn mnrfit_provider_failure_retains_cause_and_borrowed_lease() {
    let provider = Box::leak(Box::new(FailingDownloadProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let resident = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let error = mnrfit_builtin(
        Value::GpuTensor(resident.handle.clone()),
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
        Vec::new(),
    )
    .await
    .expect_err("registered provider must fail the X download");
    assert_eq!(provider.downloads.load(AtomicOrdering::SeqCst), 1);
    assert_eq!(error.context.builtin.as_deref(), Some("mnrfit"));
    assert!(error.message().contains("mnrfit fixture"));
    assert!(error.source.is_some());
    assert_eq!(resident.probe.retirements(), 0);
    drop(resident.handle);
    assert_eq!(resident.probe.retirements(), 1);
}

#[tokio::test]
async fn mnrfit_decodes_host_controls_before_provider_effects_and_skips_ignored_fields() {
    let provider = Box::leak(Box::new(FailingDownloadProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let resident = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let y = Value::Tensor(Tensor::new(vec![1.0, 1.0, 2.0, 2.0], vec![4, 1]).unwrap());

    for options in [
        vec![Value::from("Model"), Value::from("unsupported")],
        vec![Value::from("IterationLimit"), Value::Num(-1.0)],
        vec![Value::from("Weights"), Value::Num(f64::NAN)],
    ] {
        mnrfit_builtin(
            Value::GpuTensor(resident.handle.clone()),
            y.clone(),
            options,
        )
        .await
        .expect_err("known-invalid host option must reject before X download");
        assert_eq!(provider.downloads.load(AtomicOrdering::SeqCst), 0);
    }
    let mut invalid_statset = StructValue::new();
    invalid_statset.insert("MaxIter", Value::Num(-1.0));
    invalid_statset.insert("Ignored", Value::GpuTensor(resident.handle.clone()));
    mnrfit_builtin(
        Value::GpuTensor(resident.handle.clone()),
        y.clone(),
        vec![Value::from("Options"), Value::Struct(invalid_statset)],
    )
    .await
    .expect_err("known-invalid selected statset field must reject before X download");
    assert_eq!(provider.downloads.load(AtomicOrdering::SeqCst), 0);

    let mut valid_statset = StructValue::new();
    valid_statset.insert("MaxIter", Value::Num(30.0));
    valid_statset.insert("Ignored", Value::GpuTensor(resident.handle.clone()));
    let x = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap());
    mnrfit_builtin(
        x.clone(),
        y.clone(),
        vec![Value::from("Options"), Value::Struct(valid_statset)],
    )
    .await
    .expect("unused resident statset field must not be downloaded");
    assert_eq!(provider.downloads.load(AtomicOrdering::SeqCst), 0);

    let error = mnrfit_builtin(
        x,
        y,
        vec![
            Value::from("Weights"),
            Value::GpuTensor(resident.handle.clone()),
        ],
    )
    .await
    .expect_err("selected resident option must retain provider failure");
    assert_eq!(provider.downloads.load(AtomicOrdering::SeqCst), 1);
    assert!(error.source.is_some());
    assert_eq!(resident.probe.retirements(), 0);
    drop(resident.handle);
    assert_eq!(resident.probe.retirements(), 1);
}
