use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostNumericTensorView, HostTensorView,
    NumericElementType,
};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

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

    fn resident_scores(&self) -> GpuTensorHandle {
        let scores = Tensor::new(vec![0.9, 0.8, 0.4, 0.1], vec![4, 1]).unwrap();
        crate::builtins::common::gpu_helpers::upload_tensor(self, &scores).unwrap()
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
        self.downloads.fetch_add(1, AtomicOrdering::SeqCst);
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
        self.downloads.fetch_add(1, AtomicOrdering::SeqCst);
        self.inner.download_numeric(handle)
    }
}

struct RejectDownloadProvider {
    owner: TestAllocationOwner,
    downloads: AtomicUsize,
}

impl RejectDownloadProvider {
    fn new() -> &'static Self {
        Box::leak(Box::new(Self {
            owner: TestAllocationOwner::new(),
            downloads: AtomicUsize::new(0),
        }))
    }
}

impl AccelProvider for RejectDownloadProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "perfcurve gather failure fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, AtomicOrdering::SeqCst);
        Box::pin(async { anyhow::bail!("perfcurve source download rejected") })
    }
    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, AtomicOrdering::SeqCst);
        Box::pin(async { anyhow::bail!("perfcurve source native download rejected") })
    }
}

fn labels() -> Value {
    tensor(&[1.0, 0.0, 1.0, 0.0], &[4, 1])
}

#[test]
fn later_invalid_host_control_precedes_resident_score_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountingProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let scores = provider.resident_scores();
    let error = block_on(perfcurve_builtin(
        labels(),
        Value::GpuTensor(scores.clone()),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("XCrit"),
            Value::from("tpr"),
            Value::from("Weights"),
            tensor(&[1.0, -1.0, 1.0, 1.0], &[4, 1]),
        ]),
    ))
    .expect_err("negative host weights");
    assert!(error.message().contains("Weights"));
    assert_eq!(provider.downloads.load(AtomicOrdering::SeqCst), 0);
    // A failed admission has not consumed the caller's resident allocation.
    block_on(provider.inner.download(&scores)).expect("borrowed scores remain downloadable");
}

#[test]
fn known_invalid_host_label_or_option_geometry_precedes_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountingProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let scores = provider.resident_scores();
    let label_error = block_on(perfcurve_builtin(
        tensor(&[1.0, 0.0, 1.0, 0.0], &[2, 2]),
        Value::GpuTensor(scores.clone()),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("matrix labels");
    assert!(label_error.message().contains("vector"));
    let cost_error = block_on(perfcurve_builtin(
        labels(),
        Value::GpuTensor(scores),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Cost"),
            tensor(&[0.0, 1.0, 1.0, 0.0], &[4, 1]),
        ]),
    ))
    .expect_err("invalid cost geometry");
    assert!(cost_error.message().contains("Cost"));
    assert_eq!(provider.downloads.load(AtomicOrdering::SeqCst), 0);
}

#[test]
fn known_invalid_negative_classes_precede_resident_score_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountingProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let scores = provider.resident_scores();
    for (observed_labels, options, expected) in [
        (
            tensor(&[1.0, 1.0, 1.0, 1.0], &[4, 1]),
            vec![],
            "negative class",
        ),
        (
            labels(),
            vec![Value::from("NegClass"), Value::Num(1.0)],
            "posclass",
        ),
        (
            labels(),
            vec![Value::from("NegClass"), Value::from("not-numeric")],
            "numeric",
        ),
        (
            labels(),
            vec![Value::from("NegClass"), tensor(&[], &[0, 1])],
            "NegClass",
        ),
    ] {
        let error = block_on(perfcurve_builtin(
            observed_labels,
            Value::GpuTensor(scores.clone()),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(options),
        ))
        .expect_err("host-known invalid class plan");
        assert!(error.message().contains(expected), "{}", error.message());
        assert_eq!(provider.downloads.load(AtomicOrdering::SeqCst), 0);
    }
    block_on(provider.inner.download(&scores)).expect("source allocation remains live");
}

#[test]
fn unsupported_output_count_precedes_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountingProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let _outputs = output_count::push_output_count(Some(8)).expect("output-count scope");
    let scores = provider.resident_scores();
    let error = block_on(perfcurve_builtin(
        labels(),
        Value::GpuTensor(scores),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("eight outputs are not supported");
    assert!(error.message().contains("too many output"));
    assert_eq!(provider.downloads.load(AtomicOrdering::SeqCst), 0);
}

#[test]
fn discarded_call_admits_host_structure_without_download_or_curve_execution() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountingProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let _outputs = output_count::push_output_count(Some(0)).expect("output-count scope");
    let scores = provider.resident_scores();

    let error = block_on(perfcurve_builtin(
        labels(),
        Value::GpuTensor(scores.clone()),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Weights"),
            tensor(&[1.0, -1.0, 1.0, 1.0], &[4, 1]),
        ]),
    ))
    .expect_err("known-invalid host control is rejected even without output");
    assert!(error.message().contains("Weights"));
    assert_eq!(provider.downloads.load(AtomicOrdering::SeqCst), 0);

    let empty = block_on(perfcurve_builtin(
        labels(),
        Value::GpuTensor(scores.clone()),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect("discarded call does not materialize the resident score vector");
    assert!(matches!(empty, Value::OutputList(outputs) if outputs.is_empty()));
    assert_eq!(provider.downloads.load(AtomicOrdering::SeqCst), 0);

    let empty = block_on(perfcurve_builtin(
        labels(),
        tensor(&[f64::INFINITY, 0.8, 0.4, 0.1], &[4, 1]),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect("discarded call does not run the data-dependent curve kernel");
    assert!(matches!(empty, Value::OutputList(outputs) if outputs.is_empty()));
    assert_eq!(provider.downloads.load(AtomicOrdering::SeqCst), 0);
    block_on(provider.inner.download(&scores)).expect("borrowed score allocation remains live");
}

#[test]
fn failed_gather_preserves_provider_cause_and_borrowed_input() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = RejectDownloadProvider::new();
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let error = block_on(perfcurve_builtin(
        labels(),
        Value::GpuTensor(resident.handle.clone()),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("provider download must fail");
    assert_eq!(error.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    let source = error
        .source
        .as_ref()
        .expect("original runtime gather cause");
    let mut found = source.to_string().contains("perfcurve source");
    let mut cause = source.source();
    while let Some(next) = cause {
        found |= next.to_string().contains("perfcurve source");
        cause = next.source();
    }
    assert!(found, "provider error remains in nested causal chain");
    assert_eq!(provider.downloads.load(AtomicOrdering::SeqCst), 1);
    resident
        .probe
        .with_data(&resident.handle, |_| ())
        .expect("borrowed allocation stays live");
}
