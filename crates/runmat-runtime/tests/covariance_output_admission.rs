use futures::executor::block_on;
use runmat_accelerate_api::test_support::{RetirementProbe, TestAllocation, TestAllocationOwner};
use runmat_accelerate_api::*;
use runmat_runtime::{call_builtin_async_with_outputs, GpuGatherRetry};
use runmat_value::Value;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[path = "support/retained_provider.rs"]
mod retained_provider;

#[derive(Clone, Copy, Debug)]
enum ResultKind {
    Valid,
    Shape,
    Integer,
    Logical,
    Complex,
    Owner,
    InputAlias,
    WeightAlias,
    Unavailable,
}

struct Provider {
    owner: TestAllocationOwner,
    other: TestAllocationOwner,
    kind: ResultKind,
    calls: AtomicUsize,
    downloads: AtomicUsize,
    uploads: AtomicUsize,
    outputs: Mutex<Vec<RetirementProbe>>,
}
impl Provider {
    fn new(kind: ResultKind) -> &'static Self {
        Box::leak(Box::new(Self {
            owner: TestAllocationOwner::new(),
            other: TestAllocationOwner::new(),
            kind,
            calls: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
            uploads: AtomicUsize::new(0),
            outputs: Mutex::new(vec![]),
        }))
    }
    fn input(&self, shape: Vec<usize>) -> TestAllocation {
        self.owner
            .allocate_zeroed(shape, NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap()
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "covariance output admission fixture".into()
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        anyhow::bail!("no upload is permitted by this witness")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("no download is permitted by this witness") })
    }
    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("no numeric download is permitted by this witness") })
    }
    fn covariance<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        _: Option<&'a GpuTensorHandle>,
        weights: Option<&'a GpuTensorHandle>,
        _: &'a CovarianceOptions,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            match self.kind {
                ResultKind::InputAlias => {
                    return input
                        .with_view(GpuTensorView::contiguous(vec![2, 2]))
                        .map_err(ProviderExecutionError::from_source)
                }
                ResultKind::WeightAlias => {
                    return weights
                        .unwrap()
                        .with_view(GpuTensorView::contiguous(vec![2, 2]))
                        .map_err(ProviderExecutionError::from_source)
                }
                _ => {}
            }
            let owner = if matches!(self.kind, ResultKind::Owner) {
                &self.other
            } else {
                &self.owner
            };
            let allocation = if matches!(self.kind, ResultKind::Logical) {
                owner.allocate_logical(vec![0; 4], vec![2, 2])?
            } else {
                owner.allocate_zeroed(
                    if matches!(self.kind, ResultKind::Shape) {
                        vec![1, 4]
                    } else {
                        vec![2, 2]
                    },
                    if matches!(self.kind, ResultKind::Integer) {
                        NumericElementType::I32
                    } else {
                        NumericElementType::F64
                    },
                    if matches!(self.kind, ResultKind::Complex) {
                        GpuTensorStorage::ComplexInterleaved
                    } else {
                        GpuTensorStorage::Real
                    },
                )?
            };
            if matches!(self.kind, ResultKind::Unavailable) {
                allocation.probe.mark_unavailable();
            }
            self.outputs.lock().unwrap().push(allocation.probe);
            Ok(allocation.handle)
        })
    }
}

#[test]
fn malformed_covariance_outputs_retire_once_and_preserve_borrowed_inputs() {
    for kind in [
        ResultKind::Shape,
        ResultKind::Integer,
        ResultKind::Logical,
        ResultKind::Complex,
        ResultKind::Owner,
        ResultKind::InputAlias,
        ResultKind::Unavailable,
    ] {
        let provider = Provider::new(kind);
        let _active = ThreadProviderGuard::set(Some(provider));
        let input = provider.input(vec![2, 2]);
        let error = block_on(call_builtin_async_with_outputs(
            "cov",
            &[Value::GpuTensor(input.handle.clone())],
            1,
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:cov:Internal"), "{kind:?}");
        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
        assert_eq!(input.probe.retirements(), 0);
        assert!(input.handle.allocation_lease().validate_live().is_ok());
        if matches!(kind, ResultKind::Unavailable) {
            let mut source: &(dyn std::error::Error + 'static) =
                error.source.as_ref().unwrap().as_ref();
            loop {
                if let Some(actual) =
                    source.downcast_ref::<allocation::AllocationAcquisitionError>()
                {
                    assert!(matches!(
                        actual,
                        allocation::AllocationAcquisitionError::NotLive
                    ));
                    break;
                }
                source = source.source().expect("original typed liveness failure");
            }
        }
        drop(error);
        for output in provider.outputs.lock().unwrap().iter() {
            assert_eq!(output.retirements(), 1, "{kind:?}");
        }
        drop(input.handle);
        assert_eq!(input.probe.retirements(), 1);
    }
}

#[test]
fn covariance_weight_alias_is_rejected_without_retiring_the_borrowed_weight() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let provider = Provider::new(ResultKind::WeightAlias);
    let _active = ThreadProviderGuard::set(Some(provider));
    let input = provider.input(vec![4, 2]);
    let weights = provider.input(vec![4, 1]);
    let error = block_on(call_builtin_async_with_outputs(
        "cov",
        &[
            Value::GpuTensor(input.handle.clone()),
            Value::GpuTensor(weights.handle.clone()),
        ],
        1,
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:cov:Internal"));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(weights.probe.retirements(), 0);
    assert!(weights.handle.allocation_lease().validate_live().is_ok());
    drop((input.handle, weights.handle));
    assert_eq!(input.probe.retirements(), 1);
    assert_eq!(weights.probe.retirements(), 1);
}

#[test]
fn unknown_covariance_operand_owner_never_reaches_unrelated_active_hooks() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for unknown_weight in [false, true] {
        let provider = Provider::new(ResultKind::Valid);
        let _active = ThreadProviderGuard::set(Some(provider));
        let foreign_owner = TestAllocationOwner::new();
        let foreign = foreign_owner
            .allocate_zeroed(
                if unknown_weight {
                    vec![4, 1]
                } else {
                    vec![4, 2]
                },
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )
            .unwrap();
        let input = provider.input(vec![4, 2]);
        let args = if unknown_weight {
            vec![
                Value::GpuTensor(input.handle.clone()),
                Value::GpuTensor(foreign.handle.clone()),
            ]
        } else {
            vec![Value::GpuTensor(foreign.handle.clone())]
        };
        let error = block_on(call_builtin_async_with_outputs("cov", &args, 1)).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:cov:Internal"));
        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
        assert_eq!(foreign.probe.retirements(), 0);
        drop(args);
        drop((input.handle, foreign.handle));
        assert_eq!(foreign.probe.retirements(), 1);
        assert_eq!(input.probe.retirements(), 1);
    }
}

#[test]
fn valid_covariance_output_adopts_explicit_intent_and_owns_only_its_result() {
    let provider = Provider::new(ResultKind::Valid);
    let _active = ThreadProviderGuard::set(Some(provider));
    let input = provider.input(vec![2, 2]);
    let explicit = input
        .handle
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    let result = block_on(call_builtin_async_with_outputs(
        "cov",
        &[Value::GpuTensor(explicit)],
        1,
    ))
    .unwrap();
    let Value::GpuTensor(output) = result.value() else {
        panic!("resident result")
    };
    assert_eq!(output.shape(), [2, 2]);
    assert_eq!(
        handle_provenance(output),
        Some(GpuHandleProvenance::Explicit)
    );
    assert!(!output.shares_allocation_with(&input.handle));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.outputs.lock().unwrap()[0].retirements(), 0);
    drop(result);
    assert_eq!(provider.outputs.lock().unwrap()[0].retirements(), 1);
    assert_eq!(input.probe.retirements(), 0);
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

#[test]
fn covariance_shape_admission_matches_host_vectors_matrices_and_empty_inputs() {
    use runmat_accelerate::simple_provider::InProcessProvider;
    use runmat_value::Tensor;
    let provider = retained_provider::retained_test_provider!(PROVIDER: InProcessProvider = InProcessProvider::default());
    let _active = ThreadProviderGuard::set(Some(provider));
    for shape in [
        vec![],
        vec![3],
        vec![1, 3],
        vec![3, 1],
        vec![0, 0],
        vec![0, 3],
        vec![3, 0],
        vec![3, 2],
    ] {
        let data = vec![0.0; shape.iter().product()];
        let input = provider
            .upload(&HostTensorView {
                data: &data,
                shape: &shape,
            })
            .unwrap();
        let input_weak = input.allocation_lease().downgrade();
        let host = block_on(call_builtin_async_with_outputs(
            "cov",
            &[Value::Tensor(Tensor::new(data, shape.clone()).unwrap())],
            1,
        ))
        .unwrap();
        let resident = block_on(call_builtin_async_with_outputs(
            "cov",
            &[Value::GpuTensor(input.clone())],
            1,
        ))
        .unwrap();
        let Value::GpuTensor(output) = resident.value() else {
            panic!("resident covariance")
        };
        let Value::Tensor(expected) = host.value() else {
            panic!("covariance host result must be a dense tensor, got {host:?}");
        };
        let actual = block_on(provider.download(output)).unwrap();
        assert_eq!(actual.shape, expected.shape(), "{shape:?}");
        let expected_data = expected.materialize_f64();
        assert_eq!(actual.data.len(), expected_data.len());
        for (actual, expected) in actual.data.iter().zip(expected_data) {
            assert!(
                (*actual == expected) || (actual.is_nan() && expected.is_nan()),
                "{shape:?}"
            );
        }
        let output_weak = output.allocation_lease().downgrade();
        drop(resident);
        assert!(output_weak.acquire().is_none());
        assert!(input_weak.acquire().is_some());
        drop(input);
        assert!(input_weak.acquire().is_none());
    }
}
