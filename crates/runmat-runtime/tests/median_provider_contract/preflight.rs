use super::*;
use runmat_accelerate_api::test_support::TestAllocationOwner;

struct PreflightProvider {
    owner: TestAllocationOwner,
    effects: AtomicUsize,
}
impl AccelProvider for PreflightProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "median preflight owner".into()
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        self.effects.fetch_add(1, Ordering::SeqCst);
        anyhow::bail!("unexpected upload")
    }
    fn upload_numeric_raw(
        &self,
        _: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.effects.fetch_add(1, Ordering::SeqCst);
        Err(anyhow::anyhow!("unexpected native upload").into())
    }
    fn reduce_median_dim<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.effects.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("unexpected reduction") })
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.effects.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("unexpected download") })
    }
    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.effects.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("unexpected native download") })
    }
}

#[test]
fn all_resident_owners_are_admitted_before_control_downloads() {
    for dead in [false, true] {
        for weights in [false, true] {
            let provider = Box::leak(Box::new(PreflightProvider {
                owner: TestAllocationOwner::new(),
                effects: AtomicUsize::new(0),
            }));
            let _active = ThreadProviderGuard::set(Some(provider));
            let unknown = TestAllocationOwner::new();
            let primary = (if dead { &provider.owner } else { &unknown })
                .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            if dead {
                primary.probe.mark_unavailable();
            }
            let control = provider
                .owner
                .allocate_numeric(HostNumericTensorOwned {
                    data: HostNumericDataOwned::F64(if weights { vec![1., 2.] } else { vec![1.] }),
                    shape: if weights { vec![2, 1] } else { vec![1, 1] },
                    storage: GpuTensorStorage::Real,
                })
                .unwrap();
            let mut args = vec![Value::GpuTensor(primary.handle.clone())];
            if weights {
                args.push(Value::String("Weights".into()));
            }
            args.push(Value::GpuTensor(control.handle.clone()));
            let error = block_on(call_builtin_async_with_outputs("median", &args, 1)).unwrap_err();
            assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
            if dead {
                let mut cause: &(dyn std::error::Error + 'static) =
                    error.source.as_deref().unwrap();
                while !cause.is::<allocation::AllocationAcquisitionError>() {
                    cause = cause.source().expect("retain allocation liveness cause");
                }
            }
            assert_eq!(provider.effects.load(Ordering::SeqCst), 0);
            assert!(control.handle.allocation_lease().validate_live().is_ok());
            // A pure prefix error still wins before resident materialization.
            let error = block_on(call_builtin_async_with_outputs(
                "median",
                &[
                    Value::GpuTensor(primary.handle.clone()),
                    Value::String("invalid-option".into()),
                    Value::GpuTensor(control.handle.clone()),
                ],
                1,
            ))
            .unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:median:InvalidArgument"));
            assert_eq!(provider.effects.load(Ordering::SeqCst), 0);
            drop(args);
            drop((primary.handle, control.handle));
            assert_eq!(primary.probe.retirements(), 1);
            assert_eq!(control.probe.retirements(), 1);
        }
    }
}

#[test]
fn later_unknown_control_blocks_first_control_download() {
    let provider = Box::leak(Box::new(PreflightProvider {
        owner: TestAllocationOwner::new(),
        effects: AtomicUsize::new(0),
    }));
    let _active = ThreadProviderGuard::set(Some(provider));
    let unknown = TestAllocationOwner::new();
    let primary = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let axis = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![1.]),
            shape: vec![1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let weights = unknown
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let error = block_on(call_builtin_async_with_outputs(
        "median",
        &[
            Value::GpuTensor(primary.handle.clone()),
            Value::GpuTensor(axis.handle.clone()),
            Value::String("Weights".into()),
            Value::GpuTensor(weights.handle.clone()),
        ],
        1,
    ))
    .unwrap_err();
    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
    assert_eq!(provider.effects.load(Ordering::SeqCst), 0);
    drop((primary.handle, axis.handle, weights.handle));
    assert_eq!(primary.probe.retirements(), 1);
    assert_eq!(axis.probe.retirements(), 1);
    assert_eq!(weights.probe.retirements(), 1);
}

#[test]
fn cancellation_retires_intermediate_but_preserves_resident_axis_control() {
    use std::{
        future::Future,
        task::{Context, Poll},
    };
    let provider = Provider::new(Mode::LaterPending);
    let _active = ThreadProviderGuard::set(Some(provider));
    let input = input(provider);
    let axes = provider
        .inner
        .upload(&HostTensorView {
            data: &[1., 2.],
            shape: &[1, 2],
        })
        .unwrap();
    let input_weak = input.allocation_lease().downgrade();
    let axes_weak = axes.allocation_lease().downgrade();
    let args = [
        Value::GpuTensor(input.clone()),
        Value::GpuTensor(axes.clone()),
    ];
    let mut future = Box::pin(call_builtin_async_with_outputs("median", &args, 1));
    let waker = futures::task::noop_waker();
    assert!(matches!(
        future.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    assert!(provider.produced.lock().unwrap()[0].acquire().is_some());
    drop(future);
    assert!(provider
        .produced
        .lock()
        .unwrap()
        .iter()
        .all(|lease| lease.acquire().is_none()));
    assert_eq!(
        block_on(provider.inner.download(&input)).unwrap().data,
        [1., 3., 2., 8.]
    );
    assert_eq!(
        block_on(provider.inner.download(&axes)).unwrap().data,
        [1., 2.]
    );
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    drop(args);
    drop((input, axes));
    assert!(input_weak.acquire().is_none());
    assert!(axes_weak.acquire().is_none());
}

#[test]
fn empty_single_restoration_preserves_explicit_intent_and_terminal_failures() {
    for explicit in [false, true] {
        for mode in [Mode::RestoreUnsupported, Mode::RestoreFailure] {
            let provider = Provider::new(mode);
            let _active = ThreadProviderGuard::set(Some(provider));
            let input = provider
                .inner
                .upload_numeric_raw(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&[]),
                    shape: &[0, 3],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap()
                .with_provenance(if explicit {
                    GpuHandleProvenance::Explicit
                } else {
                    GpuHandleProvenance::Automatic
                })
                .unwrap();
            let weak = input.allocation_lease().downgrade();
            let result = block_on(call_builtin_async_with_outputs(
                "median",
                &[Value::GpuTensor(input.clone())],
                1,
            ));
            if !explicit && matches!(mode, Mode::RestoreUnsupported) {
                let result = result.unwrap();
                let Value::Tensor(host) = result.value() else {
                    panic!("automatic unsupported restoration retains exact host carrier")
                };
                assert_eq!(host.shape(), &[1, 3]);
                assert_eq!(host.numeric_dtype(), runmat_value::NumericDType::F64);
                assert!(host.materialize_f64().iter().all(|value| value.is_nan()));
            } else {
                let error = result.unwrap_err();
                assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
                let source = error
                    .source
                    .as_ref()
                    .unwrap()
                    .downcast_ref::<NumericUploadError>()
                    .unwrap();
                match source {
                    NumericUploadError::Provider(source)
                        if matches!(mode, Mode::RestoreUnsupported) =>
                    {
                        assert!(source.is_unsupported())
                    }
                    NumericUploadError::Provider(source) => assert_eq!(
                        std::error::Error::source(source)
                            .and_then(|source| source.downcast_ref::<std::io::Error>())
                            .unwrap()
                            .kind(),
                        std::io::ErrorKind::BrokenPipe
                    ),
                    other => panic!("unexpected failure carrier: {other:?}"),
                }
            }
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
            assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
            assert!(input.allocation_lease().validate_live().is_ok());
            drop(input);
            assert!(weak.acquire().is_none());
        }
    }
}
