//! Scalar max must terminate on an operational error after capability selection.
use super::*;
use runmat_accelerate_api::test_support::RetirementProbe;
use std::sync::Mutex;

#[derive(Clone, Copy, Debug)]
enum FillStage {
    Unsupported,
    Success,
}

struct ContinuationProvider {
    owner: TestAllocationOwner,
    stage: FillStage,
    nested: bool,
    scalar_left: bool,
    fills: AtomicUsize,
    scalars: AtomicUsize,
    elements: AtomicUsize,
    downloads: AtomicUsize,
    fill_probe: Mutex<Option<RetirementProbe>>,
}

impl ContinuationProvider {
    fn failure<T>(&self) -> anyhow::Result<T> {
        if self.nested {
            Err(
                runmat_runtime::build_runtime_error("continuation provider contract")
                    .with_identifier("RunMat:test:ContinuationContract")
                    .with_builtin("continuation-origin")
                    .with_source(DeviceFailure)
                    .build()
                    .into(),
            )
        } else {
            Err(DeviceFailure.into())
        }
    }
}

impl AccelProvider for ContinuationProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "extrema continuation failure fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("operational failure must not retry an upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        panic!("operational failure must not gather")
    }
    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        panic!("operational failure must not gather native data")
    }
    fn fill_like(
        &self,
        input: &GpuTensorHandle,
        scalar: f64,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        self.fills.fetch_add(1, Ordering::SeqCst);
        input.allocation_lease().validate_live().unwrap();
        assert_eq!(scalar, 2.0);
        match self.stage {
            // Optional constructors certify an unavailable implementation by
            // returning None, not by emitting an unproven execution failure.
            FillStage::Unsupported => Ok(None),
            FillStage::Success => {
                let fill = self
                    .owner
                    .allocate_numeric(HostNumericTensorOwned {
                        data: HostNumericDataOwned::F64(vec![
                            scalar;
                            input.shape().iter().product()
                        ]),
                        shape: input.shape().to_vec(),
                        storage: GpuTensorStorage::Real,
                    })
                    .map_err(|source| {
                        ProviderExecutionError::from(source).with_unproven_effects()
                    })?;
                assert!(!fill.handle.shares_allocation_with(input));
                *self.fill_probe.lock().unwrap() = Some(fill.probe);
                Ok(Some(fill.handle))
            }
        }
    }
    fn scalar_max(&self, input: &GpuTensorHandle, scalar: f64) -> anyhow::Result<GpuTensorHandle> {
        self.scalars.fetch_add(1, Ordering::SeqCst);
        assert!(matches!(self.stage, FillStage::Unsupported));
        input.allocation_lease().validate_live().unwrap();
        assert_eq!(scalar, 2.0);
        self.failure()
    }
    fn elem_max<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        right: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.elements.fetch_add(1, Ordering::SeqCst);
            assert!(matches!(self.stage, FillStage::Success));
            left.allocation_lease().validate_live().unwrap();
            right.allocation_lease().validate_live().unwrap();
            assert!(!left.shares_allocation_with(right));
            let filled = if self.scalar_left { left } else { right };
            let probe = self.fill_probe.lock().unwrap().as_ref().unwrap().clone();
            assert_eq!(
                probe.retirements(),
                0,
                "temporary must survive the provider call"
            );
            probe
                .with_data(filled, |data| {
                    let HostNumericDataView::F64(values) = data else {
                        panic!("genuine Double fill")
                    };
                    assert_eq!(values, &[2.0, 2.0]);
                })
                .unwrap();
            self.failure().map_err(Into::into)
        })
    }
}

#[test]
fn public_scalar_max_continuation_errors_do_not_gather_and_retire_only_temporaries() {
    for stage in [FillStage::Unsupported, FillStage::Success] {
        for scalar_left in [false, true] {
            for nested in [false, true] {
                let provider = Box::leak(Box::new(ContinuationProvider {
                    owner: TestAllocationOwner::new(),
                    stage,
                    nested,
                    scalar_left,
                    fills: AtomicUsize::new(0),
                    scalars: AtomicUsize::new(0),
                    elements: AtomicUsize::new(0),
                    downloads: AtomicUsize::new(0),
                    fill_probe: Mutex::new(None),
                }));
                let _selected = ThreadProviderGuard::set(Some(provider));
                let input = provider
                    .owner
                    .allocate_numeric(HostNumericTensorOwned {
                        data: HostNumericDataOwned::F64(vec![3.0, -1.0]),
                        shape: vec![2, 1],
                        storage: GpuTensorStorage::Real,
                    })
                    .unwrap();
                let original = input.handle.wire_descriptor();
                let args = if scalar_left {
                    vec![Value::Num(2.0), Value::GpuTensor(input.handle.clone())]
                } else {
                    vec![Value::GpuTensor(input.handle.clone()), Value::Num(2.0)]
                };
                let error = block_on(runmat_runtime::call_builtin_async("max", &args)).unwrap_err();
                assert_eq!(provider.fills.load(Ordering::SeqCst), 1);
                assert_eq!(
                    provider.scalars.load(Ordering::SeqCst),
                    usize::from(matches!(stage, FillStage::Unsupported))
                );
                assert_eq!(
                    provider.elements.load(Ordering::SeqCst),
                    usize::from(matches!(stage, FillStage::Success))
                );
                assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
                assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
                let mut cause: &(dyn std::error::Error + 'static) = &error;
                while !cause.is::<DeviceFailure>() {
                    cause = cause
                        .source()
                        .expect("continuation retains typed device failure");
                }
                if nested {
                    assert_eq!(error.identifier(), Some("RunMat:test:ContinuationContract"));
                    assert_eq!(error.message(), "continuation provider contract");
                    assert_eq!(
                        error.context.builtin.as_deref(),
                        Some("continuation-origin")
                    );
                }
                let fill_probe = provider.fill_probe.lock().unwrap().clone();
                match (&fill_probe, stage) {
                    (Some(probe), FillStage::Success) => assert_eq!(probe.retirements(), 1),
                    (None, FillStage::Unsupported) => {}
                    _ => panic!("fill allocation must correspond to the executed capability"),
                }
                assert_eq!(input.handle.wire_descriptor(), original);
                assert_eq!(input.probe.retirements(), 0);
                input
                    .probe
                    .with_data(&input.handle, |data| {
                        let HostNumericDataView::F64(values) = data else {
                            panic!("Double input preserved")
                        };
                        assert_eq!(values, &[3.0, -1.0]);
                    })
                    .unwrap();
                drop(error);
                drop(args);
                input.handle.release().unwrap();
                assert_eq!(input.probe.retirements(), 1);
                if let Some(probe) = fill_probe {
                    assert_eq!(probe.retirements(), 1);
                }
            }
        }
    }
}
