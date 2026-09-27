use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelProviderFuture, GpuTensorStorage,
    HostTensorOwned, HostTensorView, NumericElementType, ReduceDimResult, ThreadProviderGuard,
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn scalar_extrema_use_native_single_storage_selection() {
    use runmat_value::{ComplexStorage, ComplexTensor, NumericStorage};

    for (name, expected) in [("min", 1.25), ("max", 3.5)] {
        for shape in [vec![1, 1], vec![1, 1, 1]] {
            let left = Value::Tensor(Tensor::from_f32(vec![3.5], shape.clone()).unwrap());
            let right = Value::Tensor(Tensor::from_f32(vec![1.25], shape.clone()).unwrap());
            let result = futures::executor::block_on(crate::call_builtin_async(
                name,
                &[left.clone(), right],
            ))
            .unwrap();
            let Value::Tensor(result) = result else {
                panic!("{name}: Single scalar must retain its native carrier");
            };
            assert_eq!(result.shape, shape);
            assert_eq!(
                result.into_numeric_storage().unwrap(),
                NumericStorage::F32(vec![expected])
            );

            // Preserve the existing mixed-class policy, in both operand orders.
            for args in [[left.clone(), Value::Num(1.25)], [Value::Num(1.25), left]] {
                assert_eq!(
                    futures::executor::block_on(crate::call_builtin_async(name, &args)).unwrap(),
                    Value::Num(f64::from(expected))
                );
            }

            // Zero imaginary parts still belong to the complex Single carrier.
            let args = [
                Value::ComplexTensor(
                    ComplexTensor::from_f32(vec![(3.5, 0.0)], shape.clone()).unwrap(),
                ),
                Value::ComplexTensor(
                    ComplexTensor::from_f32(vec![(1.25, 0.0)], shape.clone()).unwrap(),
                ),
            ];
            let result =
                futures::executor::block_on(crate::call_builtin_async(name, &args)).unwrap();
            let Value::ComplexTensor(result) = result else {
                panic!("{name}: complex Single scalar must retain its native carrier");
            };
            assert_eq!(result.shape, shape);
            assert_eq!(
                result.into_complex_storage(),
                ComplexStorage::F32(vec![(expected, 0.0)].into())
            );
        }
    }
}

#[derive(Clone, Copy)]
enum Fault {
    Execution,
    Unsupported,
    Download,
}

struct Provider {
    owner: TestAllocationOwner,
    fault: Fault,
    reductions: AtomicUsize,
    downloads: AtomicUsize,
}

impl Provider {
    fn reject<T>(&self) -> anyhow::Result<T> {
        self.reductions.fetch_add(1, Ordering::SeqCst);
        match self.fault {
            Fault::Execution => Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "unsupported transport state is not a capability rejection",
            )
            .into()),
            Fault::Unsupported | Fault::Download => {
                Err(runmat_accelerate_api::unsupported_provider_operation(
                    "dimensional reduction unavailable",
                ))
            }
        }
    }
}

impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "extrema execution authority fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("unexpected restoration")
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            self.downloads.fetch_add(1, Ordering::SeqCst);
            if matches!(self.fault, Fault::Download) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::ConnectionReset,
                    "download failed",
                )
                .into());
            }
            assert_eq!(input.device_id(), self.device_id());
            input.allocation_lease().validate_live()?;
            Ok(HostTensorOwned {
                data: vec![0.0; input.shape().iter().product()],
                shape: input.shape().to_vec(),
                storage: input.physical().storage(),
            })
        })
    }
    fn reduce_min_dim<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: usize,
    ) -> AccelProviderFuture<'a, ReduceDimResult> {
        Box::pin(async move { self.reject() })
    }
    fn reduce_max_dim<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: usize,
    ) -> AccelProviderFuture<'a, ReduceDimResult> {
        Box::pin(async move { self.reject() })
    }

    fn elem_min<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reject() })
    }
    fn elem_max<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reject() })
    }
    fn fill_like(&self, _: &GpuTensorHandle, _: f64) -> anyhow::Result<GpuTensorHandle> {
        self.reject()
    }
    fn scalar_min(&self, _: &GpuTensorHandle, _: f64) -> anyhow::Result<GpuTensorHandle> {
        self.reject()
    }
    fn scalar_max(&self, _: &GpuTensorHandle, _: f64) -> anyhow::Result<GpuTensorHandle> {
        self.reject()
    }
}

#[test]
fn dimensional_extrema_only_gather_after_typed_unsupported() {
    for name in ["min", "max"] {
        for fault in [Fault::Execution, Fault::Unsupported, Fault::Download] {
            let provider = Box::leak(Box::new(Provider {
                owner: TestAllocationOwner::new(),
                fault,
                reductions: AtomicUsize::new(0),
                downloads: AtomicUsize::new(0),
            }));
            let _selected = ThreadProviderGuard::set(Some(provider));
            let input = provider
                .owner
                .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let result = futures::executor::block_on(crate::call_builtin_async(
                name,
                &[Value::GpuTensor(input.handle.clone())],
            ));
            assert_eq!(provider.reductions.load(Ordering::SeqCst), 1);
            match fault {
                Fault::Unsupported => assert_eq!(result.unwrap(), Value::Num(0.0)),
                Fault::Execution | Fault::Download => {
                    let error = result.unwrap_err();
                    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
                    let mut source = std::error::Error::source(&error);
                    let mut retained = None;
                    while let Some(error) = source {
                        if let Some(error) = error.downcast_ref::<std::io::Error>() {
                            retained = Some(error.kind());
                        }
                        source = error.source();
                    }
                    assert_eq!(
                        retained,
                        Some(if matches!(fault, Fault::Execution) {
                            std::io::ErrorKind::BrokenPipe
                        } else {
                            std::io::ErrorKind::ConnectionReset
                        }),
                        "original {name} provider failure must survive"
                    );
                }
            }
            assert_eq!(
                provider.downloads.load(Ordering::SeqCst),
                usize::from(!matches!(fault, Fault::Execution))
            );
            assert_eq!(input.probe.retirements(), 0);
            input.handle.allocation_lease().validate_live().unwrap();
            drop(input.handle);
            assert_eq!(input.probe.retirements(), 1);
        }
    }
}

#[test]
fn dimensional_extrema_never_dispatch_unregistered_input_to_selected_provider() {
    let selected = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        fault: Fault::Execution,
        reductions: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(selected));
    let input = TestAllocationOwner::new()
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    for name in ["min", "max"] {
        let error = futures::executor::block_on(crate::call_builtin_async(
            name,
            &[Value::GpuTensor(input.handle.clone())],
        ))
        .unwrap_err();
        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
        assert!(error.message().contains("exact owning provider"));
    }
    assert_eq!(selected.reductions.load(Ordering::SeqCst), 0);
    assert_eq!(selected.downloads.load(Ordering::SeqCst), 0);
    input.handle.allocation_lease().validate_live().unwrap();
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

#[test]
fn elementwise_extrema_propagate_execution_failure_and_keep_capability_fallback() {
    for name in ["min", "max"] {
        for form in 0..3 {
            for fault in [Fault::Execution, Fault::Unsupported] {
                let provider = Box::leak(Box::new(Provider {
                    owner: TestAllocationOwner::new(),
                    fault,
                    reductions: AtomicUsize::new(0),
                    downloads: AtomicUsize::new(0),
                }));
                let _selected = ThreadProviderGuard::set(Some(provider));
                let left = provider
                    .owner
                    .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
                    .unwrap();
                let right = provider
                    .owner
                    .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
                    .unwrap();
                let args = match form {
                    0 => vec![
                        Value::GpuTensor(left.handle.clone()),
                        Value::GpuTensor(right.handle.clone()),
                    ],
                    1 => vec![Value::GpuTensor(left.handle.clone()), Value::Num(2.0)],
                    _ => vec![Value::Num(2.0), Value::GpuTensor(right.handle.clone())],
                };
                let result = futures::executor::block_on(crate::call_builtin_async(name, &args));
                if matches!(fault, Fault::Execution) {
                    let error = result.unwrap_err();
                    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
                    let source = error.source.as_ref().unwrap();
                    let source = source
                        .downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
                        .unwrap();
                    assert_eq!(
                        std::error::Error::source(source)
                            .unwrap()
                            .downcast_ref::<std::io::Error>()
                            .unwrap()
                            .kind(),
                        std::io::ErrorKind::BrokenPipe,
                    );
                    assert_eq!(provider.reductions.load(Ordering::SeqCst), 1);
                    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
                } else {
                    let Value::Tensor(output) = result.unwrap() else {
                        panic!("host capability fallback");
                    };
                    assert_eq!(output.shape, vec![2, 1]);
                    let expected = if name == "max" && form != 0 { 2.0 } else { 0.0 };
                    assert_eq!(output.materialize_f64(), vec![expected, expected]);
                    assert_eq!(
                        provider.downloads.load(Ordering::SeqCst),
                        if form == 0 { 2 } else { 1 }
                    );
                }
                drop(args);
                for input in [left, right] {
                    assert_eq!(input.probe.retirements(), 0);
                    input.handle.allocation_lease().validate_live().unwrap();
                    drop(input.handle);
                    assert_eq!(input.probe.retirements(), 1);
                }
            }
        }
    }
}
