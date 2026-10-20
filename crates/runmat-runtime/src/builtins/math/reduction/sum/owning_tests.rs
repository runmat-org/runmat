use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelNumericDownloadFuture, GpuHandleProvenance, GpuTensorStorage,
    HostNumericDataOwned, HostNumericTensorOwned, HostNumericTensorView, HostTensorView,
    NumericElementType, ProviderPrecision, ThreadProviderGuard,
};
use std::{
    future::Future,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    task::{Context, Poll},
};
struct Provider {
    owner: TestAllocationOwner,
    cleaned: Mutex<Vec<RetirementProbe>>,
    fail: bool,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "sum cleaned-input lifecycle".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("execution failure must not gather") })
    }
    fn map_nan_to_zero(
        &self,
        input: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let result = self.owner.allocate_zeroed(
            input.shape().to_vec(),
            NumericElementType::F64,
            GpuTensorStorage::Real,
        )?;
        self.cleaned.lock().unwrap().push(result.probe);
        Ok(result.handle)
    }
    fn reduce_sum_dim<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.fail {
                Err(anyhow::Error::new(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "sum reduction failed",
                ))
                .into())
            } else {
                futures::future::pending().await
            }
        })
    }
}

#[test]
fn omitnan_sum_retires_cleaned_owner_on_cancel_and_failure_without_gathering() {
    for fail in [false, true] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            cleaned: Mutex::new(Vec::new()),
            fail,
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let parsed = ParsedArguments {
            invocation: SumInvocation::Sum,
            selection: DimSelection::Dim(1),
            nan_mode: ReductionNaN::Omit,
            output: OutputTemplate::Double,
        };
        let mut future = Box::pin(super::omit::sum_gpu_with_omitnan(
            input.handle.clone(),
            &parsed,
        ));
        let waker = futures::task::noop_waker();
        match (fail, future.as_mut().poll(&mut Context::from_waker(&waker))) {
            (false, Poll::Pending) => {}
            (true, Poll::Ready(Err(error))) => {
                assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                let source = error.source.as_ref().expect("provider source retained");
                let provider = source
                    .downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
                    .expect("typed provider execution carrier retained");
                let cause = std::error::Error::source(provider)
                    .expect("original provider cause retained")
                    .downcast_ref::<std::io::Error>()
                    .expect("original I/O error remains downcastable");
                assert_eq!(cause.kind(), std::io::ErrorKind::BrokenPipe);
                assert_eq!(cause.to_string(), "sum reduction failed");
            }
            (_, outcome) => panic!("unexpected production sum outcome: {outcome:?}"),
        }
        drop(future);
        let cleaned = provider.cleaned.lock().unwrap();
        assert_eq!(cleaned.len(), 1);
        assert_eq!(cleaned[0].retirements(), 1);
        assert_eq!(input.probe.retirements(), 0);
        input.handle.allocation_lease().validate_live().unwrap();
    }
}

struct OutputProvider {
    owner: TestAllocationOwner,
    precision: ProviderPrecision,
    calls: AtomicUsize,
    results: Mutex<Vec<RetirementProbe>>,
    malformed_map: bool,
    malformed_second_reduction: bool,
}

impl OutputProvider {
    fn new(malformed_map: bool, malformed_second_reduction: bool) -> Self {
        Self {
            owner: TestAllocationOwner::new(),
            precision: ProviderPrecision::F32,
            calls: AtomicUsize::new(0),
            results: Mutex::new(Vec::new()),
            malformed_map,
            malformed_second_reduction,
        }
    }

    fn output(&self, shape: Vec<usize>, element: NumericElementType) -> GpuTensorHandle {
        let allocation = self
            .owner
            .allocate_zeroed(shape, element, GpuTensorStorage::Real)
            .unwrap();
        self.results.lock().unwrap().push(allocation.probe);
        allocation.handle
    }
}

impl AccelProvider for OutputProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "sum typed output fixture".into()
    }
    fn precision(&self) -> ProviderPrecision {
        self.precision
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected legacy upload")
    }
    fn upload_numeric_raw(
        &self,
        _: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        Err(runmat_accelerate_api::unsupported_provider_operation(
            "F64 output is not supported by this F32 owner",
        )
        .into())
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unexpected legacy download") })
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            let len = handle.shape().iter().product();
            Ok(HostNumericTensorOwned {
                data: match handle.physical().element_type() {
                    NumericElementType::F32 => HostNumericDataOwned::F32(vec![0.0; len]),
                    NumericElementType::U8 => HostNumericDataOwned::U8(vec![0; len]),
                    other => anyhow::bail!("unexpected download element type: {other:?}"),
                },
                shape: handle.shape().to_vec(),
                storage: GpuTensorStorage::Real,
            })
        })
    }
    fn map_nan_to_zero(
        &self,
        input: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        Ok(self.output(
            input.shape().to_vec(),
            if self.malformed_map {
                match input.physical().element_type() {
                    NumericElementType::F64 => NumericElementType::F32,
                    NumericElementType::F32 => NumericElementType::F64,
                    other => {
                        return Err(
                            anyhow::anyhow!("unexpected NaN-map element type: {other:?}").into(),
                        )
                    }
                }
            } else {
                input.physical().element_type()
            },
        ))
    }
    fn reduce_sum_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if runmat_accelerate_api::handle_is_logical(input) {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "F32-only owner cannot accumulate logical values in double",
                )
                .into());
            }
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let mut shape = input.shape().to_vec();
            if !self.malformed_second_reduction || call == 0 {
                shape[dim] = 1;
            }
            Ok(self.output(shape, NumericElementType::F64))
        })
    }
}

#[test]
fn partial_vecdim_rejects_second_result_and_retires_first_without_touching_input() {
    let provider = Box::leak(Box::new(OutputProvider::new(false, true)));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_zeroed(
            vec![2, 2, 2],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let parsed = ParsedArguments {
        invocation: SumInvocation::Sum,
        selection: DimSelection::Vec(vec![1, 2]),
        nan_mode: ReductionNaN::Include,
        output: OutputTemplate::Default,
    };
    let error = futures::executor::block_on(sum_gpu(input.handle.clone(), &parsed)).unwrap_err();
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    let results = provider.results.lock().unwrap();
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|probe| probe.retirements() == 1));
    assert_eq!(input.probe.retirements(), 0);
    input.handle.allocation_lease().validate_live().unwrap();
}

#[test]
fn malformed_nan_map_fails_before_reduction_and_retires_result() {
    let provider = Box::leak(Box::new(OutputProvider {
        precision: ProviderPrecision::F64,
        ..OutputProvider::new(true, false)
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let parsed = ParsedArguments {
        invocation: SumInvocation::Sum,
        selection: DimSelection::Dim(1),
        nan_mode: ReductionNaN::Omit,
        output: OutputTemplate::Default,
    };
    let error = futures::executor::block_on(super::omit::sum_gpu_with_omitnan(
        input.handle.clone(),
        &parsed,
    ))
    .unwrap_err();
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    let results = provider.results.lock().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].retirements(), 1);
    assert_eq!(input.probe.retirements(), 0);
    input.handle.allocation_lease().validate_live().unwrap();
}

#[test]
fn logical_f32_resident_default_native_and_noop_adapt_to_double() {
    let provider = Box::leak(Box::new(OutputProvider::new(false, false)));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_logical_zeroed(vec![2, 2], ProviderPrecision::F32)
        .unwrap();
    let automatic = input
        .handle
        .with_provenance(GpuHandleProvenance::Automatic)
        .unwrap();
    for options in [
        vec![],
        vec![Value::from("native")],
        vec![Value::Int(IntValue::I32(5))],
        vec![Value::from("omitnan")],
    ] {
        let result = futures::executor::block_on(sum_with_nan_policy(
            Value::GpuTensor(automatic.clone()),
            crate::builtins::common::test_support::builtin_arguments(options),
            None,
        ))
        .expect("automatic logical sum");
        let Value::Tensor(result) = result else {
            panic!("unsupported F64 output must remain on host for automatic placement");
        };
        assert_eq!(result.numeric_dtype(), NumericDType::F64);
        assert!(result.materialize_f64().iter().all(|&value| value == 0.0));
    }
    assert_eq!(input.probe.retirements(), 0);
    input.handle.allocation_lease().validate_live().unwrap();

    let explicit = input
        .handle
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    let error = futures::executor::block_on(sum_with_nan_policy(
        Value::GpuTensor(explicit),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
        None,
    ))
    .unwrap_err();
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(error.message().contains("restore"));
}
