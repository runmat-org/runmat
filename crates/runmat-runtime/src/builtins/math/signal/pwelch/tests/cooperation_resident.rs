//! Provider-stage suspension must retain admitted outputs and retire them on
//! cancellation or abandonment without touching the borrowed source.

use super::*;
use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
use futures::task::noop_waker;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView,
    NumericElementType, ProviderExecutionFuture, ProviderSpectralResult, ReduceDimResult,
    ThreadProviderGuard,
};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::future::Future;
use std::num::NonZeroU64;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};
use std::task::{Context, Poll};

#[derive(Clone, Copy)]
enum SuspendAfter {
    Spectral,
    Mean,
}

struct StageProvider {
    owner: TestAllocationOwner,
    unsupported_extrema: bool,
    spectral_calls: AtomicUsize,
    mean_calls: AtomicUsize,
    extrema_calls: AtomicUsize,
    download_calls: AtomicUsize,
    outputs: Mutex<Vec<RetirementProbe>>,
}

impl StageProvider {
    fn new() -> Self {
        Self::with_unsupported_extrema(false)
    }

    fn with_unsupported_extrema(unsupported_extrema: bool) -> Self {
        Self {
            owner: TestAllocationOwner::new(),
            unsupported_extrema,
            spectral_calls: AtomicUsize::new(0),
            mean_calls: AtomicUsize::new(0),
            extrema_calls: AtomicUsize::new(0),
            download_calls: AtomicUsize::new(0),
            outputs: Mutex::new(Vec::new()),
        }
    }

    fn output(&self, shape: Vec<usize>, storage: GpuTensorStorage) -> GpuTensorHandle {
        let allocation = self
            .owner
            .allocate_zeroed(shape, NumericElementType::F64, storage)
            .unwrap();
        self.outputs.lock().unwrap().push(allocation.probe);
        allocation.handle
    }
}

impl AccelProvider for StageProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "pwelch stage lifecycle fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.download_calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("unexpected download") })
    }
    fn uniform_spectral_estimate<'a>(
        &'a self,
        request: &'a ProviderSpectralRequest<'a>,
    ) -> ProviderExecutionFuture<'a, ProviderSpectralResult> {
        Box::pin(async move {
            self.spectral_calls.fetch_add(1, Ordering::SeqCst);
            let rows = request.nfft / 2 + 1;
            let shape = vec![rows, request.frame_count];
            Ok(ProviderSpectralResult {
                s: self.output(shape.clone(), GpuTensorStorage::ComplexInterleaved),
                ps: self.output(shape, GpuTensorStorage::Real),
                rows,
                cols: request.frame_count,
            })
        })
    }
    fn reduce_mean_nd<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        _: &'a [usize],
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async move {
            let result: anyhow::Result<GpuTensorHandle> = async move {
                self.mean_calls.fetch_add(1, Ordering::SeqCst);
                let mut shape = input.shape().to_vec();
                shape[1] = 1;
                Ok(self.output(shape, GpuTensorStorage::Real))
            }
            .await;
            result.map(Some).map_err(|source| {
                runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
            })
        })
    }
    fn reduce_max_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        _: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ReduceDimResult> {
        Box::pin(async move {
            self.extrema_calls.fetch_add(1, Ordering::SeqCst);
            if self.unsupported_extrema {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "extrema intentionally unsupported",
                )
                .into());
            }
            let mut shape = input.shape().to_vec();
            shape[1] = 1;
            Ok(ReduceDimResult {
                values: self.output(shape.clone(), GpuTensorStorage::Real),
                indices: self.output(shape, GpuTensorStorage::Real),
            })
        })
    }
    fn reduce_min_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        axis: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ReduceDimResult> {
        self.reduce_max_dim(input, axis)
    }
}

#[test]
fn unsupported_extrema_download_failure_keeps_owned_welch_cause() {
    let provider = Box::leak(Box::new(StageProvider::with_unsupported_extrema(true)));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let source = provider
        .owner
        .allocate_zeroed(vec![8, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let cancellation = CancellationSource::new();
    let mut cooperation = SignalCooperation::with_budget(
        CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::NeverYield,
            cancellation.token(),
            || Box::pin(async { Ok(()) }),
        ),
        BUILTIN_NAME,
    );
    let options = PwelchOptions {
        window: vec![1.0; 4],
        noverlap: 0,
        grid: FrequencyGrid::Uniform { nfft: 8 },
        fs: Some(8.0),
        range: FrequencyRange::Onesided,
        scale: SpectrumScale::Psd,
        trace: WelchTrace::MaxHold,
    };
    let error = futures::executor::block_on(output_gpu(
        &source.handle,
        8,
        1,
        &options,
        &[],
        &mut cooperation,
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:pwelch:Internal"));
    assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    assert!(error.source.is_some());
    assert_eq!(provider.spectral_calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.extrema_calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.download_calls.load(Ordering::SeqCst), 1);
    assert_eq!(source.probe.retirements(), 0);
}

#[test]
fn invalid_signal_carrier_precedes_resident_control_download() {
    let provider = Box::leak(Box::new(StageProvider::new()));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let resident_window = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let source = CancellationSource::new();
    for invalid_signal in [
        Value::from("invalid signal"),
        Value::Tensor(runmat_value::Tensor::new(Vec::new(), vec![0, 1]).unwrap()),
        Value::Tensor(runmat_value::Tensor::new(vec![1.0], vec![1, 1, 1]).unwrap()),
    ] {
        let mut cooperation = SignalCooperation::with_budget(
            CpuCooperationBudget::with_test_turn(
                CpuCooperationPolicy::NeverYield,
                source.token(),
                || Box::pin(async { Ok(()) }),
            ),
            BUILTIN_NAME,
        );
        let error = futures::executor::block_on(evaluate_with_cooperation(
            invalid_signal,
            crate::call::FunctionArgumentView::from_slice(&[Value::GpuTensor(
                resident_window.handle.clone(),
            )]),
            &mut cooperation,
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:pwelch:InvalidSignal"));
        assert_eq!(provider.download_calls.load(Ordering::SeqCst), 0);
        assert_eq!(resident_window.probe.retirements(), 0);
    }
    let invalid_resident_signal = provider
        .owner
        .allocate_zeroed(
            vec![1, 1, 1],
            NumericElementType::F64,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let error = futures::executor::block_on(evaluate(
        Value::GpuTensor(invalid_resident_signal.handle.clone()),
        &[Value::GpuTensor(resident_window.handle.clone())],
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:pwelch:InvalidSignal"));
    assert_eq!(provider.download_calls.load(Ordering::SeqCst), 0);
    assert_eq!(invalid_resident_signal.probe.retirements(), 0);
    assert_eq!(resident_window.probe.retirements(), 0);
    let error = futures::executor::block_on(evaluate(
        Value::Complex(1.0, 1.0),
        &[
            Value::GpuTensor(resident_window.handle.clone()),
            Value::from("onesided"),
        ],
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:pwelch:InvalidRange"));
    assert_eq!(provider.download_calls.load(Ordering::SeqCst), 0);
    assert_eq!(resident_window.probe.retirements(), 0);
}

#[test]
fn resident_hold_trace_uses_typed_extrema_and_keeps_borrowed_source_live() {
    for trace in [WelchTrace::MaxHold, WelchTrace::MinHold] {
        let provider = Box::leak(Box::new(StageProvider::new()));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let source_input = provider
            .owner
            .allocate_zeroed(vec![8, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let cancel_source = CancellationSource::new();
        let mut cooperation = SignalCooperation::with_budget(
            CpuCooperationBudget::with_test_turn(
                CpuCooperationPolicy::NeverYield,
                cancel_source.token(),
                || Box::pin(async { Ok(()) }),
            ),
            BUILTIN_NAME,
        );
        let options = PwelchOptions {
            window: vec![1.0; 4],
            noverlap: 0,
            grid: FrequencyGrid::Uniform { nfft: 8 },
            fs: Some(8.0),
            range: FrequencyRange::Onesided,
            scale: SpectrumScale::Psd,
            trace,
        };
        let _count = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let value = futures::executor::block_on(output_gpu(
            &source_input.handle,
            8,
            1,
            &options,
            &[],
            &mut cooperation,
        ))
        .unwrap()
        .unwrap();
        let Value::OutputList(outputs) = value else {
            panic!("expected one requested output");
        };
        assert!(matches!(outputs.as_slice(), [Value::GpuTensor(_)]));
        assert_eq!(provider.extrema_calls.load(Ordering::SeqCst), 1);
        assert_eq!(provider.mean_calls.load(Ordering::SeqCst), 0);
        assert_eq!(provider.download_calls.load(Ordering::SeqCst), 0);
        assert_eq!(source_input.probe.retirements(), 0);
    }
}

fn pending_budget(
    provider: &'static StageProvider,
    source: &CancellationSource,
    stage: SuspendAfter,
) -> CpuCooperationBudget {
    CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || -> crate::context::RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
            let reached = match stage {
                SuspendAfter::Spectral => provider.spectral_calls.load(Ordering::SeqCst) > 0,
                SuspendAfter::Mean => provider.mean_calls.load(Ordering::SeqCst) > 0,
            };
            if reached {
                Box::pin(std::future::pending::<Result<(), ExecutionServiceError>>())
            } else {
                Box::pin(async { Ok(()) })
            }
        },
    )
}

#[test]
fn admitted_provider_outputs_retire_after_pending_cancel_or_drop() {
    for stage in [SuspendAfter::Spectral, SuspendAfter::Mean] {
        for cancel in [false, true] {
            let provider = Box::leak(Box::new(StageProvider::new()));
            let _selected = ThreadProviderGuard::set(Some(provider));
            let source_input = provider
                .owner
                .allocate_zeroed(vec![8, 1], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let source = CancellationSource::new();
            let mut cooperation = SignalCooperation::with_budget(
                pending_budget(provider, &source, stage),
                BUILTIN_NAME,
            );
            let options = PwelchOptions {
                window: vec![1.0; 4],
                noverlap: 0,
                grid: FrequencyGrid::Uniform { nfft: 8 },
                fs: Some(8.0),
                range: FrequencyRange::Onesided,
                scale: SpectrumScale::Psd,
                trace: WelchTrace::Mean,
            };
            let mut future = Box::pin(output_gpu(
                &source_input.handle,
                8,
                1,
                &options,
                &[],
                &mut cooperation,
            ));
            let waker = noop_waker();
            let mut context = Context::from_waker(&waker);
            assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
            assert_eq!(provider.spectral_calls.load(Ordering::SeqCst), 1);
            assert_eq!(
                provider.mean_calls.load(Ordering::SeqCst),
                usize::from(matches!(stage, SuspendAfter::Mean))
            );
            let expected_outputs = if matches!(stage, SuspendAfter::Mean) {
                3
            } else {
                2
            };
            assert_eq!(provider.outputs.lock().unwrap().len(), expected_outputs);
            if cancel {
                source.cancel(CancellationReason::User);
                let Poll::Ready(Err(error)) = future.as_mut().poll(&mut context) else {
                    panic!("cancelled provider stage must not publish output");
                };
                assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
                assert!(matches!(
                    error
                        .source
                        .as_deref()
                        .and_then(|cause| cause.downcast_ref::<ExecutionServiceError>()),
                    Some(ExecutionServiceError::Cancelled)
                ));
            }
            drop(future);
            for probe in provider.outputs.lock().unwrap().iter() {
                assert_eq!(probe.retirements(), 1);
            }
            assert_eq!(source_input.probe.retirements(), 0);
            drop(source_input.handle);
            assert_eq!(source_input.probe.retirements(), 1);
        }
    }
}
