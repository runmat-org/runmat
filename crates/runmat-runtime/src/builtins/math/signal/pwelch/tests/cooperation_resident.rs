//! Provider-stage suspension must retain admitted outputs and retire them on
//! cancellation or abandonment without touching the borrowed source.

use super::*;
use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
use futures::task::noop_waker;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView,
    NumericElementType, ProviderExecutionFuture, ProviderSpectralResult, ThreadProviderGuard,
};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::future::Future;
use std::num::NonZeroU64;
use std::pin::Pin;
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
    spectral_calls: AtomicUsize,
    mean_calls: AtomicUsize,
    outputs: Mutex<Vec<RetirementProbe>>,
}

impl StageProvider {
    fn new() -> Self {
        Self {
            owner: TestAllocationOwner::new(),
            spectral_calls: AtomicUsize::new(0),
            mean_calls: AtomicUsize::new(0),
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
    ) -> runmat_accelerate_api::AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.mean_calls.fetch_add(1, Ordering::SeqCst);
            let mut shape = input.shape().to_vec();
            shape[1] = 1;
            Ok(self.output(shape, GpuTensorStorage::Real))
        })
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
