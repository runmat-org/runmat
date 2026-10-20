//! End-to-end owner, effect, and cancellation witnesses for the public Single
//! adaptation after sum's double accumulator.

use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelNumericDownloadFuture, GpuTensorStorage, HostTensorView,
    NumericElementType, ThreadProviderGuard,
};
use runmat_value::IntValue;
use std::{
    error::Error as _,
    future::Future,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    task::{Context, Poll},
};

#[derive(Clone, Copy)]
enum CastOutcome {
    Valid,
    OperationalFailure,
    WrongPhysicalClass,
    BorrowedSource,
    CleanupFailure,
    Pending,
}

struct StageProvider {
    owner: TestAllocationOwner,
    source: Mutex<Option<GpuTensorHandle>>,
    reductions: Mutex<Vec<RetirementProbe>>,
    casts: Mutex<Vec<RetirementProbe>>,
    reduction_calls: AtomicUsize,
    cast_calls: AtomicUsize,
    download_calls: AtomicUsize,
    cast_outcome: CastOutcome,
    pending_second_reduction: bool,
}

impl StageProvider {
    fn new(cast_outcome: CastOutcome, pending_second_reduction: bool) -> Self {
        Self {
            owner: TestAllocationOwner::new(),
            source: Mutex::new(None),
            reductions: Mutex::new(Vec::new()),
            casts: Mutex::new(Vec::new()),
            reduction_calls: AtomicUsize::new(0),
            cast_calls: AtomicUsize::new(0),
            download_calls: AtomicUsize::new(0),
            cast_outcome,
            pending_second_reduction,
        }
    }

    fn source(&self, shape: Vec<usize>) -> runmat_accelerate_api::test_support::TestAllocation {
        let source = self
            .owner
            .allocate_zeroed(shape, NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap();
        *self.source.lock().unwrap() = Some(source.handle.clone());
        source
    }

    fn output(&self, shape: Vec<usize>, class: NumericElementType, cast: bool) -> GpuTensorHandle {
        let output = self
            .owner
            .allocate_zeroed(shape, class, GpuTensorStorage::Real)
            .unwrap();
        if cast {
            if matches!(self.cast_outcome, CastOutcome::CleanupFailure) {
                output
                    .probe
                    .fail_retirement("cast result retirement failed");
            }
            self.casts.lock().unwrap().push(output.probe);
        } else {
            self.reductions.lock().unwrap().push(output.probe);
        }
        output.handle
    }
}

impl AccelProvider for StageProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "sum public Single stage fixture".into()
    }

    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.download_calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("sum must not gather in this fixture") })
    }

    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.download_calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("sum must not gather numeric data in this fixture") })
    }

    fn reduce_sum_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let call = self.reduction_calls.fetch_add(1, Ordering::SeqCst);
            if self.pending_second_reduction && call == 1 {
                return futures::future::pending().await;
            }
            let mut shape = input.shape().to_vec();
            shape[dim] = 1;
            Ok(self.output(shape, NumericElementType::F64, false))
        })
    }

    fn unary_single<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.cast_calls.fetch_add(1, Ordering::SeqCst);
            match self.cast_outcome {
                CastOutcome::Valid => {
                    Ok(self.output(input.shape().to_vec(), NumericElementType::F32, true))
                }
                CastOutcome::OperationalFailure => Err(anyhow::Error::new(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "single cast failed",
                ))
                .into()),
                CastOutcome::WrongPhysicalClass | CastOutcome::CleanupFailure => {
                    Ok(self.output(input.shape().to_vec(), NumericElementType::F64, true))
                }
                CastOutcome::BorrowedSource => {
                    Ok(self.source.lock().unwrap().as_ref().unwrap().clone())
                }
                CastOutcome::Pending => futures::future::pending().await,
            }
        })
    }
}

fn sum_first_axis(source: GpuTensorHandle) -> BuiltinResult<Value> {
    block_on(sum_with_nan_policy(
        Value::GpuTensor(source),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::I32(
            1,
        ))]),
        None,
    ))
}

#[test]
fn resident_public_single_cast_succeeds_without_download() {
    let provider = Box::leak(Box::new(StageProvider::new(CastOutcome::Valid, false)));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let source = provider.source(vec![2, 2]);
    let result = sum_first_axis(source.handle.clone()).unwrap();
    let Value::GpuTensor(output) = result else {
        panic!("supported cast should publish a resident result")
    };
    assert_eq!(output.physical().element_type(), NumericElementType::F32);
    assert_eq!(provider.reduction_calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.cast_calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.download_calls.load(Ordering::SeqCst), 0);
    assert_eq!(provider.reductions.lock().unwrap()[0].retirements(), 1);
    assert_eq!(source.probe.retirements(), 0);
    output.release().unwrap();
    assert_eq!(provider.casts.lock().unwrap()[0].retirements(), 1);
}

#[test]
fn resident_public_single_cast_failure_and_malformed_results_never_gather() {
    for outcome in [
        CastOutcome::OperationalFailure,
        CastOutcome::WrongPhysicalClass,
        CastOutcome::BorrowedSource,
        CastOutcome::CleanupFailure,
    ] {
        let provider = Box::leak(Box::new(StageProvider::new(outcome, false)));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let source = provider.source(vec![1, 2]);
        let error = sum_first_axis(source.handle.clone()).unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(provider.reduction_calls.load(Ordering::SeqCst), 1);
        assert_eq!(provider.cast_calls.load(Ordering::SeqCst), 1);
        assert_eq!(provider.download_calls.load(Ordering::SeqCst), 0);
        assert_eq!(provider.reductions.lock().unwrap()[0].retirements(), 1);
        assert_eq!(source.probe.retirements(), 0);
        source.handle.allocation_lease().validate_live().unwrap();
        match outcome {
            CastOutcome::OperationalFailure => {
                let source = error.source.as_ref().expect("operational cause retained");
                let carrier = source
                    .downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
                    .expect("provider carrier retained");
                assert_eq!(carrier.source().unwrap().to_string(), "single cast failed");
            }
            CastOutcome::BorrowedSource => assert!(provider.casts.lock().unwrap().is_empty()),
            CastOutcome::WrongPhysicalClass => {
                assert_eq!(provider.casts.lock().unwrap()[0].retirements(), 1);
            }
            CastOutcome::CleanupFailure => {
                assert!(error.source.is_some(), "typed cleanup cause retained");
                assert_eq!(provider.casts.lock().unwrap()[0].retirements(), 1);
            }
            CastOutcome::Valid | CastOutcome::Pending => unreachable!(),
        }
    }
}

#[test]
fn abandoned_actual_sum_futures_retire_cast_and_multistage_intermediates() {
    let cast_provider = Box::leak(Box::new(StageProvider::new(CastOutcome::Pending, false)));
    {
        let _selected = ThreadProviderGuard::set(Some(cast_provider));
        let source = cast_provider.source(vec![2, 2]);
        let mut future = Box::pin(sum_with_nan_policy(
            Value::GpuTensor(source.handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::I32(1),
            )]),
            None,
        ));
        let waker = futures::task::noop_waker();
        assert!(matches!(
            future.as_mut().poll(&mut Context::from_waker(&waker)),
            Poll::Pending
        ));
        assert_eq!(cast_provider.reduction_calls.load(Ordering::SeqCst), 1);
        assert_eq!(cast_provider.cast_calls.load(Ordering::SeqCst), 1);
        drop(future);
        assert_eq!(cast_provider.reductions.lock().unwrap()[0].retirements(), 1);
        assert_eq!(cast_provider.download_calls.load(Ordering::SeqCst), 0);
        assert_eq!(source.probe.retirements(), 0);
        source.handle.allocation_lease().validate_live().unwrap();
    }

    let staged_provider = Box::leak(Box::new(StageProvider::new(CastOutcome::Valid, true)));
    let _selected = ThreadProviderGuard::set(Some(staged_provider));
    let source = staged_provider.source(vec![2, 2, 2]);
    let dims = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
    let mut future = Box::pin(sum_with_nan_policy(
        Value::GpuTensor(source.handle.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(dims)]),
        None,
    ));
    let waker = futures::task::noop_waker();
    assert!(matches!(
        future.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert_eq!(staged_provider.reduction_calls.load(Ordering::SeqCst), 2);
    assert_eq!(staged_provider.cast_calls.load(Ordering::SeqCst), 0);
    drop(future);
    assert_eq!(
        staged_provider.reductions.lock().unwrap()[0].retirements(),
        1
    );
    assert_eq!(staged_provider.download_calls.load(Ordering::SeqCst), 0);
    assert_eq!(source.probe.retirements(), 0);
    source.handle.allocation_lease().validate_live().unwrap();
}
