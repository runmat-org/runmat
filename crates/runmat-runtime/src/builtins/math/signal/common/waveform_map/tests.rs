use super::*;
use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
use futures::executor::block_on;
use futures::task::noop_waker;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_value::Value;
use std::cell::Cell;
use std::future::Future;
use std::num::NonZeroU64;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

#[test]
fn canonical_pulse_wave_bindings_observe_invocation_cancellation() {
    use crate::builtin::RuntimeBuiltinInvocation;
    use crate::context::RuntimeContext;
    use crate::execution::RuntimeExecutionService;

    let source = CancellationSource::new();
    let runtime =
        RuntimeContext::with_cancellation(Rc::new(RuntimeExecutionService::new()), source.token());
    source.cancel(CancellationReason::User);
    for name in ["rectpuls", "tripuls", "sinc", "square", "sawtooth"] {
        let bindings = crate::builtin::runtime_builtin_bindings_by_name(name);
        assert_eq!(bindings.len(), 1, "one canonical {name} binding");
        let args = [Value::Tensor(
            Tensor::new(vec![0.0; 8193], vec![1, 8193]).unwrap(),
        )];
        let error = block_on((bindings[0].implementation)(RuntimeBuiltinInvocation::new(
            &args,
            Some(runtime.clone()),
            Some(1),
        )))
        .expect_err("cancelled invocation must not publish a waveform");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"), "{name}");
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(
            error.failure_origin,
            crate::runtime_error::RuntimeFailureOrigin::ExecutionControlCancellation
        );
        assert!(matches!(
            error
                .source
                .as_deref()
                .and_then(|cause| cause.downcast_ref::<ExecutionServiceError>()),
            Some(ExecutionServiceError::Cancelled)
        ));
    }
}

struct PendingTurn(Rc<Cell<usize>>);

impl Future for PendingTurn {
    type Output = Result<(), ExecutionServiceError>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

impl Drop for PendingTurn {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

#[test]
fn single_and_double_mapping_retain_class_shape_and_values() {
    let input = Tensor::from_f32(vec![-2.0, 0.0, 2.0], vec![1, 3]).unwrap();
    let single = block_on(map_real_waveform(
        input.clone(),
        WaveformOutputMode::PreserveSingle,
        |x| x + 1.0,
        &mut SignalCooperation::new(None, "tripuls"),
    ))
    .unwrap()
    .unwrap();
    assert_eq!(single.shape(), [1, 3]);
    assert_eq!(
        single.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![-1.0, 1.0, 3.0])
    );
    let double = block_on(map_real_waveform(
        input,
        WaveformOutputMode::Double,
        |x| x + 1.0,
        &mut SignalCooperation::new(None, "rectpuls"),
    ))
    .unwrap()
    .unwrap();
    assert_eq!(
        double.into_numeric_storage().unwrap(),
        NumericStorage::F64(vec![-1.0, 1.0, 3.0])
    );
}

#[test]
fn pending_turn_cancel_and_drop_do_not_publish_waveform() {
    for cancel in [true, false] {
        let source = CancellationSource::new();
        let dropped = Rc::new(Cell::new(0));
        let dropped_turn = dropped.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            move || Box::pin(PendingTurn(dropped_turn.clone())),
        );
        let evaluated = Rc::new(Cell::new(0));
        let count = evaluated.clone();
        let tensor = Tensor::new(vec![1.0; 8193], vec![1, 8193]).unwrap();
        let mut cooperation = SignalCooperation::with_budget(budget, "square");
        let mut work = Box::pin(map_real_waveform(
            tensor,
            WaveformOutputMode::Double,
            move |x| {
                count.set(count.get() + 1);
                x
            },
            &mut cooperation,
        ));
        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(matches!(work.as_mut().poll(&mut context), Poll::Pending));
        assert_eq!(evaluated.get(), MAP_CHUNK);
        if cancel {
            source.cancel(CancellationReason::User);
            let Poll::Ready(Err(error)) = work.as_mut().poll(&mut context) else {
                panic!("expected cancellation");
            };
            assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
            assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
            assert!(matches!(
                error
                    .source
                    .as_deref()
                    .and_then(|source| source.downcast_ref::<ExecutionServiceError>()),
                Some(ExecutionServiceError::Cancelled)
            ));
        }
        drop(work);
        assert_eq!(dropped.get(), 1);
        assert_eq!(evaluated.get(), MAP_CHUNK);
    }
}
