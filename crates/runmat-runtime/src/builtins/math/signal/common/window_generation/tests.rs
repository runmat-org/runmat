use super::*;
use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
use futures::executor::block_on;
use futures::task::noop_waker;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::cell::Cell;
use std::future::Future;
use std::num::NonZeroU64;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

fn options(len: usize) -> WindowOptions {
    WindowOptions {
        len,
        sampling: WindowSampling::Symmetric,
        output_type: super::super::WindowOutputType::Double,
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
fn forced_turn_preserves_coefficient_values_and_count() {
    let expected = block_on(window_tensor(
        options(8193),
        |idx, _| idx as f64 / 3.0,
        &mut SignalCooperation::new(None, "hann"),
    ))
    .unwrap()
    .unwrap();
    let source = CancellationSource::new();
    let turns = Rc::new(Cell::new(0));
    let count = turns.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            count.set(count.get() + 1);
            Box::pin(async { Ok(()) })
        },
    );
    let actual = block_on(window_tensor(
        options(8193),
        |idx, _| idx as f64 / 3.0,
        &mut SignalCooperation::with_budget(budget, "hann"),
    ))
    .unwrap()
    .unwrap();
    let (Value::Tensor(expected), Value::Tensor(actual)) = (expected, actual) else {
        panic!("expected host tensors");
    };
    assert_eq!(actual.shape(), [8193, 1]);
    assert_eq!(actual.materialize_f64(), expected.materialize_f64());
    assert_eq!(turns.get(), 3);
}

#[test]
fn single_window_uses_the_same_f64_coefficient_formula_before_native_cast() {
    let mut requested = options(8193);
    requested.output_type = WindowOutputType::Single;
    let source = CancellationSource::new();
    let turns = Rc::new(Cell::new(0));
    let count = turns.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            count.set(count.get() + 1);
            Box::pin(async { Ok(()) })
        },
    );
    let output = block_on(window_tensor(
        requested,
        |idx, total| idx as f64 / total as f64,
        &mut SignalCooperation::with_budget(budget, "hann"),
    ))
    .unwrap()
    .unwrap();
    let Value::Tensor(output) = output else {
        panic!("expected native single tensor");
    };
    assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);
    let expected = (0..8193)
        .map(|idx| (idx as f64 / 8193.0) as f32)
        .collect::<Vec<_>>();
    assert_eq!(
        output.into_numeric_storage().unwrap(),
        NumericStorage::F32(expected)
    );
    assert_eq!(turns.get(), 3);
}

#[test]
fn pending_turn_cancellation_and_abandonment_do_not_publish_output() {
    for cancel in [true, false] {
        let source = CancellationSource::new();
        let dropped_turns = Rc::new(Cell::new(0));
        let dropped = dropped_turns.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            move || Box::pin(PendingTurn(dropped.clone())),
        );
        let completed = Rc::new(Cell::new(0));
        let count = completed.clone();
        let mut cooperation = SignalCooperation::with_budget(budget, "blackman");
        let mut work = Box::pin(window_tensor(
            options(8193),
            move |_, _| {
                count.set(count.get() + 1);
                0.5
            },
            &mut cooperation,
        ));
        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(matches!(work.as_mut().poll(&mut context), Poll::Pending));
        assert_eq!(completed.get(), COEFFICIENT_CHUNK);
        if cancel {
            source.cancel(CancellationReason::User);
            let result = work.as_mut().poll(&mut context);
            let Poll::Ready(Err(error)) = result else {
                panic!("cancelled work must not publish a window");
            };
            assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
            assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
            assert!(matches!(
                error
                    .source
                    .as_deref()
                    .and_then(|cause| cause.downcast_ref::<ExecutionServiceError>()),
                Some(ExecutionServiceError::Cancelled)
            ));
        }
        drop(work);
        assert_eq!(dropped_turns.get(), 1);
        assert_eq!(completed.get(), COEFFICIENT_CHUNK);
    }
}

#[test]
fn service_failure_retains_owning_source() {
    let source = CancellationSource::new();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        || Box::pin(async { Err(ExecutionServiceError::Infrastructure("turn failed".into())) }),
    );
    let error = block_on(window_tensor(
        options(4097),
        |_, _| 1.0,
        &mut SignalCooperation::with_budget(budget, "hamming"),
    ))
    .unwrap_err();
    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
    assert_eq!(error.context.builtin.as_deref(), Some("hamming"));
    assert!(matches!(
        error
            .source
            .as_deref()
            .and_then(|cause| cause.downcast_ref::<ExecutionServiceError>()),
        Some(ExecutionServiceError::Infrastructure(_))
    ));
}
