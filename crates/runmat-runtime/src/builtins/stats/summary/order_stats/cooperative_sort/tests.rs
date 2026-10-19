use super::*;
use crate::execution::CpuCooperationPolicy;
use futures::task::noop_waker;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::cell::Cell;
use std::future::Future;
use std::num::NonZeroU64;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

#[test]
fn sort_errors_preserve_allocation_and_execution_sources() {
    use std::error::Error;

    let allocation = Vec::<u8>::new().try_reserve(usize::MAX).unwrap_err();
    let expected = allocation.to_string();
    let error = CooperativeSortError::Resource(allocation);
    assert_eq!(
        error.to_string(),
        format!("sort allocation failed: {expected}")
    );
    assert!(error
        .source()
        .unwrap()
        .downcast_ref::<std::collections::TryReserveError>()
        .is_some());
    assert_eq!(error.source().unwrap().to_string(), expected);

    let error = CooperativeSortError::Service(ExecutionServiceError::Cancelled);
    assert_eq!(
        error.to_string(),
        "sort execution failed: execution was cancelled"
    );
    assert_eq!(
        error
            .source()
            .unwrap()
            .downcast_ref::<ExecutionServiceError>(),
        Some(&ExecutionServiceError::Cancelled),
    );
}

#[test]
fn stable_order_matches_library_sort_at_empty_single_and_run_boundaries() {
    for count in [0, 1, RUN_LEN - 1, RUN_LEN, RUN_LEN + 1, 2 * RUN_LEN + 3] {
        let mut actual = (0..count)
            .map(|index| match index % 7 {
                0 => -0.0,
                1 => 0.0,
                2 => f64::NEG_INFINITY,
                3 => f64::INFINITY,
                other => other as f64,
            })
            .rev()
            .collect::<Vec<_>>();
        let mut expected = actual.clone();
        expected.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
        futures::executor::block_on(stable_sort_non_nan_f64(&mut actual, None)).unwrap();
        let actual_bits = actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let expected_bits = expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(actual_bits, expected_bits, "count {count}");
    }
}

#[test]
fn cancellation_and_service_failure_keep_typed_cause_during_run_and_merge() {
    for (count, fail_on_turn) in [(RUN_LEN, 1), (RUN_LEN + 1, 3)] {
        let source = CancellationSource::new();
        let turns = Rc::new(Cell::new(0));
        let counted = turns.clone();
        let mut budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            move || {
                counted.set(counted.get() + 1);
                let fail = counted.get() == fail_on_turn;
                Box::pin(async move {
                    if fail {
                        Err(ExecutionServiceError::Infrastructure("turn failed".into()))
                    } else {
                        Ok(())
                    }
                })
            },
        );
        let mut values = (0..count).rev().map(|value| value as f64).collect();
        let error =
            futures::executor::block_on(stable_sort_non_nan_f64(&mut values, Some(&mut budget)))
                .unwrap_err();
        assert!(matches!(
            error,
            CooperativeSortError::Service(ExecutionServiceError::Infrastructure(_))
        ));
        assert_eq!(turns.get(), fail_on_turn);
    }

    let source = Rc::new(CancellationSource::new());
    let cancel_on_turn = source.clone();
    let mut budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            cancel_on_turn.cancel(CancellationReason::User);
            Box::pin(async { Ok(()) })
        },
    );
    let mut values = vec![2.0, 1.0];
    let error =
        futures::executor::block_on(stable_sort_non_nan_f64(&mut values, Some(&mut budget)))
            .unwrap_err();
    assert!(matches!(
        error,
        CooperativeSortError::Service(ExecutionServiceError::Cancelled)
    ));
}

struct PendingTurn(Rc<Cell<bool>>);

impl Future for PendingTurn {
    type Output = Result<(), ExecutionServiceError>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

impl Drop for PendingTurn {
    fn drop(&mut self) {
        self.0.set(true);
    }
}

#[test]
fn abandoning_a_pending_merge_turn_releases_the_service_future() {
    let source = CancellationSource::new();
    let turns = Rc::new(Cell::new(0));
    let dropped = Rc::new(Cell::new(false));
    let counted = turns.clone();
    let drop_flag = dropped.clone();
    let mut budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            counted.set(counted.get() + 1);
            if counted.get() == 3 {
                Box::pin(PendingTurn(drop_flag.clone()))
            } else {
                Box::pin(async { Ok(()) })
            }
        },
    );
    let mut values = (0..RUN_LEN + 1).rev().map(|value| value as f64).collect();
    let mut operation = Box::pin(stable_sort_non_nan_f64(&mut values, Some(&mut budget)));
    let waker = noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(
        operation.as_mut().poll(&mut context),
        Poll::Pending
    ));
    assert_eq!(turns.get(), 3);
    assert!(!dropped.get());
    drop(operation);
    assert!(dropped.get());
}
