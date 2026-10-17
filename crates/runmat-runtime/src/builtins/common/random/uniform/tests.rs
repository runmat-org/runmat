use super::*;
use crate::builtins::common::random::{
    enter_parallel_random_stream, reserve_parallel_random_streams, set_seed, snapshot, test_guard,
    with_rng_state_in, RandomState, DEFAULT_USER_SEED,
};
use crate::execution::{CpuCooperationPolicy, ExecutionServiceError, RuntimeExecutionService};
use futures::executor::block_on;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::cell::Cell;
use std::future::Future;
use std::num::NonZeroU64;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

fn context() -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
}

fn serial_budget(context: &RuntimeContext) -> CpuCooperationBudget {
    CpuCooperationBudget::new(context.execution().clone(), context.cancellation())
}

fn generate_uniform(len: usize, label: &str) -> BuiltinResult<Vec<f64>> {
    let mut turn = super::super::turn::try_acquire(None, label)?;
    super::generate_uniform(&mut turn, len, label)
}

async fn generate_uniform_cooperative(
    len: usize,
    label: &str,
    context: Option<&RuntimeContext>,
    budget: &mut CpuCooperationBudget,
) -> BuiltinResult<Vec<f64>> {
    let mut turn = super::super::turn::acquire(context, label).await?;
    super::generate_uniform_cooperative(&mut turn, len, label, context, budget).await
}

fn state(context: &RuntimeContext) -> RandomState {
    with_rng_state_in(Some(context), "rng test", |rng| rng.state.clone()).unwrap()
}

fn advanced(mut state: RandomState, count: u64) -> RandomState {
    state.advance_uniforms(count, None).unwrap();
    state
}

#[test]
fn jump_ahead_matches_exact_serial_draws_and_snapshot() {
    let context = context();
    {
        let _scope = context.enter();
        set_seed(41).unwrap();
    }
    let start = state(&context);
    let mut expected_state = start.clone();
    expected_state.prepare_mutation(None).unwrap();
    let expected: Vec<f64> = (0..131_071)
        .map(|_| next_uniform_state(&mut expected_state))
        .collect();
    assert_eq!(advanced(start, 131_071), expected_state);
    let actual = block_on(generate_uniform_cooperative(
        expected.len(),
        "rng test",
        Some(&context),
        &mut serial_budget(&context),
    ))
    .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(state(&context), expected_state);
    let next = {
        let _scope = context.enter();
        generate_uniform(1, "rng test").unwrap()
    };
    assert_eq!(next, vec![next_uniform_state(&mut expected_state)]);
}

#[test]
fn zero_and_failed_allocation_do_not_consume_session_draws() {
    let context = context();
    let before = state(&context);
    assert!(block_on(generate_uniform_cooperative(
        0,
        "rng test",
        Some(&context),
        &mut serial_budget(&context),
    ))
    .unwrap()
    .is_empty());
    assert_eq!(state(&context), before);
    let error = block_on(generate_uniform_cooperative(
        usize::MAX,
        "rng test",
        Some(&context),
        &mut serial_budget(&context),
    ))
    .expect_err("the requested vector cannot be allocated");
    assert!(error.source.is_some());
    assert_eq!(state(&context), before);
}

#[test]
fn pre_cancelled_invocation_does_not_reserve_or_change_next_draw() {
    let source = CancellationSource::new();
    source.cancel(CancellationReason::User);
    let context = context();
    let before = state(&context);
    let mut budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        || Box::pin(async { Ok(()) }),
    );
    let error = block_on(generate_uniform_cooperative(
        64,
        "rng test",
        Some(&context),
        &mut budget,
    ))
    .expect_err("pre-cancelled generation");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(state(&context), before);
}

#[test]
fn unpolled_future_does_not_reserve() {
    let context = context();
    let before = state(&context);
    let mut budget = serial_budget(&context);
    let future = generate_uniform_cooperative(32, "rng test", Some(&context), &mut budget);
    drop(future);
    assert_eq!(state(&context), before);
}

struct PendingTurn {
    dropped: Rc<Cell<bool>>,
}

impl Future for PendingTurn {
    type Output = Result<(), ExecutionServiceError>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

impl Drop for PendingTurn {
    fn drop(&mut self) {
        self.dropped.set(true);
    }
}

#[test]
fn pending_drop_consumes_whole_reserved_range_without_holding_rng_borrow() {
    let context = context();
    let first = state(&context);
    let dropped_turn = Rc::new(Cell::new(false));
    let observe_drop = dropped_turn.clone();
    let source = CancellationSource::new();
    let mut budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            Box::pin(PendingTurn {
                dropped: observe_drop.clone(),
            })
        },
    );
    let mut future = Box::pin(generate_uniform_cooperative(
        17,
        "rng test",
        Some(&context),
        &mut budget,
    ));
    let waker = futures::task::noop_waker();
    let mut task = Context::from_waker(&waker);
    assert!(matches!(future.as_mut().poll(&mut task), Poll::Pending));
    assert_eq!(state(&context), advanced(first.clone(), 17));
    assert!(super::super::turn::try_acquire(Some(&context), "rng test").is_err());
    let expected_next = {
        let mut state = advanced(first.clone(), 17);
        state.prepare_mutation(None).unwrap();
        next_uniform_state(&mut state)
    };
    drop(future);
    let mut turn = super::super::turn::try_acquire(Some(&context), "rng test").unwrap();
    assert_eq!(
        turn.with_state("rng test", |rng| {
            rng.state.prepare_mutation(Some(&context)).unwrap();
            next_uniform_state(&mut rng.state)
        })
        .unwrap(),
        expected_next
    );
    drop(turn);
    assert!(dropped_turn.get());
    assert_eq!(state(&context), advanced(first, 18));
}

#[test]
fn post_reservation_cancel_and_service_failure_keep_typed_causes_and_range() {
    for cancel in [true, false] {
        let context = context();
        let first = state(&context);
        let source = Rc::new(CancellationSource::new());
        let turn_source = source.clone();
        let mut budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            move || {
                if cancel {
                    turn_source.cancel(CancellationReason::User);
                    Box::pin(async { Ok(()) })
                } else {
                    Box::pin(async {
                        Err(ExecutionServiceError::Infrastructure("turn lost".into()))
                    })
                }
            },
        );
        let error = block_on(generate_uniform_cooperative(
            9,
            "rng test",
            Some(&context),
            &mut budget,
        ))
        .expect_err("turn failure after reservation");
        assert_eq!(state(&context), advanced(first, 9));
        if cancel {
            assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        } else {
            assert!(error
                .source
                .as_deref()
                .is_some_and(|source| { source.is::<ExecutionServiceError>() }));
        }
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    }
}

#[test]
fn fallback_and_parallel_iteration_streams_use_same_reservation_authority() {
    let _guard = test_guard();
    super::super::reset_rng();
    let expected = super::super::expected_uniform_sequence(7);
    assert_eq!(generate_uniform(7, "rng test").unwrap(), expected);
    assert_eq!(
        snapshot().unwrap().state,
        advanced(RandomState::seeded(DEFAULT_USER_SEED), 7)
    );

    let parent = context();
    let streams = reserve_parallel_random_streams(&parent, 1).unwrap();
    let original_parent = state(&parent);
    let actual = {
        let _scope = enter_parallel_random_stream(&parent, streams[0].clone()).unwrap();
        let start = state(&parent);
        let draws = block_on(generate_uniform_cooperative(
            5,
            "rng test",
            Some(&parent),
            &mut serial_budget(&parent),
        ))
        .unwrap();
        let mut manual = start;
        manual.prepare_mutation(None).unwrap();
        assert_eq!(
            draws,
            (0..5)
                .map(|_| next_uniform_state(&mut manual))
                .collect::<Vec<_>>()
        );
        assert_eq!(state(&parent), manual);
        draws
    };
    assert_eq!(actual.len(), 5);
    assert_eq!(state(&parent), original_parent);
}
