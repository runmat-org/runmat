use super::*;
use crate::builtins::common::random::{
    enter_parallel_random_stream, next_uniform_state, reserve_parallel_random_streams, set_seed,
    test_guard, with_rng_state_in, RandomState,
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

fn budget(context: &RuntimeContext) -> CpuCooperationBudget {
    CpuCooperationBudget::new(context.execution().clone(), context.cancellation())
}

fn generate_normal(len: usize, label: &str) -> BuiltinResult<Vec<f64>> {
    let mut turn = super::super::turn::try_acquire(None, label)?;
    super::generate_normal(&mut turn, len, label)
}

async fn generate_normal_cooperative(
    len: usize,
    label: &str,
    context: Option<&RuntimeContext>,
    budget: &mut CpuCooperationBudget,
) -> BuiltinResult<Vec<f64>> {
    let mut turn = super::super::turn::acquire(context, label).await?;
    super::generate_normal_cooperative(&mut turn, len, label, context, budget).await
}

fn state(context: &RuntimeContext) -> RandomState {
    with_rng_state_in(Some(context), "rng test", |rng| rng.state.clone()).unwrap()
}

fn expected(len: usize, start: RandomState) -> (Vec<f64>, RandomState) {
    let mut local = start;
    local.prepare_mutation(None).unwrap();
    let mut values = Vec::with_capacity(len);
    while values.len() < len {
        push_pair(&mut values, len, &mut local);
    }
    (values, local)
}

#[test]
fn seeded_even_odd_zero_and_next_word_match_serial_transform() {
    for len in [0, 1, 2, 3, 17, 65_537] {
        let context = context();
        {
            let _scope = context.enter();
            set_seed(41).unwrap();
        }
        let before = state(&context);
        let (oracle, after) = expected(len, before.clone());
        let actual = block_on(generate_normal_cooperative(
            len,
            "rng test",
            Some(&context),
            &mut budget(&context),
        ))
        .unwrap();
        assert_eq!(actual, oracle);
        assert_eq!(state(&context), after);
        let mut next_state = after.clone();
        next_state.prepare_mutation(None).unwrap();
        let next = next_uniform_state(&mut next_state);
        assert_eq!(
            with_rng_state_in(Some(&context), "rng test", |rng| {
                rng.state.prepare_mutation(Some(&context)).unwrap();
                next_uniform_state(&mut rng.state)
            })
            .unwrap(),
            next
        );
    }
}

#[test]
fn sync_and_cooperative_normal_draws_share_sequence_and_state() {
    let _guard = test_guard();
    for len in [0, 1, 2, 3, 257] {
        super::super::reset_rng();
        let sync = generate_normal(len, "rng test").unwrap();
        let sync_state = super::super::snapshot().unwrap().state;
        super::super::reset_rng();
        let cooperative = block_on(generate_normal_cooperative(
            len,
            "rng test",
            None,
            &mut budget(&context()),
        ))
        .unwrap();
        assert_eq!(cooperative, sync);
        assert_eq!(super::super::snapshot().unwrap().state, sync_state);
    }
}

#[test]
fn exact_output_fill_keeps_mt_and_legacy_pair_consumption() {
    for legacy in [false, true] {
        let context = context();
        let _legacy = legacy.then(|| {
            enter_parallel_random_stream(
                &context,
                runmat_execution::ParallelRandomStream::RunMatLcgV1 { state: 41 },
            )
            .unwrap()
        });
        let before = state(&context);
        let (expected, following) = expected(5, before);
        let mut actual = vec![0.0; 5];
        let mut turn = block_on(super::super::turn::acquire(Some(&context), "rng test")).unwrap();
        block_on(super::fill_normal_cooperative(
            &mut turn,
            actual.len(),
            "rng test",
            &context,
            &mut budget(&context),
            |index, value| {
                actual[index] = value;
                Ok(())
            },
        ))
        .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(state(&context), following);
    }
}

#[test]
fn invalid_storage_pre_cancel_and_unpolled_future_consume_no_draws() {
    let context = context();
    let before = state(&context);
    assert!(block_on(generate_normal_cooperative(
        usize::MAX,
        "rng test",
        Some(&context),
        &mut budget(&context),
    ))
    .is_err());
    assert_eq!(state(&context), before);

    let cancellation = CancellationSource::new();
    cancellation.cancel(CancellationReason::User);
    let mut cancelled = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        cancellation.token(),
        || Box::pin(async { Ok(()) }),
    );
    let error = block_on(generate_normal_cooperative(
        5,
        "rng test",
        Some(&context),
        &mut cancelled,
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(state(&context), before);

    let mut pending = budget(&context);
    drop(generate_normal_cooperative(
        5,
        "rng test",
        Some(&context),
        &mut pending,
    ));
    assert_eq!(state(&context), before);
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
fn pending_drop_and_later_failures_preserve_the_consumed_prefix() {
    let pending_context = context();
    let start = state(&pending_context);
    let dropped = Rc::new(Cell::new(false));
    let dropped_in_turn = dropped.clone();
    let cancellation = CancellationSource::new();
    let mut pending_budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        cancellation.token(),
        move || Box::pin(PendingTurn(dropped_in_turn.clone())),
    );
    let mut future = Box::pin(generate_normal_cooperative(
        5,
        "rng test",
        Some(&pending_context),
        &mut pending_budget,
    ));
    let waker = futures::task::noop_waker();
    let mut task = Context::from_waker(&waker);
    assert!(matches!(future.as_mut().poll(&mut task), Poll::Pending));
    assert_eq!(state(&pending_context), expected(1, start).1);
    drop(future);
    assert!(dropped.get());

    for cancel in [true, false] {
        let context = context();
        let start = state(&context);
        let source = Rc::new(CancellationSource::new());
        let turn_source = source.clone();
        let mut failing_budget = CpuCooperationBudget::with_test_turn(
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
        let error = block_on(generate_normal_cooperative(
            7,
            "rng test",
            Some(&context),
            &mut failing_budget,
        ))
        .unwrap_err();
        assert_eq!(state(&context), expected(1, start).1);
        if cancel {
            assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        } else {
            assert!(error
                .source
                .as_deref()
                .is_some_and(|source| source.is::<ExecutionServiceError>()));
        }
    }
}

#[test]
fn iteration_owned_parallel_stream_preserves_parent_and_pair_state() {
    let parent = context();
    let streams = reserve_parallel_random_streams(&parent, 1).unwrap();
    let parent_state = state(&parent);
    {
        let _scope = enter_parallel_random_stream(&parent, streams[0].clone()).unwrap();
        let start = state(&parent);
        let (oracle, after) = expected(5, start);
        let actual = block_on(generate_normal_cooperative(
            5,
            "rng test",
            Some(&parent),
            &mut budget(&parent),
        ))
        .unwrap();
        assert_eq!(actual, oracle);
        assert_eq!(state(&parent), after);
    }
    assert_eq!(state(&parent), parent_state);
}
