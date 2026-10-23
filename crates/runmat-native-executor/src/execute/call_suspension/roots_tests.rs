use super::*;
use crate::execute::region::observation::test_support::observed_state;
use crate::region::placement_test_support::Probe;
use runmat_runtime::native::NativeSitePhase;
use runmat_value::HandleRef;
use std::cell::Cell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::Poll;

#[test]
fn managed_arguments_suspend_before_first_poll_and_survive_publication() {
    if runmat_gc::test_support::run_isolated_gc_test(concat!(
        module_path!(),
        "::managed_arguments_suspend_before_first_poll_and_survive_publication"
    )) {
        return;
    }
    runmat_gc::gc_test_context(|| {
        for pending in [false, true] {
            let mut state = observed_state(&Rc::new(Probe::default()));
            let request = NativeSiteRequest {
                function: 0,
                block: 0,
                position: 0,
                phase: NativeSitePhase::RVALUE,
                ordinal: 0,
                reserved: 0,
            };
            state.current_request = Some(request);

            let initial = runmat_gc::gc_allocate_rooted(Value::String("native result".into()))
                .expect("allocate managed result");
            let handle = initial.handle();
            let value = Value::HandleObject(HandleRef {
                class_name: "NativeCompletionHandle".into(),
                target: handle,
                valid: true,
            });
            let rooted = RootedValues::new(vec![value.clone()]).expect("root native result");
            initial
                .unroot()
                .expect("transfer allocation root to result");

            let observed = Arc::new(AtomicBool::new(false));
            let observed_in_callback = Arc::clone(&observed);
            let target = (handle.addr(), handle.epoch());
            let observer = runmat_gc::RootRetirementObserver::register(Arc::new(move |event| {
                let runmat_gc::RootRetirement::Explicit(handle) = event else {
                    return;
                };
                if (handle.addr(), handle.epoch()) != target
                    || observed_in_callback.swap(true, Ordering::SeqCst)
                {
                    return;
                }
                runmat_gc::gc_collect_major().expect("collect as source result root retires");
                assert_eq!(
                    runmat_gc::gc_clone_value(&handle)
                        .expect("native arena owns result before source root retires"),
                    Value::String("native result".into())
                );
            }))
            .expect("install retirement observer");

            let polls = Rc::new(Cell::new(0));
            let polls_in_future = Rc::clone(&polls);
            let outcome = begin_rooted_arguments(&mut state, None, move || async move {
                let mut first_poll = true;
                futures::future::poll_fn(move |cx| {
                    polls_in_future.set(polls_in_future.get() + 1);
                    if pending && first_poll {
                        first_poll = false;
                        cx.waker().wake_by_ref();
                        Poll::Pending
                    } else {
                        Poll::Ready(())
                    }
                })
                .await;
                Ok(rooted)
            });
            assert!(matches!(outcome, Err(NativeExecutorError::CallSuspended)));
            assert_eq!(polls.get(), 0, "native entry must not poll the callback");
            runmat_gc::gc_collect_major().expect("collect with pending result owner");
            assert_eq!(
                runmat_gc::gc_clone_value(&handle).expect("pending result remains live"),
                Value::String("native result".into())
            );
            let pending_call = state.pending_call.as_ref().expect("pending native call");
            let (continuation, generation) = (pending_call.continuation, pending_call.generation);
            assert_eq!(
                futures::executor::block_on(complete(&mut state, continuation, generation))
                    .expect("complete pending native call"),
                request
            );
            assert_eq!(polls.get(), if pending { 2 } else { 1 });
            let references = completed_roots(&state).collect::<Vec<_>>();
            assert_eq!(references.len(), 1);
            assert_eq!(state.arena.get(references[0]).unwrap(), &value);
            assert_eq!(
                take_completed(&mut state, None).unwrap(),
                Some(vec![value.clone()])
            );

            assert!(observed.load(Ordering::SeqCst));
            runmat_gc::gc_collect_major().expect("collect after native arena publication");
            assert_eq!(
                runmat_gc::gc_clone_value(&handle).expect("native arena retains result"),
                Value::String("native result".into())
            );
            drop(observer);
            drop(state);
            runmat_gc::gc_collect_major().expect("collect after native arena drop");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        }
    });
}

#[test]
fn sequence_and_preparation_futures_are_not_polled_under_native_entry() {
    runmat_gc::gc_test_context(|| {
        let request = NativeSiteRequest {
            function: 0,
            block: 0,
            position: 0,
            phase: NativeSitePhase::RVALUE,
            ordinal: 0,
            reserved: 0,
        };
        for kind in 0..3 {
            let mut state = observed_state(&Rc::new(Probe::default()));
            state.current_request = Some(request);
            let polls = Rc::new(Cell::new(0));
            let outcome = match kind {
                0 => {
                    let polls = Rc::clone(&polls);
                    begin_rooted_sequence(
                        &mut state,
                        None,
                        move || futures::future::poll_fn(move |_| {
                            polls.set(polls.get() + 1);
                            Poll::<Result<RootedValueSequence, runmat_runtime::RuntimeError>>::Pending
                        }),
                        1,
                    )
                    .map(|_| ())
                }
                1 => {
                    let polls = Rc::clone(&polls);
                    begin_prepared_receiver(&mut state, None, move || {
                        futures::future::poll_fn(move |_| {
                            polls.set(polls.get() + 1);
                            Poll::<
                                NativeExecutorResult<
                                    runmat_runtime::object::protocol::PreparedSubscriptReceiver,
                                >,
                            >::Pending
                        })
                    })
                    .map(|_| ())
                }
                _ => {
                    let polls = Rc::clone(&polls);
                    begin_sequence_stage(&mut state, None, move || {
                        futures::future::poll_fn(move |_| {
                            polls.set(polls.get() + 1);
                            Poll::<NativeExecutorResult<SequenceStage>>::Pending
                        })
                    })
                    .map(|_| ())
                }
            };
            assert!(matches!(outcome, Err(NativeExecutorError::CallSuspended)));
            assert_eq!(polls.get(), 0, "pending site polled inside generated entry");
            assert_eq!(state.pending_call.as_ref().unwrap().request, request);
            drop(state);
            assert_eq!(polls.get(), 0, "dropping an unpolled call must not poll it");
        }
    });
}

#[test]
fn single_arguments_use_the_synchronous_arena_path() {
    runmat_gc::gc_test_context(|| {
        let mut state = observed_state(&Rc::new(Probe::default()));
        let arguments = [
            runmat_mir::MirCallArg::Single(runmat_mir::MirOperand::Constant(
                runmat_mir::MirConstant::Number("7".into()),
            )),
            runmat_mir::MirCallArg::Single(runmat_mir::MirOperand::Constant(
                runmat_mir::MirConstant::Bool(true),
            )),
        ];
        assert_eq!(
            super::super::call::materialize_arguments(&mut state, &arguments).unwrap(),
            vec![Value::Num(7.0), Value::Bool(true)]
        );
        assert!(state.pending_call.is_none());
        assert_eq!(
            super::super::call::materialize_arguments(&mut state, &[]).unwrap(),
            Vec::<Value>::new()
        );
        assert!(state.pending_call.is_none());
    });
}

#[test]
fn unpolled_native_call_releases_its_captured_input_root_on_abandonment() {
    if runmat_gc::test_support::run_isolated_gc_test(concat!(
        module_path!(),
        "::unpolled_native_call_releases_its_captured_input_root_on_abandonment"
    )) {
        return;
    }
    runmat_gc::gc_test_context(|| {
        let mut state = observed_state(&Rc::new(Probe::default()));
        state.current_request = Some(NativeSiteRequest {
            function: 0,
            block: 0,
            position: 0,
            phase: NativeSitePhase::RVALUE,
            ordinal: 0,
            reserved: 0,
        });
        let allocation = runmat_gc::gc_allocate_rooted(Value::String("captured".into())).unwrap();
        let handle = allocation.handle();
        let argument = Value::HandleObject(HandleRef {
            class_name: "NativeInput".into(),
            target: handle,
            valid: true,
        });
        let captured = RootedValues::new(vec![argument]).unwrap();
        allocation.unroot().unwrap();
        let result = begin_rooted_arguments(&mut state, None, move || async move {
            let _captured = captured;
            futures::future::pending::<()>().await;
            unreachable!("abandoned call never resumes")
        });
        assert!(matches!(result, Err(NativeExecutorError::CallSuspended)));
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());
        drop(state);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}

#[test]
fn ready_sequence_and_error_complete_only_after_native_resume() {
    runmat_gc::gc_test_context(|| {
        for fail in [false, true] {
            let mut state = observed_state(&Rc::new(Probe::default()));
            let request = NativeSiteRequest {
                function: 0,
                block: 0,
                position: 0,
                phase: NativeSitePhase::RVALUE,
                ordinal: 0,
                reserved: 0,
            };
            state.current_request = Some(request);
            let polls = Rc::new(Cell::new(0));
            let polls_in_future = Rc::clone(&polls);
            let result = begin_rooted_sequence(
                &mut state,
                None,
                move || async move {
                    polls_in_future.set(polls_in_future.get() + 1);
                    if fail {
                        Err(runmat_runtime::runtime_error::semantic_error(
                            "NativeResumeFixture",
                            "ready failure",
                        ))
                    } else {
                        let sequence =
                            runmat_value::ValueSequence::single(Value::Num(9.0)).unwrap();
                        RootedValueSequence::new(sequence)
                            .map_err(|error| error.into_runtime_error())
                    }
                },
                1,
            );
            assert!(matches!(result, Err(NativeExecutorError::CallSuspended)));
            assert_eq!(polls.get(), 0);
            let pending = state.pending_call.as_ref().unwrap();
            let identity = (pending.continuation, pending.generation);
            let outcome = futures::executor::block_on(complete(&mut state, identity.0, identity.1));
            assert_eq!(polls.get(), 1, "ready callback may not replay");
            if fail {
                assert!(matches!(outcome, Err(NativeExecutorError::Runtime(error))
                    if error.identifier() == Some("RunMat:NativeResumeFixture")));
                assert!(state.completed_call.is_none());
            } else {
                assert_eq!(outcome.unwrap(), request);
                assert_eq!(
                    take_completed(&mut state, None).unwrap(),
                    Some(vec![Value::Num(9.0)])
                );
            }
        }
    });
}
