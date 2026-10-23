use super::*;
use crate::execute::operation_completion::CompletedOperation;
use crate::execute::{call_suspension, operand};
use runmat_mir::{MirConstant, MirLocalId, MirOperand, MirRvalue, MirShortCircuitOp};

fn assert_nested_read(state: &mut HostState, expected: f64) {
    let local = operand::materialize_operand(state, &MirOperand::Local(MirLocalId(0))).unwrap();
    assert_eq!(local, Value::Num(expected));
    let nested = MirRvalue::ShortCircuit {
        left: MirOperand::Constant(MirConstant::Bool(true)),
        op: MirShortCircuitOp::And,
        right_temps: vec![],
        right: MirOperand::Local(MirLocalId(0)),
    };
    let request = runmat_runtime::native::NativeSiteRequest {
        function: 0,
        block: 0,
        position: 0,
        phase: runmat_runtime::native::NativeSitePhase::RVALUE,
        ordinal: 0,
        reserved: 0,
    };
    state.current_request = Some(request);
    for _ in 0..2 {
        state.begin_contextual_site(request).unwrap();
        assert!(matches!(
            operand::evaluate_rvalue(state, &nested, 1, None),
            Err(NativeExecutorError::CallSuspended)
        ));
        let pending = state.pending_call.as_ref().unwrap();
        let (continuation, generation) = (pending.continuation, pending.generation);
        assert_eq!(
            futures::executor::block_on(
                call_suspension::complete(state, continuation, generation,)
            )
            .unwrap(),
            request
        );
    }
    state.begin_contextual_site(request).unwrap();
    let output = operand::evaluate_rvalue(state, &nested, 1, None).unwrap();
    state.finish_contextual_site().unwrap();
    assert_eq!(
        state.arena.get(output[0]).unwrap(),
        &Value::Bool(expected != 0.0)
    );
}

#[test]
fn completed_operation_reconciles_success_and_caught_error_before_nested_reads() {
    runmat_gc::gc_test_context(|| {
        for fail in [false, true] {
            let mut state = omitted_input_state();
            let service = state.workspace.as_ref().unwrap().service();
            let call_runtime = state.runtime.clone();
            let selected_runtime = call_runtime.clone();
            let completed = futures::executor::block_on(selected_runtime.scope(async move {
                service.assign_with_context(&call_runtime, "x", Value::Num(7.0))?;
                if fail {
                    Err(runmat_runtime::runtime_error::semantic_error(
                        "ExpectedCompletedMutationFailure",
                        "partial write then error",
                    ))
                } else {
                    Ok(Value::Num(1.0))
                }
            }))
            .map_err(NativeExecutorError::from);
            let outcome = CompletedOperation::new(completed).finish(&mut state);
            if fail {
                assert!(
                    matches!(outcome, Err(NativeExecutorError::Runtime(error)) if error.identifier() == Some("RunMat:ExpectedCompletedMutationFailure"))
                );
            } else {
                assert_eq!(outcome.unwrap(), Value::Num(1.0));
            }
            // This is the same read used by subsequent embedded expressions or
            // a catch handler, not a workspace-service lookup masking stale locals.
            assert_nested_read(&mut state, 7.0);
        }
    });
}

#[test]
fn ready_call_completion_reconciles_partial_writes_for_both_outcomes() {
    runmat_gc::gc_test_context(|| {
        for fail in [false, true] {
            let mut state = omitted_input_state();
            let service = state.workspace.as_ref().unwrap().service();
            let call_runtime = state.runtime.clone();
            let request = runmat_runtime::native::NativeSiteRequest {
                function: 0,
                block: 0,
                position: 0,
                phase: runmat_runtime::native::NativeSitePhase::RVALUE,
                ordinal: 0,
                reserved: 0,
            };
            state.current_request = Some(request);
            let outcome =
                call_suspension::begin_rooted_arguments(&mut state, None, move || async move {
                    service.assign_with_context(&call_runtime, "x", Value::Num(9.0))?;
                    if fail {
                        Err(runmat_runtime::runtime_error::semantic_error(
                            "ExpectedReadyMutationFailure",
                            "partial write then error",
                        ))
                    } else {
                        runmat_runtime::execution::RootedValues::new(vec![Value::Bool(true)])
                            .map_err(|error| error.into_runtime_error())
                    }
                });
            assert!(matches!(outcome, Err(NativeExecutorError::CallSuspended)));
            missing_error(&mut state);
            let pending = state.pending_call.as_ref().unwrap();
            let (continuation, generation) = (pending.continuation, pending.generation);
            let outcome = futures::executor::block_on(call_suspension::complete(
                &mut state,
                continuation,
                generation,
            ));
            if fail {
                assert!(
                    matches!(outcome, Err(NativeExecutorError::Runtime(error)) if error.identifier() == Some("RunMat:ExpectedReadyMutationFailure"))
                );
            } else {
                assert_eq!(outcome.unwrap(), request);
                assert_eq!(
                    call_suspension::take_completed(&mut state, None).unwrap(),
                    Some(vec![Value::Bool(true)])
                );
            }
            assert_nested_read(&mut state, 9.0);
        }
    });
}

#[test]
fn pending_semantic_completion_reconciles_once_before_nested_reads() {
    runmat_gc::gc_test_context(|| {
        let mut state = omitted_input_state();
        let service = state.workspace.as_ref().unwrap().service();
        let call_runtime = state.runtime.clone();
        let request = runmat_runtime::native::NativeSiteRequest {
            function: 0,
            block: 0,
            position: 0,
            phase: runmat_runtime::native::NativeSitePhase::RVALUE,
            ordinal: 0,
            reserved: 0,
        };
        state.current_request = Some(request);
        let writes = Rc::new(std::cell::Cell::new(0));
        let observed_writes = Rc::clone(&writes);
        let outcome = call_suspension::begin_semantic_operation(
            &mut state,
            None,
            call_suspension::SemanticOperationKind::IndexedRead,
            Vec::new(),
            move || async move {
                observed_writes.set(observed_writes.get() + 1);
                service.assign_with_context(&call_runtime, "x", Value::Num(11.0))?;
                let mut first_poll = true;
                futures::future::poll_fn(move |cx| {
                    if std::mem::take(&mut first_poll) {
                        cx.waker().wake_by_ref();
                        std::task::Poll::Pending
                    } else {
                        std::task::Poll::Ready(())
                    }
                })
                .await;
                runmat_runtime::execution::RootedValues::new(vec![Value::Num(1.0)])
                    .map_err(|error| NativeExecutorError::from(error.into_runtime_error()))
            },
        );
        assert!(matches!(outcome, Err(NativeExecutorError::CallSuspended)));
        assert_eq!(
            writes.get(),
            0,
            "generated entry must not poll an operation"
        );
        let pending = state.pending_call.as_ref().unwrap();
        let (continuation, generation) = (pending.continuation, pending.generation);
        assert_eq!(
            futures::executor::block_on(call_suspension::complete(
                &mut state,
                continuation,
                generation,
            ))
            .unwrap(),
            request
        );
        let output = call_suspension::take_completed_semantic_operation(
            &mut state,
            None,
            call_suspension::SemanticOperationKind::IndexedRead,
        )
        .unwrap()
        .unwrap();
        output.with_owned_output_storage(|values| {
            assert_eq!(values.as_slice(), &[Value::Num(1.0)]);
        });
        assert_eq!(writes.get(), 1, "resumption must not replay the write");
        assert_nested_read(&mut state, 11.0);
    });
}
