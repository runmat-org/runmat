use super::*;
use crate::execute::{call_suspension, operand, sync};
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
    let output = operand::evaluate_rvalue(state, &nested, 1, None).unwrap();
    assert_eq!(
        state.arena.get(output[0]).unwrap(),
        &Value::Bool(expected != 0.0)
    );
}

#[test]
fn sync_completion_reconciles_success_and_caught_error_before_nested_reads() {
    runmat_gc::gc_test_context(|| {
        for fail in [false, true] {
            let mut state = omitted_input_state();
            let service = state.workspace.as_ref().unwrap().service();
            let outcome = sync::complete(
                &state.runtime,
                async move {
                    service.assign("x", Value::Num(7.0))?;
                    if fail {
                        Err(runmat_runtime::runtime_error::semantic_error(
                            "ExpectedSyncMutationFailure",
                            "partial write then error",
                        ))
                    } else {
                        Ok(Value::Num(1.0))
                    }
                },
                "sync mutation fixture",
            )
            .finish(&mut state);
            if fail {
                assert!(
                    matches!(outcome, Err(NativeExecutorError::Runtime(error)) if error.identifier() == Some("RunMat:ExpectedSyncMutationFailure"))
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
            let outcome = call_suspension::begin(
                &mut state,
                None,
                Box::pin(async move {
                    service.assign("x", Value::Num(9.0))?;
                    if fail {
                        Err(runmat_runtime::runtime_error::semantic_error(
                            "ExpectedReadyMutationFailure",
                            "partial write then error",
                        )
                        .into())
                    } else {
                        Ok(vec![Value::Bool(true)])
                    }
                }),
            );
            if fail {
                assert!(
                    matches!(outcome, Err(NativeExecutorError::Runtime(error)) if error.identifier() == Some("RunMat:ExpectedReadyMutationFailure"))
                );
            } else {
                assert_eq!(outcome.unwrap(), vec![Value::Bool(true)]);
            }
            assert_nested_read(&mut state, 9.0);
        }
    });
}

#[test]
fn unsupported_sync_suspension_still_reconciles_effects_before_error() {
    runmat_gc::gc_test_context(|| {
        let mut state = omitted_input_state();
        let service = state.workspace.as_ref().unwrap().service();
        let outcome = sync::complete(
            &state.runtime,
            async move {
                service.assign("x", Value::Num(11.0))?;
                futures::future::pending::<()>().await;
                Ok::<(), runmat_runtime::RuntimeError>(())
            },
            "unsupported sync suspension fixture",
        )
        .finish(&mut state);
        assert!(matches!(
            outcome,
            Err(NativeExecutorError::UnsupportedSite(_))
        ));
        assert_nested_read(&mut state, 11.0);
    });
}
