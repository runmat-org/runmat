use futures::executor::block_on;
use runmat_value::{HandleRef, Value};

use crate::{
    ExecutableSource, InvocationControl, ProcedureInvocation, ProcedureTarget, RunMatSession,
};

#[test]
fn public_invocation_roots_arguments_before_poll_and_retires_them_on_drop() {
    runmat_gc::gc_test_context(|| {
        for coverage in [false, true] {
            for execute in [false, true] {
                let mut session = RunMatSession::with_options(false, false).unwrap();
                let unit = block_on(session.compile_executable_unit(
                    ExecutableSource::new(
                        "invocation-roots@1",
                        "keepManaged.m",
                        "function out = keepManaged(value)\nout = value;\nend\n",
                    ),
                    None,
                ))
                .unwrap();
                let caller_root =
                    runmat_gc::gc_allocate_rooted(Value::String("payload".into())).unwrap();
                let handle = caller_root.handle();
                let invocation = ProcedureInvocation {
                    target: ProcedureTarget::Function("keepManaged".into()),
                    arguments: vec![Value::HandleObject(HandleRef {
                        class_name: "InvocationHandle".into(),
                        target: handle,
                        valid: true,
                    })],
                    requested_outputs: 1,
                };
                let control = InvocationControl::default();
                let output = if coverage {
                    let pending =
                        session.invoke_executable_with_coverage(&unit, invocation, &control);
                    caller_root.unroot().unwrap();
                    runmat_gc::gc_collect_major().unwrap();
                    assert!(runmat_gc::gc_clone_value(&handle).is_ok());
                    if execute {
                        Some(block_on(pending).unwrap().0)
                    } else {
                        drop(pending);
                        None
                    }
                } else {
                    let pending = session.invoke_executable(&unit, invocation, &control);
                    caller_root.unroot().unwrap();
                    runmat_gc::gc_collect_major().unwrap();
                    assert!(runmat_gc::gc_clone_value(&handle).is_ok());
                    if execute {
                        Some(block_on(pending).unwrap())
                    } else {
                        drop(pending);
                        None
                    }
                };
                if let Some(output) = output.as_ref() {
                    assert!(matches!(
                        output.as_slice(),
                        [Value::HandleObject(result)] if result.target == handle
                    ));
                    runmat_gc::gc_collect_major().unwrap();
                    assert!(runmat_gc::gc_clone_value(&handle).is_ok());
                }
                drop(output);
                runmat_gc::gc_collect_major().unwrap();
                assert!(runmat_gc::gc_clone_value(&handle).is_err());
            }
        }
    });
}

#[test]
fn rejected_invocations_check_control_before_registering_argument_roots() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new(
                "rejected-invocation@1",
                "keepManaged.m",
                "function out = keepManaged(value)\nout = value;\nend\n",
            ),
            None,
        ))
        .unwrap();
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        cancellation.cancel(runmat_execution::CancellationReason::User);
        for (control, identifier) in [
            (
                InvocationControl::default().with_cancellation(cancellation.token()),
                "RunMat:ExecutionCancelled",
            ),
            (
                InvocationControl::default().with_deadline_unix_ms(0),
                "RunMat:ExecutionDeadline",
            ),
        ] {
            for coverage in [false, true] {
                let caller_root =
                    runmat_gc::gc_allocate_rooted(Value::String("payload".into())).unwrap();
                let handle = caller_root.handle();
                let invocation = ProcedureInvocation {
                    target: ProcedureTarget::Function("keepManaged".into()),
                    arguments: vec![Value::HandleObject(HandleRef {
                        class_name: "InvocationHandle".into(),
                        target: handle,
                        valid: true,
                    })],
                    requested_outputs: 1,
                };
                let error = if coverage {
                    let pending =
                        session.invoke_executable_with_coverage(&unit, invocation, &control);
                    caller_root.unroot().unwrap();
                    runmat_gc::gc_collect_major().unwrap();
                    assert!(runmat_gc::gc_clone_value(&handle).is_err());
                    block_on(pending).unwrap_err()
                } else {
                    let pending = session.invoke_executable(&unit, invocation, &control);
                    caller_root.unroot().unwrap();
                    runmat_gc::gc_collect_major().unwrap();
                    assert!(runmat_gc::gc_clone_value(&handle).is_err());
                    block_on(pending).unwrap_err()
                };
                let crate::RunError::Runtime(error) = error else {
                    panic!("control rejection must retain the runtime error domain");
                };
                assert_eq!(error.identifier(), Some(identifier));
            }
        }
    });
}
