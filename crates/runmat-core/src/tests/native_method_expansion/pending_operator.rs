//! A user overload must remain owned by its exact native operator site while pending.

use super::*;
use crate::{ExecutableSource, InvocationControl, ProcedureInvocation, ProcedureTarget};
use futures::channel::oneshot;
use futures::executor::block_on;
use futures::task::noop_waker_ref;
use std::collections::VecDeque;
use std::sync::Mutex;

const CLASS: &str = r#"
classdef PendingOperator
    properties
        x
    end
    methods
        function obj = PendingOperator(x)
            obj.x = x;
        end
        function result = plus(left, right)
            input('operator: ');
            result = left + right.x;
        end
    end
end
"#;

const SOURCE: &str = r#"
function value = pendingOperator()
    value = 2 + PendingOperator(5);
end
function value = catchPendingOperator()
    try
        2 + PendingOperator(5);
    catch
    end
    value = 3 + PendingOperator(5);
end
"#;

type InputTurn = (
    Option<oneshot::Receiver<()>>,
    Result<crate::execution::InputResponse, String>,
);

fn invoke(function: &str) -> ProcedureInvocation {
    ProcedureInvocation {
        target: ProcedureTarget::Function(function.into()),
        arguments: crate::ProcedureArguments::Source(Vec::new()),
        requested_outputs: 1,
    }
}

fn with_operator(test: impl FnOnce(&mut RunMatSession, &crate::ExecutableUnit, Arc<AtomicUsize>)) {
    runmat_gc::gc_test_context(|| {
        let _cwd_lock = super::super::cwd_lock();
        let directory = tempfile::TempDir::new().expect("operator class directory");
        std::fs::write(directory.path().join("PendingOperator.m"), CLASS)
            .expect("write operator class");
        // Companion class discovery is tied to a real primary source path.
        // The compiled in-memory source must also exist beside its class file.
        std::fs::write(directory.path().join("pending_operator_driver.m"), SOURCE)
            .expect("write operator source");
        assert_eq!(
            std::fs::read_to_string(directory.path().join("PendingOperator.m")).unwrap(),
            CLASS
        );
        assert_eq!(
            std::fs::read_to_string(directory.path().join("pending_operator_driver.m")).unwrap(),
            SOURCE
        );
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 2);
        let _cwd = super::super::push_cwd(directory.path());
        let mut session = RunMatSession::with_options(false, false).expect("operator session");
        let unit = block_on(
            session.compile_executable_unit(
                ExecutableSource::frozen(
                    "pending-operator@1",
                    "pending_operator_driver.m",
                    directory
                        .path()
                        .join("pending_operator_driver.m")
                        .to_string_lossy(),
                    SOURCE,
                ),
                None,
            ),
        )
        .expect("compile operator functions");
        assert!(
            unit.bytecode().instructions.is_empty(),
            "a function-only executable must not gain a script entrypoint"
        );
        let function = unit
            .functions()
            .names
            .get("pendingOperator")
            .and_then(|identity| unit.functions().functions.get(identity))
            .expect("compiled primary function");
        assert!(
            function.instructions.iter().any(|instruction| matches!(
                instruction,
                runmat_vm::Instr::RegisterClass { name, .. }
                    if name.display_name() == "PendingOperator"
            )),
            "compiled function must retain the companion class declaration"
        );
        let calls = Arc::new(AtomicUsize::new(0));
        test(&mut session, &unit, calls);
    });
}

fn install_input(session: &mut RunMatSession, calls: Arc<AtomicUsize>, turns: Vec<InputTurn>) {
    let turns = Arc::new(Mutex::new(VecDeque::from(turns)));
    session.install_async_input_handler(move |_| {
        calls.fetch_add(1, Ordering::SeqCst);
        let (gate, result) = turns
            .lock()
            .expect("operator input turns")
            .pop_front()
            .expect("unexpected operator replay");
        async move {
            if let Some(gate) = gate {
                gate.await
                    .map_err(|_| "operator gate abandoned".to_string())?;
            }
            result
        }
    });
}

fn pending_once(
    mut call: std::pin::Pin<
        &mut impl Future<
            Output = Result<runmat_runtime::execution::RootedValueSequence, crate::RunError>,
        >,
    >,
    calls: &AtomicUsize,
) {
    for _ in 0..8 {
        let mut task = Context::from_waker(noop_waker_ref());
        if let Poll::Ready(outcome) = call.as_mut().poll(&mut task) {
            panic!("operator completed before input gate: {outcome:?}");
        }
        if calls.load(Ordering::SeqCst) == 1 {
            return;
        }
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "operator input was not reached"
    );
}

#[test]
fn pending_native_rhs_operator_resumes_without_replay() {
    with_operator(|session, unit, calls| {
        let (release, gate) = oneshot::channel();
        install_input(
            session,
            calls.clone(),
            vec![(
                Some(gate),
                Ok(crate::execution::InputResponse::Line("1".into())),
            )],
        );
        let control = InvocationControl::default().force_generic_native();
        let mut pending =
            Box::pin(session.invoke_executable(unit, invoke("pendingOperator"), &control));
        pending_once(pending.as_mut(), &calls);
        release.send(()).expect("release operator input");
        let value = block_on(pending).expect("resumed overloaded operator");
        assert_eq!(value.as_slice(), &[Value::Num(7.0)]);
        assert_eq!(calls.load(Ordering::SeqCst), 1, "operator did not replay");
    });
}

#[test]
fn pending_native_rhs_operator_resumes_once_and_drop_releases_its_future() {
    with_operator(|session, unit, calls| {
        let (release, gate) = oneshot::channel();
        install_input(
            session,
            calls.clone(),
            vec![
                (
                    Some(gate),
                    Ok(crate::execution::InputResponse::Line("1".into())),
                ),
                (None, Ok(crate::execution::InputResponse::Line("1".into()))),
            ],
        );
        let control = InvocationControl::default().force_generic_native();
        let mut pending =
            Box::pin(session.invoke_executable(unit, invoke("pendingOperator"), &control));
        pending_once(pending.as_mut(), &calls);
        drop(pending);
        assert!(
            release.send(()).is_err(),
            "dropped operator retained input future"
        );
        let value = block_on(session.invoke_executable(unit, invoke("pendingOperator"), &control))
            .expect("fresh invocation after abandoned operator");
        assert_eq!(value.as_slice(), &[Value::Num(7.0)]);
        assert_eq!(calls.load(Ordering::SeqCst), 2, "each call executes once");
    });
}

#[test]
fn pending_native_rhs_operator_cancellation_retires_its_future() {
    with_operator(|session, unit, calls| {
        let (release, gate) = oneshot::channel();
        install_input(
            session,
            calls.clone(),
            vec![(
                Some(gate),
                Ok(crate::execution::InputResponse::Line("1".into())),
            )],
        );
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let control = InvocationControl::default()
            .force_generic_native()
            .with_cancellation(cancellation.token());
        let mut pending =
            Box::pin(session.invoke_executable(unit, invoke("pendingOperator"), &control));
        pending_once(pending.as_mut(), &calls);
        cancellation.cancel(runmat_execution::CancellationReason::User);
        let error = block_on(pending).expect_err("cancelled operator cannot publish a value");
        let crate::RunError::Runtime(error) = error else {
            panic!("expected typed operator cancellation: {error:?}");
        };
        assert_eq!(error.identifier(), Some("RunMat:ExecutionCancelled"));
        assert!(
            release.send(()).is_err(),
            "cancelled operator retained input future"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1, "operator did not replay");
    });
}

#[test]
fn caught_late_operator_error_retires_site_before_next_operator() {
    with_operator(|session, unit, calls| {
        let (release, gate) = oneshot::channel();
        install_input(
            session,
            calls.clone(),
            vec![
                (Some(gate), Err("late operator input failure".into())),
                (None, Ok(crate::execution::InputResponse::Line("1".into()))),
            ],
        );
        let control = InvocationControl::default().force_generic_native();
        let mut pending =
            Box::pin(session.invoke_executable(unit, invoke("catchPendingOperator"), &control));
        pending_once(pending.as_mut(), &calls);
        release.send(()).expect("release late operator error");
        let value = block_on(pending).expect("catch and fresh operator");
        assert_eq!(value.as_slice(), &[Value::Num(8.0)]);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "throwing site was not replayed"
        );
    });
}
