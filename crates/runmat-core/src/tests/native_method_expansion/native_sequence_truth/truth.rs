use super::*;
use crate::InvocationControl;
use futures::channel::oneshot;

#[test]
fn generated_short_circuit_preserves_rhs_effects_and_cancellation() {
    with_native_unit(|session, unit, calls| {
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
            Box::pin(session.invoke_executable(unit, invoke("shortCircuit", 1), &control));
        pending_once(pending.as_mut(), &calls);
        assert_eq!(
            input::observed_rhs_calls(),
            Some(1.0),
            "skipped RHS must not run and entered RHS must run once"
        );
        release.send(()).expect("release short-circuit RHS");
        let values = block_on(pending).expect("resume generated short-circuit");
        assert_eq!(values.as_slice(), &[Value::Bool(true)]);
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        let (release, gate) = oneshot::channel();
        install_input(
            session,
            calls.clone(),
            vec![(
                Some(gate),
                Ok(crate::execution::InputResponse::Line("1".into())),
            )],
        );
        calls.store(0, Ordering::SeqCst);
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let control = InvocationControl::default()
            .force_generic_native()
            .with_cancellation(cancellation.token());
        let mut pending =
            Box::pin(session.invoke_executable(unit, invoke("shortCircuit", 1), &control));
        pending_once(pending.as_mut(), &calls);
        cancellation.cancel(runmat_execution::CancellationReason::User);
        let error = block_on(pending).expect_err("cancelled RHS must not publish output");
        let crate::RunError::Runtime(error) = error else {
            panic!("expected typed cancellation: {error:?}");
        };
        assert_eq!(error.identifier(), Some("RunMat:ExecutionCancelled"));
        assert!(
            release.send(()).is_err(),
            "cancelled RHS retained its future"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1, "cancelled RHS replayed");

        install_input(
            session,
            calls.clone(),
            vec![(None, Err("native RHS service failure".into()))],
        );
        calls.store(0, Ordering::SeqCst);
        let control = InvocationControl::default().force_generic_native();
        let error = block_on(session.invoke_executable(unit, invoke("shortCircuit", 1), &control))
            .expect_err("failed RHS service cannot publish output");
        let crate::RunError::Runtime(error) = error else {
            panic!("expected runtime RHS failure: {error:?}");
        };
        assert!(error.to_string().contains("native RHS service failure"));
        assert_eq!(calls.load(Ordering::SeqCst), 1, "failed RHS replayed");
    });
}
