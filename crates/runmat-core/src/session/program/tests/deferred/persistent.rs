use super::*;
mod dependencies;
mod surrogate;

async fn run(
    session: &mut RunMatSession,
    source: &str,
) -> Result<runmat_runtime::execution::RootedResult<crate::abi::ExecutionOutcome>, crate::RunError>
{
    let request = crate::abi::ExecutionRequest::for_source(
        crate::abi::SourceInput::Text {
            name: "retained_original.m".into(),
            text: source.into(),
        },
        runmat_parser::CompatMode::RunMat,
        crate::abi::HostExecutionPolicy::default(),
        session.abi_workspace_handle,
    );
    session.execute_request(request).await.result
}

async fn capture(session: &mut RunMatSession, service: &CapturingService) -> CapturedCall {
    run(session, "f = retainedAsync(5);")
        .await
        .expect("original retained callable capture");
    let captured = service.take_original();
    assert!(
        captured.original.is_some(),
        "published capture must retain the exact original Core product for its standalone oracle"
    );
    captured
}

async fn cold_value(call: &CapturedCall) -> runmat_runtime::execution::RootedValueSequence {
    let envelope = call.envelope();
    assert_eq!(
        call.origin(),
        runmat_execution::FunctionBindingOrigin::from_revision(
            &envelope.manifest.identity.program,
            call.origin().function
        )
        .unwrap()
    );
    cold_envelope_value(
        &envelope,
        call.call.invocation.arguments().to_vec(),
        call.call.invocation.requested_outputs(),
    )
    .await
}

async fn cold_envelope_value(
    envelope: &runmat_execution::ExecutableUnitEnvelope,
    arguments: Vec<Value>,
    outputs: usize,
) -> runmat_runtime::execution::RootedValueSequence {
    let mut cold = RunMatSession::with_options(false, false).unwrap();
    assert!(
        cold.invoke_portable_unit(
            envelope,
            arguments.clone(),
            outputs,
            &crate::InvocationControl::default()
        )
        .await
        .is_err(),
        "foreign language environment must reject"
    );
    cold.set_compat_mode(runmat_parser::CompatMode::RunMat);
    cold.invoke_portable_unit(
        envelope,
        arguments,
        outputs,
        &crate::InvocationControl::default(),
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn prior_raw_function_capture_preserves_original_product_across_redefinition_and_failure() {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let service = Rc::new(CapturingService::new(&session));
    session.install_execution_services(service.clone());
    let original = "async function y = retainedAsync(x); y = x + 7; end";
    run(&mut session, original).await.unwrap();
    let first = capture(&mut session, &service).await;
    let envelope = first.envelope();
    assert_eq!(
        ExecutableUnit::from_portable_envelope(&envelope)
            .unwrap()
            .source()
            .text,
        original
    );
    assert_eq!(cold_value(&first).await.as_slice(), &[Value::Num(12.0)]);
    run(&mut session, "f = retainedAsync(5);\nfunction y = localOne(x); y = x + 1; end\nfunction y = localTwo(x); y = x + 2; end").await.unwrap();
    let colliding = service.take_original();
    assert_eq!(colliding.envelope(), first.envelope());
    assert_eq!(colliding.origin(), first.origin());

    let rejected = run(
        &mut session,
        "error('rejected declaration');\nasync function y = retainedAsync(x); y = x + 90; end",
    )
    .await
    .unwrap();
    assert!(
        rejected
            .diagnostics
            .iter()
            .any(
                |diagnostic| diagnostic.severity == crate::abi::DiagnosticSeverity::Error
                    && diagnostic.message.contains("rejected declaration")
            ),
        "raw runtime failures are reported as structured diagnostics"
    );
    let after_failure = capture(&mut session, &service).await;
    assert_eq!(after_failure.envelope(), first.envelope());
    assert_eq!(after_failure.origin(), first.origin());
    run(
        &mut session,
        "async function y = retainedAsync(x); y = x + 30; end",
    )
    .await
    .unwrap();
    let replacement = capture(&mut session, &service).await;
    assert_ne!(
        replacement.envelope().manifest.identity.program,
        first.envelope().manifest.identity.program
    );
    assert_ne!(replacement.origin(), first.origin());
    assert_eq!(
        cold_value(&replacement).await.as_slice(),
        &[Value::Num(35.0)]
    );
    assert_eq!(cold_value(&first).await.as_slice(), &[Value::Num(12.0)]);
    assert!(session.runtime_context.program_revision().is_none());
    assert!(session.program_service.active.borrow().is_empty());
}

#[tokio::test]
async fn retained_program_owner_validates_caller_and_preserves_original_cold_product() {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let original = "async function y = retainedAsync(x); y = x + 7; end";
    run(&mut session, original).await.unwrap();
    let function = session
        .function_registry
        .resolve_name("retainedAsync")
        .unwrap();
    let origin = session
        .function_registry
        .get(function)
        .unwrap()
        .binding_origin
        .unwrap();
    let caller = session
        .compile_executable_unit(
            ExecutableSource::new("separate-original-caller", "caller.m", "ans = 0;"),
            None,
        )
        .await
        .unwrap();
    let guard = session.program_service.enter(&caller);
    let envelope = session
        .program_service
        .freeze_executable(origin, &caller.revision().program_revision)
        .unwrap();
    assert_ne!(
        envelope.manifest.identity.program,
        caller.revision().program_revision
    );
    assert_eq!(
        ExecutableUnit::from_portable_envelope(&envelope)
            .unwrap()
            .source()
            .text,
        original
    );
    assert!(session
        .program_service
        .freeze_executable(origin, &envelope.manifest.identity.program)
        .is_err());
    let bound = session
        .program_service
        .freeze_bound_callable(origin, "retainedAsync")
        .unwrap();
    assert_eq!(bound.executable, envelope);
    drop(guard);
    assert!(session
        .program_service
        .freeze_executable(origin, &caller.revision().program_revision)
        .is_err());
    // The retained program API returns an authenticated original compiler
    // envelope, not a deferred graph issuer. Execute that original product
    // directly rather than constructing a synthetic DeferredCall authority.
    assert_eq!(
        cold_envelope_value(&envelope, vec![Value::Num(5.0)], 1)
            .await
            .as_slice(),
        &[Value::Num(12.0)]
    );
}

#[tokio::test]
async fn earlier_async_declaration_is_captured_by_interpreter_and_forced_native_caller() {
    for native in [false, true] {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let service = Rc::new(CapturingService::new(&session));
        session.install_execution_services(service.clone());
        run(
            &mut session,
            "async function y = retainedAsync(x); y = x + 7; end",
        )
        .await
        .unwrap();
        let original = capture(&mut session, &service).await;
        let caller = session
            .compile_executable_unit(
                ExecutableSource::new(
                    "later-deferred-caller",
                    "caller.m",
                    "function f = caller(x); f = retainedAsync(x); end",
                ),
                None,
            )
            .await
            .unwrap();
        let mut control = crate::InvocationControl::default();
        if native {
            control = control.force_generic_native();
        }
        let result = session
            .invoke_executable(
                &caller,
                crate::ProcedureInvocation {
                    target: crate::ProcedureTarget::Function("caller".into()),
                    arguments: crate::ProcedureArguments::Source(vec![Value::Num(5.0)]),
                    requested_outputs: 1,
                },
                &control,
            )
            .await
            .unwrap();
        assert!(matches!(result.as_slice(), [Value::Future(_)]));
        let captured = service.take_original();
        assert!(
            captured.original.is_some(),
            "the earlier published declaration must resolve its original Core owner"
        );
        assert_ne!(
            captured.envelope().manifest.identity.program,
            caller.revision().program_revision
        );
        assert_eq!(captured.envelope(), original.envelope());
        assert_eq!(captured.origin(), original.origin());
        assert_eq!(cold_value(&captured).await.as_slice(), &[Value::Num(12.0)]);
        assert!(session.runtime_context.program_revision().is_none());
        assert!(session.program_service.active.borrow().is_empty());
    }
}
