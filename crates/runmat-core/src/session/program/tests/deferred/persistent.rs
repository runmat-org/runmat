use super::*;
mod dependencies;

async fn run(
    session: &mut RunMatSession,
    source: &str,
) -> Result<crate::abi::ExecutionOutcome, crate::RunError> {
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

async fn capture(session: &mut RunMatSession, service: &CapturingService) -> DeferredCall {
    run(session, "f = retainedAsync(5);")
        .await
        .expect("original retained callable capture");
    service.original.borrow_mut().take().expect("original call")
}

async fn cold_value(call: &DeferredCall) -> Value {
    let envelope = runmat_execution::ExecutableUnitEnvelope::from_canonical_bytes(
        call.program.as_deref().unwrap(),
    )
    .unwrap();
    assert_eq!(
        call.program_revision.as_ref(),
        Some(&envelope.manifest.identity.program)
    );
    let mut cold = RunMatSession::with_options(false, false).unwrap();
    assert!(
        cold.invoke_portable_unit(
            &envelope,
            call.arguments.clone(),
            call.requested_outputs,
            &crate::InvocationControl::default()
        )
        .await
        .is_err(),
        "foreign language environment must reject"
    );
    cold.set_compat_mode(runmat_parser::CompatMode::RunMat);
    cold.invoke_portable_unit(
        &envelope,
        call.arguments.clone(),
        call.requested_outputs,
        &crate::InvocationControl::default(),
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn prior_raw_function_capture_preserves_original_product_across_redefinition_and_failure() {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let service = Rc::new(CapturingService::default());
    session.install_execution_services(service.clone());
    let original = "async function y = retainedAsync(x); y = x + 7; end";
    run(&mut session, original).await.unwrap();
    let first = capture(&mut session, &service).await;
    let envelope = runmat_execution::ExecutableUnitEnvelope::from_canonical_bytes(
        first.program.as_deref().unwrap(),
    )
    .unwrap();
    assert_eq!(
        ExecutableUnit::from_portable_envelope(&envelope)
            .unwrap()
            .source()
            .text,
        original
    );
    assert_eq!(cold_value(&first).await, Value::Num(12.0));
    run(&mut session, "f = retainedAsync(5);\nfunction y = localOne(x); y = x + 1; end\nfunction y = localTwo(x); y = x + 2; end").await.unwrap();
    let colliding = service
        .original
        .borrow_mut()
        .take()
        .expect("retained target survived unit-local ordinal collision");
    assert_eq!(colliding.program, first.program);
    assert_eq!(colliding.function, first.function);

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
    assert_eq!(after_failure.program, first.program);
    assert_eq!(after_failure.program_revision, first.program_revision);
    run(
        &mut session,
        "async function y = retainedAsync(x); y = x + 30; end",
    )
    .await
    .unwrap();
    let replacement = capture(&mut session, &service).await;
    assert_ne!(replacement.program_revision, first.program_revision);
    assert_ne!(replacement.function, first.function);
    assert_eq!(cold_value(&replacement).await, Value::Num(35.0));
    assert_eq!(cold_value(&first).await, Value::Num(12.0));
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
    let function = ProgramFunctionId(u32::try_from(function.0).unwrap());
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
        .freeze_executable(function, &caller.revision().program_revision)
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
        .freeze_executable(function, &envelope.manifest.identity.program)
        .is_err());
    let bound = session
        .program_service
        .freeze_bound_callable(function, "retainedAsync")
        .unwrap();
    assert_eq!(bound.executable, envelope);
    drop(guard);
    assert!(session
        .program_service
        .freeze_executable(function, &caller.revision().program_revision)
        .is_err());
    let call = DeferredCall {
        environment: runmat_execution::EnvironmentSnapshot::default(),
        function: function.0 as usize,
        arguments: vec![Value::Num(5.0)],
        requested_outputs: 1,
        program_revision: Some(envelope.manifest.identity.program.clone()),
        program: Some(envelope.canonical_bytes().unwrap()),
    };
    assert_eq!(cold_value(&call).await, Value::Num(12.0));
}

#[tokio::test]
async fn earlier_async_declaration_is_captured_by_interpreter_and_forced_native_caller() {
    for native in [false, true] {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let service = Rc::new(CapturingService::default());
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
                    arguments: vec![Value::Num(5.0)],
                    requested_outputs: 1,
                },
                &control,
            )
            .await
            .unwrap();
        assert!(matches!(result, Value::Future(_)));
        let captured = service
            .original
            .borrow_mut()
            .take()
            .expect("retained external future");
        assert_ne!(
            captured.program_revision.as_ref(),
            Some(&caller.revision().program_revision)
        );
        assert_eq!(captured.program, original.program);
        assert_eq!(captured.function, original.function);
        assert_eq!(cold_value(&captured).await, Value::Num(12.0));
        assert!(session.runtime_context.program_revision().is_none());
        assert!(session.program_service.active.borrow().is_empty());
    }
}
