use super::super::CallableArtifactIndex;
use super::*;
use std::collections::HashMap;

fn evaluator() -> Rc<SessionSourceEvaluator> {
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    SessionSourceEvaluator::new(
        runmat_runtime::workspace::BaseWorkspaceOwner::new(),
        runtime.local_identity(),
    )
}

fn scalar_bundle() -> RuntimeCallableArtifactBundle {
    RuntimeCallableArtifactBundle::new(
        vec![ValuePayload::Inline(Box::new(
            runmat_execution::value::InlineValue::Logical(true),
        ))],
        Vec::new(),
        RuntimeCallableArtifactLimits::default(),
    )
    .expect("valid scalar bundle")
}

fn prepared(
    evaluator: &Rc<SessionSourceEvaluator>,
    expected: Rc<FunctionRegistryGeneration>,
) -> Box<dyn RuntimePreparedCallableAdmission> {
    prepared_with_registry(evaluator, expected, runmat_vm::FunctionRegistry::default())
}

fn prepared_with_registry(
    evaluator: &Rc<SessionSourceEvaluator>,
    expected: Rc<FunctionRegistryGeneration>,
    registry: runmat_vm::FunctionRegistry,
) -> Box<dyn RuntimePreparedCallableAdmission> {
    Box::new(PreparedCallableAdmission {
        evaluator: Rc::downgrade(evaluator),
        expected_epoch: expected.epoch(),
        expected,
        replacement: evaluator.prepare_import(
            registry,
            CallableArtifactIndex::default(),
            Vec::new(),
        ),
        values: runmat_runtime::execution::RootedValues::new(Vec::new())
            .expect("root empty values"),
    })
}

#[test]
fn dropping_prepared_admission_does_not_publish() {
    let evaluator = evaluator();
    let expected = evaluator.current_generation();
    drop(prepared(&evaluator, expected.clone()));
    assert!(Rc::ptr_eq(&evaluator.current_generation(), &expected));
}

#[test]
fn commit_is_exact_generation_checked_and_retains_replaced_owner() {
    let evaluator = evaluator();
    let expected = evaluator.current_generation();
    let original_registry = expected.registry().expect("original registry");
    let first = prepared(&evaluator, expected.clone());
    let stale = prepared(&evaluator, expected.clone());
    let committed = first.commit().expect("commit exact prepared generation");
    assert!(Rc::ptr_eq(&evaluator.current_generation(), &expected));
    assert_eq!(Rc::strong_count(&original_registry), 2);

    let error = match stale.commit() {
        Ok(_) => panic!("stale prepared generation must fail closed"),
        Err(error) => error,
    };
    assert_eq!(
        error.identifier(),
        Some("RunMat:CallableArtifact:DestinationChanged")
    );
    drop(committed);
    assert_eq!(Rc::strong_count(&original_registry), 1);
}

#[test]
fn commit_preserves_active_foreground_publication_identity() {
    let evaluator = evaluator();
    let existing_id = runmat_hir::FunctionId(3);
    let existing = runmat_vm::FunctionBytecode {
        function: existing_id,
        display_name: "foreground_definition".into(),
        ..Default::default()
    };
    let foreground_registry =
        runmat_vm::FunctionRegistry::new(HashMap::from([(existing_id, existing)]));
    let publication = evaluator
        .publish_for_execution(foreground_registry.clone(), None)
        .expect("publish active generation");
    let active = evaluator.current_generation();
    let imported_id = runmat_hir::FunctionId(8);
    let imported = runmat_vm::FunctionBytecode {
        function: imported_id,
        display_name: "retained_import".into(),
        binding_origin: Some(runmat_execution::FunctionBindingOrigin {
            program: runmat_execution::Digest::sha256("retained import"),
            function: runmat_types::ProgramFunctionId(8),
        }),
        ..Default::default()
    };
    let imported_owner = Arc::clone(&imported.definition_owner);
    let imported_lease = runmat_value::FunctionDefinitionOwner::issue(&imported_owner, 8);
    let mut imported_registry = foreground_registry.clone();
    imported_registry.insert_replacing_name(imported);
    imported_registry.remove(imported_id);
    assert!(imported_registry.resolve_name("retained_import").is_none());
    let committed = prepared_with_registry(&evaluator, active.clone(), imported_registry)
        .commit()
        .expect("commit import inside active foreground generation");
    assert!(Rc::ptr_eq(&evaluator.current_generation(), &active));

    publication.finish(foreground_registry);
    assert!(Rc::ptr_eq(&evaluator.current_generation(), &active));
    let finished = active
        .registry()
        .expect("finished generation remains current");
    assert_eq!(
        finished.resolve_name("foreground_definition"),
        Some(existing_id),
        "foreground definitions must still commit"
    );
    assert!(
        finished.resolve_name("retained_import").is_none(),
        "imported exact bodies must not publish ordinary names"
    );
    assert!(
        finished.get_owned(imported_id, &imported_lease).is_some(),
        "live imported exact body must survive foreground finish"
    );
    drop(committed);
}

#[test]
fn failed_foreground_restores_around_an_in_place_import() {
    let evaluator = evaluator();
    let restore = evaluator.current_generation();
    let publication = evaluator
        .publish_for_execution(runmat_vm::FunctionRegistry::default(), None)
        .expect("publish active generation");
    let active = evaluator.current_generation();
    let committed = prepared(&evaluator, active.clone())
        .commit()
        .expect("commit import inside active foreground generation");

    drop(publication);
    assert!(Rc::ptr_eq(&evaluator.current_generation(), &restore));
    assert!(
        active.registry().is_err(),
        "failed active generation retires"
    );
    drop(committed);
}

#[test]
fn commit_rejects_in_place_generation_publication() {
    let evaluator = evaluator();
    let publication = evaluator
        .publish_for_execution(runmat_vm::FunctionRegistry::default(), None)
        .expect("publish active generation");
    let expected = evaluator.current_generation();
    let staged = prepared(&evaluator, expected.clone());
    publication.finish(runmat_vm::FunctionRegistry::default());
    assert!(Rc::ptr_eq(&evaluator.current_generation(), &expected));

    let error = match staged.commit() {
        Ok(_) => panic!("in-place generation mutation must invalidate preparation"),
        Err(error) => error,
    };
    assert_eq!(
        error.identifier(),
        Some("RunMat:CallableArtifact:DestinationChanged")
    );
}

#[test]
fn unpolled_and_foreign_preparation_leave_destination_unchanged() {
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let evaluator = SessionSourceEvaluator::new(
        runmat_runtime::workspace::BaseWorkspaceOwner::new(),
        runtime.local_identity(),
    );
    let expected = evaluator.current_generation();
    let service = SessionCallableArtifactService {
        evaluator: Rc::downgrade(&evaluator),
    };

    let unpolled = service.prepare_admit(
        runtime.clone(),
        scalar_bundle(),
        RuntimeCallableArtifactLimits::default(),
    );
    drop(unpolled);
    assert!(Rc::ptr_eq(&evaluator.current_generation(), &expected));

    let foreign = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let error = match futures::executor::block_on(service.prepare_admit(
        foreign,
        scalar_bundle(),
        RuntimeCallableArtifactLimits::default(),
    )) {
        Ok(_) => panic!("foreign preparation must fail before publication"),
        Err(error) => error,
    };
    assert_eq!(
        error.identifier(),
        Some("RunMat:CallableArtifact:ForeignSession")
    );
    assert!(Rc::ptr_eq(&evaluator.current_generation(), &expected));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn escaped_handle_after_error_exports_into_fresh_session_without_publishing_name() {
    let mut source = crate::RunMatSession::with_options(false, false).expect("source session");
    let outcome = crate::execute_text_request_for_testing(
        &mut source,
        r#"
        escaped_artifact = @artifact_helper;
        error("RunMat:test:afterHandle", "after handle publication");
        function value = artifact_helper(input)
            value = input + 40;
        end
        "#,
    )
    .expect("runtime failure is returned in the execution outcome");
    assert_eq!(
        outcome.error.as_ref().and_then(|error| error.identifier()),
        Some("RunMat:test:afterHandle")
    );
    let escaped = source
        .get_variables()
        .expect("source workspace")
        .get("escaped_artifact")
        .cloned()
        .expect("handle assignment survives later error");
    let source_runtime = source.runtime_context.clone();
    let source_service = Rc::clone(
        source_runtime
            .service_ports()
            .require_callable_artifacts("export an escaped callback")
            .expect("source callable artifact service"),
    );
    let bundle = futures::executor::block_on(source_service.export(
        source_runtime,
        vec![escaped],
        RuntimeCallableArtifactLimits::default(),
    ))
    .expect("export escaped exact definition");
    drop(source);

    let destination = crate::RunMatSession::with_options(false, false).expect("fresh session");
    assert!(destination
        .function_registry
        .resolve_name("artifact_helper")
        .is_none());
    let destination_runtime = destination.runtime_context.clone();
    let destination_service = Rc::clone(
        destination_runtime
            .service_ports()
            .require_callable_artifacts("import an escaped callback")
            .expect("destination callable artifact service"),
    );
    let prepared = futures::executor::block_on(destination_service.prepare_admit(
        destination_runtime.clone(),
        bundle,
        RuntimeCallableArtifactLimits::default(),
    ))
    .expect("prepare cross-session callable import");
    assert!(destination
        .function_registry
        .resolve_name("artifact_helper")
        .is_none());
    let committed = prepared.commit().expect("commit callable import");
    assert!(destination
        .function_registry
        .resolve_name("artifact_helper")
        .is_none());

    let mut destination = destination;
    let ordinary =
        crate::execute_text_request_for_testing(&mut destination, "ordinary_after_import = 7;")
            .expect("ordinary request after import");
    assert!(ordinary.error.is_none(), "{:?}", ordinary.error);
    assert!(destination
        .function_registry
        .resolve_name("artifact_helper")
        .is_none());

    let Value::BoundFunctionHandle(handle) = committed.values()[0].clone() else {
        panic!("imported value must remain an exact bound handle");
    };
    let data = handle.into_data();
    let result = futures::executor::block_on(
        runmat_runtime::user_functions::try_call_bound_function_in_context(
            &destination_runtime,
            BoundFunctionBinding {
                function: data.function,
                authority: data.authority,
            },
            &[Value::Num(2.0)],
            1,
        ),
    )
    .expect("selected destination invoker")
    .expect("imported callback execution");
    assert_eq!(result.as_slice(), &[Value::Num(42.0)]);

    // The imported generation retains the admitted code and issued source
    // table while its handle lives, but must not retain the runtime service
    // port that points back to this evaluator.
    let evaluator = Rc::downgrade(&destination.source_evaluator);
    assert!(evaluator.upgrade().is_some());
    drop(result);
    drop(committed);
    drop(destination_service);
    drop(destination_runtime);
    drop(destination);
    assert!(evaluator.upgrade().is_none());
}
