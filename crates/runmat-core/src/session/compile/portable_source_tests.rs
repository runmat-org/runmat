#[path = "portable_source_tests/graph_codec.rs"]
mod graph_codec;

#[test]
fn portable_compilation_retains_proven_runtime_contract_diagnostics() {
    let source = "A = rand(3,4); B = rand(3,4); C = A * B;";
    let mut session = crate::RunMatSession::with_options(false, false).expect("session");
    let unit = futures::executor::block_on(session.compile_executable_unit(
        crate::ExecutableSource::new("fixture", "main.m", source),
        None,
    ))
    .expect("a failing runtime operation is executable and may have a caller's handler");
    let error = unit
        .analysis()
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "RM-TYPE-MATMUL")
        .expect("static shape error must remain available to analysis consumers");
    assert_eq!(error.severity, runmat_hir::HirDiagnosticSeverity::Error);
    assert!(!error.blocks_execution());
    assert!(error.message.contains("inner dimensions 4 and 3"));
    let span = error.primary.span;
    assert!(source[span.start..span.end].contains("A * B"));
}

#[test]
fn portable_compilation_keeps_runtime_dependent_matrix_shape_admissible() {
    let mut session = crate::RunMatSession::with_options(false, false).expect("session");
    let unit = futures::executor::block_on(session.compile_executable_unit(
        crate::ExecutableSource::new(
            "fixture",
            "main.m",
            "function C = main(A, B)\nC = A * B;\nend\n",
        ),
        None,
    ))
    .expect("unknown runtime operand shapes are not a proven error");
    assert!(unit.portable_envelope().is_ok());

    let warning_only = futures::executor::block_on(session.compile_executable_unit(
        crate::ExecutableSource::new(
            "fixture",
            "warningOnly.m",
            "function y = warningOnly(x)\ny = definitely_missing(x);\nend\n",
        ),
        None,
    ));
    assert!(warning_only.is_ok(), "unresolved dynamic call is a warning");
}

#[test]
fn portable_source_ordinals_ignore_unrelated_session_source_allocations() {
    let source = crate::ExecutableSource::new(
        "fixture",
        "main.m",
        "function y = main(); y = helper(); end\nfunction y = helper(); y = 4; end\n",
    );
    let mut first = crate::RunMatSession::with_options(false, false).expect("first session");
    let first_unit =
        futures::executor::block_on(first.compile_executable_unit(source.clone(), None))
            .expect("first product");
    let first_bytes = first_unit
        .portable_envelope()
        .expect("first envelope")
        .canonical_bytes()
        .expect("first bytes");

    let mut second = crate::RunMatSession::with_options(false, false).expect("second session");
    second
        .compile_input_for_source_name("unrelated.m", "unrelated = 17;")
        .expect("allocate an unrelated runtime source identity");
    let second_unit = futures::executor::block_on(second.compile_executable_unit(source, None))
        .expect("second product");
    let second_bytes = second_unit
        .portable_envelope()
        .expect("second envelope")
        .canonical_bytes()
        .expect("second bytes");
    assert_eq!(first_bytes, second_bytes);
    assert_eq!(first_unit.mir(), second_unit.mir());
    assert_eq!(first_unit.analysis(), second_unit.analysis());
    assert_eq!(first_unit.source_map(), second_unit.source_map());
}

#[test]
fn portable_registry_paths_ignore_host_checkout_prefixes() {
    let mut session = crate::RunMatSession::with_options(false, false).expect("session");
    let unit = futures::executor::block_on(session.compile_executable_unit(
        crate::ExecutableSource::new("fixture", "main.m", "function y = main(); y = 1; end"),
        None,
    ))
    .expect("unit");
    let relocated = |prefix: &str| {
        let mut entries = unit.source_map().entries().to_vec();
        for entry in &mut entries {
            entry.full_path = Some(format!("{prefix}/{}", entry.relative_path));
        }
        crate::ExecutableUnit::new(
            unit.source().clone(),
            unit.revision().clone(),
            crate::ExecutableSourceMap::new(entries),
            unit.mir().clone(),
            unit.analysis().clone(),
            unit.bytecode().clone(),
        )
        .expect("relocated host unit")
        .portable_envelope()
        .expect("portable envelope")
        .canonical_bytes()
        .expect("portable bytes")
    };
    assert_eq!(relocated("/checkout/one"), relocated("/checkout/two"));
}

#[test]
fn changed_reachable_source_text_changes_portable_identity() {
    let mut session = crate::RunMatSession::with_options(false, false).expect("session");
    let compile = |session: &mut crate::RunMatSession, value| {
        let source = crate::ExecutableSource::new(
            "fixture",
            "main.m",
            format!(
                "function y = main(); y = helper(); end\nfunction y = helper(); y = {value}; end\n"
            ),
        );
        futures::executor::block_on(session.compile_executable_unit(source, None))
            .expect("portable product")
            .portable_envelope()
            .expect("portable envelope")
            .manifest
            .identity
    };
    assert_ne!(compile(&mut session, 1), compile(&mut session, 2));
}

#[test]
fn retained_definition_content_not_session_allocation_order_decides_portable_identity() {
    fn product(
        helper_value: usize,
        unrelated_first: bool,
    ) -> (Vec<u8>, crate::ExecutableSourceMap) {
        let mut session = crate::RunMatSession::with_options(false, false).expect("session");
        if unrelated_first {
            session
                .compile_input_for_source_name("unrelated.m", "unrelated = 17;")
                .expect("unrelated source identity");
        }
        let definition = crate::abi::ExecutionRequest::for_source(
            crate::abi::SourceInput::Text {
                name: "helper.m".into(),
                text: format!("function y = retainedHelper(); y = {helper_value}; end"),
            },
            session.compat_mode(),
            crate::abi::HostExecutionPolicy::default(),
            session.workspace_handle(),
        );
        let outcome = futures::executor::block_on(session.execute_request(definition))
            .result
            .expect("retained definition publication");
        assert!(outcome.diagnostics.is_empty(), "{outcome:?}");
        let helper = session
            .function_registry
            .resolve_name("retainedHelper")
            .and_then(|function| session.function_registry.get(function))
            .expect("published helper");
        let helper_origin = helper.binding_origin.expect("published helper origin");
        let helper_product = helper
            .selected_executable_product
            .as_ref()
            .expect("published helper product");
        let helper_content = helper_product.content_digest();
        let helper_body = helper_product
            .body_digest(helper_origin)
            .expect("published helper body");
        let unit = futures::executor::block_on(session.compile_executable_unit(
            crate::ExecutableSource::new(
                "fixture",
                "main.m",
                "function y = main(); y = retainedHelper(); end",
            ),
            None,
        ))
        .expect("unit with retained semantic dependency");
        assert!(unit.portable_envelope_for(Some("retainedHelper")).is_err());
        let envelope = unit.portable_envelope().expect("envelope");
        let bytes = envelope.canonical_bytes().expect("canonical bytes");
        let admitted = runmat_vm::admit_executable_unit(&bytes).expect("admitted envelope");
        assert!(admitted.static_imports().iter().any(|import| {
            import.target == helper_origin
                && import.target_content == helper_content
                && import.target_body == helper_body
                && import.kind
                    == runmat_execution::executable::ExecutableStaticImportKind::ExplicitExternal
        }));
        admitted
            .retained_callable_origin_fingerprints()
            .expect("exact multi-origin registry");
        (bytes, unit.source_map().clone())
    }

    let (first_bytes, first_sources) = product(7, false);
    let (shifted_bytes, shifted_sources) = product(7, true);
    assert_eq!(first_sources, shifted_sources);
    assert_eq!(first_bytes, shifted_bytes);
    let (changed_bytes, changed_sources) = product(8, true);
    assert_ne!(first_sources, changed_sources);
    assert_ne!(first_bytes, changed_bytes);
    assert!(
        first_sources
            .entries()
            .iter()
            .any(|entry| entry.relative_path == "helper.m" && entry.text.contains("y = 7")),
        "retained dependency belongs to the source map, not only the function registry"
    );
}

#[test]
fn definition_only_public_source_exports_exact_multi_function_artifact() {
    use runmat_value::Value;

    let mut source = crate::RunMatSession::with_options(false, false).expect("source session");
    let execute = |session: &mut crate::RunMatSession, name: &str, text: &str| {
        let request = crate::abi::ExecutionRequest::for_source(
            crate::abi::SourceInput::Text {
                name: name.into(),
                text: text.into(),
            },
            session.compat_mode(),
            crate::abi::HostExecutionPolicy::default(),
            session.workspace_handle(),
        );
        futures::executor::block_on(session.execute_request(request))
            .result
            .expect("public source execution")
    };
    let definition = execute(
        &mut source,
        "different_filename.m",
        "function y = retainedFirst(); y = 3; end\nfunction y = retainedSecond(); y = 7; end",
    );
    assert!(definition.diagnostics.is_empty());
    for name in ["retainedFirst", "retainedSecond"] {
        let id = source
            .function_registry
            .resolve_name(name)
            .expect("definition-only execution publishes each function");
        let body = source.function_registry.get(id).expect("published body");
        let owner = body
            .selected_source_owner
            .as_ref()
            .expect("published definition retains its admitted source table");
        assert!(owner.is_live());
        assert!(owner.owns_source(
            body.source_id
                .and_then(runmat_vm::BytecodeSourceRef::runtime)
                .expect("published body uses its durable source pool lease")
        ));
        assert_eq!(owner.durable_aliases().count(), 1);
    }
    let capture = execute(&mut source, "capture.m", "saved = @retainedSecond;");
    assert!(capture.diagnostics.is_empty());
    let saved = source
        .get_variables()
        .expect("source workspace")
        .get("saved")
        .cloned()
        .expect("retained handle");
    let source_runtime = source.runtime_context.clone();
    let service = std::rc::Rc::clone(
        source_runtime
            .service_ports()
            .require_callable_artifacts("export retained function")
            .expect("callable artifact service"),
    );
    let bundle = futures::executor::block_on(service.export(
        source_runtime,
        vec![saved],
        runmat_runtime::context::RuntimeCallableArtifactLimits::default(),
    ))
    .expect("export retained function and its definition unit");
    assert_eq!(bundle.unit_bytes().len(), 1);
    let later = execute(&mut source, "later_script.m", "unrelated = 11;");
    assert!(later.diagnostics.is_empty());
    let still_saved = source
        .get_variables()
        .expect("retained source workspace")
        .get("saved")
        .cloned()
        .expect("retained handle after an unrelated script");
    let later_runtime = source.runtime_context.clone();
    let later_bundle = futures::executor::block_on(service.export(
        later_runtime,
        vec![still_saved],
        runmat_runtime::context::RuntimeCallableArtifactLimits::default(),
    ))
    .expect("inherited callable retains its original unit");
    assert_eq!(bundle.unit_bytes(), later_bundle.unit_bytes());
    let admitted = runmat_vm::admit_executable_unit(&bundle.unit_bytes()[0])
        .expect("admit the selected function artifact");
    assert_eq!(
        admitted.envelope().manifest.identity.entrypoint,
        "retainedFirst"
    );
    assert!(
        admitted
            .registry()
            .functions
            .values()
            .any(|body| body.display_name == "retainedSecond"),
        "one envelope retains the companion definition"
    );

    let destination = crate::RunMatSession::with_options(false, false).expect("destination");
    let destination_runtime = destination.runtime_context.clone();
    let service = std::rc::Rc::clone(
        destination_runtime
            .service_ports()
            .require_callable_artifacts("import retained function")
            .expect("destination artifact service"),
    );
    let prepared = futures::executor::block_on(service.prepare_admit(
        destination_runtime.clone(),
        bundle,
        runmat_runtime::context::RuntimeCallableArtifactLimits::default(),
    ))
    .expect("prepare exact retained function import");
    let committed = prepared.commit().expect("commit retained function import");
    let Value::BoundFunctionHandle(handle) = committed.values()[0].clone() else {
        panic!("imported function retains a bound handle");
    };
    let data = handle.into_data();
    let result = futures::executor::block_on(
        runmat_runtime::user_functions::try_call_bound_function_in_context(
            &destination_runtime,
            runmat_value::BoundFunctionBinding {
                function: data.function,
                authority: data.authority,
            },
            &[],
            1,
        ),
    )
    .expect("selected destination invoker")
    .expect("imported companion executes");
    assert_eq!(result.as_slice(), &[Value::Num(7.0)]);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn pending_source_lease_coexists_with_interpreted_and_native_unit_invocations() {
    let mut session = crate::RunMatSession::with_options(false, false).expect("session");
    let held = runmat_runtime::source_context::lease_source_context_in_context(
        &session.runtime_context,
        "<eval>",
        None,
        "pending = 99;",
    )
    .expect("session source lease");
    let unit = futures::executor::block_on(session.compile_executable_unit(
        crate::ExecutableSource::new("fixture", "answer.m", "function y = answer(); y = 7; end\n"),
        None,
    ))
    .expect("unit");
    let invocation = crate::ProcedureInvocation {
        target: crate::ProcedureTarget::Function("answer".into()),
        arguments: crate::ProcedureArguments::Source(Vec::new()),
        requested_outputs: 1,
    };
    let expected =
        runmat_value::ValueSequence::single(runmat_value::Value::Num(7.0)).expect("one output");
    let established = futures::executor::block_on(session.invoke_executable(
        &unit,
        invocation.clone(),
        &crate::InvocationControl::default(),
    ))
    .expect("interpreted or tiered execution");
    assert_eq!(established, expected);
    let native = futures::executor::block_on(session.invoke_executable(
        &unit,
        invocation,
        &crate::InvocationControl::default().force_generic_native(),
    ))
    .expect("native execution");
    assert_eq!(native, expected);
    assert_eq!(
        runmat_runtime::source_context::source_info_in_context(
            &session.runtime_context,
            held.source_id(),
        )
        .expect("pending eval source remains live")
        .text
        .as_ref(),
        "pending = 99;"
    );
    let held_id = held.source_id();
    drop(held);
    assert!(runmat_runtime::source_context::source_info_in_context(
        &session.runtime_context,
        held_id,
    )
    .is_none());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn dropped_pending_public_eval_retires_only_its_source() {
    use futures::channel::oneshot;
    use futures::future::Either;
    use std::sync::{Arc, Mutex};

    let mut session = crate::RunMatSession::with_options(false, false).expect("session");
    session.top_level_await_enabled = false;
    session.dynamic_eval_enabled = false;
    let previous_source_name = session.active_source_name.clone();
    let previous_source_path = session.active_source_fullpath_name.clone();
    let previous_source_identity = session.active_source_identity.clone();
    let previous_primary = session.active_primary_source.clone();
    let previous_workspace = session.workspace_handle();
    let previous_policy = (
        session.compat_mode,
        session.top_level_await_enabled,
        session.dynamic_eval_enabled,
    );
    let runtime = session.runtime_context.clone();
    let sibling = runmat_runtime::source_context::lease_source_context_in_context(
        &runtime,
        "<sibling>",
        None,
        "sibling = 7;",
    )
    .expect("sibling source");
    let (reached_handler, handler_reached) = oneshot::channel();
    let reached_handler = Arc::new(Mutex::new(Some(reached_handler)));
    let (release, gate) = oneshot::channel::<()>();
    let gate = Arc::new(Mutex::new(Some(gate)));
    session.install_async_input_handler(move |_| {
        let source = runmat_runtime::source_context::current_source_info()
            .expect("input handler runs within the eval source scope");
        reached_handler
            .lock()
            .expect("handler signal")
            .take()
            .expect("no input replay")
            .send(source)
            .expect("public request retains its handler observer");
        let gate = gate
            .lock()
            .expect("input gate")
            .take()
            .expect("no input replay");
        async move {
            gate.await.map_err(|_| "input abandoned".to_string())?;
            Ok(crate::execution::InputResponse::Line("1".into()))
        }
    });
    let request = crate::abi::ExecutionRequest::for_source(
        crate::abi::SourceInput::Text {
            name: "pending_eval.m".into(),
            text: "result = eval(\"input('wait: '); 41 + 1\");".into(),
        },
        session.compat_mode(),
        crate::abi::HostExecutionPolicy::default(),
        session.workspace_handle(),
    );
    let mut future = Box::pin(session.execute_request(request));
    let eval_source = futures::executor::block_on(async {
        match futures::future::select(future.as_mut(), handler_reached).await {
            Either::Right((Ok(source), _pending_request)) => source,
            Either::Right((Err(_), _pending_request)) => {
                panic!("input handler observer was dropped before admission")
            }
            Either::Left((outcome, _pending_observer)) => {
                panic!("public eval terminated before reaching pending input: {outcome:?}")
            }
        }
    });
    assert_eq!(eval_source.name.as_ref(), "<eval>");
    let eval_id = eval_source.source_id.expect("eval source identity");
    assert_eq!(eval_source.text.as_ref(), "input('wait: '); 41 + 1");
    let pending_eval = runmat_runtime::source_context::source_info_in_context(&runtime, eval_id)
        .expect("pending eval source still live");
    assert_eq!(pending_eval.name.as_ref(), "<eval>");
    assert_eq!(pending_eval.text.as_ref(), eval_source.text.as_ref());
    assert_eq!(pending_eval.lexical_file_name(), "pending_eval.m");
    drop(future);
    assert_eq!(session.active_source_name, previous_source_name);
    assert_eq!(session.active_source_fullpath_name, previous_source_path);
    assert_eq!(session.active_source_identity, previous_source_identity);
    assert_eq!(session.active_primary_source, previous_primary);
    assert_eq!(session.workspace_handle(), previous_workspace);
    assert_eq!(
        (
            session.compat_mode,
            session.top_level_await_enabled,
            session.dynamic_eval_enabled,
        ),
        previous_policy
    );
    assert!(session.pending_companion_source_discovery.is_none());
    assert!(runmat_runtime::source_context::source_info_in_context(&runtime, eval_id).is_none());
    assert_eq!(
        runmat_runtime::source_context::source_info_in_context(&runtime, sibling.source_id())
            .expect("unrelated source remains live")
            .text
            .as_ref(),
        "sibling = 7;"
    );
    assert!(
        release.send(()).is_err(),
        "dropped input future released gate"
    );
}

#[test]
fn retained_function_handle_error_keeps_original_source_after_later_public_calls() {
    let mut session = crate::RunMatSession::with_options(false, false).expect("session");
    let mut execute = |name: &str, text: &str| {
        let request = crate::abi::ExecutionRequest::for_source(
            crate::abi::SourceInput::Text {
                name: name.into(),
                text: text.into(),
            },
            session.compat_mode(),
            crate::abi::HostExecutionPolicy::default(),
            session.workspace_handle(),
        );
        futures::executor::block_on(session.execute_request(request))
            .result
            .expect("public source execution")
    };
    let definition = execute(
        "original_retained.m",
        "function y = retainedBoom()\nerror(\"RunMat:retained:boom\", \"retained body failed\");\ny = 1;\nend",
    );
    assert!(definition.diagnostics.is_empty());
    let capture = execute("capture_handle.m", "saved = @retainedBoom;");
    assert!(capture.diagnostics.is_empty());
    let outcome = execute("later_call.m", "saved();");
    let error = outcome
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "RunMat:retained:boom")
        .expect("retained body error remains visible");
    assert!(
        error
            .callstack
            .iter()
            .any(|frame| frame.contains("original_retained.m")),
        "retained call must cite the definition source, not the later call: {error:?}"
    );
    assert!(
        error
            .callstack
            .iter()
            .all(|frame| !frame.contains("capture_handle.m")),
        "retained body must not be attributed to handle construction: {error:?}"
    );
}

#[test]
fn imported_unit_worker_projects_portable_source_before_builtin_dispatch() {
    use runmat_execution::{OutputContract, ProgramCallable, ProgramInvocationContext};
    use runmat_execution_artifact::{
        ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionRequest,
        ProgramExecutionResponse, ProgramTarget, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
    };
    use std::rc::Rc;

    let mut session = crate::RunMatSession::with_options(false, false).expect("compiler session");
    let unit = futures::executor::block_on(session.compile_executable_unit(
        crate::ExecutableSource::new("fixture", "worker_script.m", "ans = mfilename();"),
        None,
    ))
    .expect("portable unit");
    let envelope = unit.portable_envelope().expect("portable envelope");
    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: envelope.manifest.identity.program.clone(),
        entrypoint: envelope.manifest.identity.entrypoint.clone(),
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        execution_mode: "interpreter".into(),
        target: ProgramTarget::portable("source-projection-worker"),
        interop: envelope.manifest.interop.clone(),
        foreign_policy: None,
        accelerators: Vec::new(),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    };
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::ExecutableUnitV4,
        envelope.canonical_bytes().expect("canonical unit"),
    )
    .expect("artifact");
    let request = ProgramExecutionRequest {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe,
        artifact,
        callable: ProgramCallable::semantic(
            envelope.manifest.identity.entrypoint_function.unwrap(),
            None,
        ),
        context: ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
        arguments: Vec::new(),
        requested_outputs: 1,
    };
    let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    for _ in 0..2 {
        let response = futures::executor::block_on(
            runmat_vm::execute_program_request_with_context(request.clone(), runtime.clone()),
        );
        let ProgramExecutionResponse::Success { outputs } = response else {
            panic!("imported worker unit must use its validated source map: {response:?}");
        };
        assert_eq!(outputs.len(), 1);
        assert_eq!(
            runmat_runtime::execution::value_codec::decode_inline_value(&outputs[0])
                .expect("inline worker value"),
            runmat_value::Value::String("worker_script".into())
        );
    }
    let mut future_schema = envelope.clone();
    future_schema.manifest.revisions.source_map_schema = 3;
    let descriptor = future_schema
        .manifest
        .components
        .iter_mut()
        .find(|component| component.kind == runmat_execution::ExecutableComponentKind::SourceMap)
        .expect("source-map descriptor");
    descriptor.schema_version = 3;
    let error = ProgramArtifact::materialize(
        &request.recipe,
        ExecutableForm::ExecutableUnitV4,
        future_schema
            .canonical_bytes()
            .expect("forward schema envelope"),
    )
    .expect_err("future source-map schema is rejected at artifact admission");
    assert!(
        matches!(
            error,
            runmat_execution_artifact::ArtifactError::Invalid(ref message)
                if message == "unsupported executable source-map schema 3"
        ),
        "unexpected artifact admission error: {error:?}"
    );
}
