use super::*;

#[test]
fn eval_selected_handle_retains_its_exact_body_without_named_publication() {
    let mut session = RunMatSession::with_options(false, false).expect("session");
    let result = crate::execute_text_request_for_testing(
        &mut session,
        "eval('captured = 40; h = @() captured + 1;'); first = feval(h); clear captured; second = feval(h);",
    )
    .expect("eval source executes");
    assert!(result.error.is_none(), "{result:?}");
    let vars = session.get_variables().expect("workspace");
    assert_eq!(vars.get("first"), Some(&Value::Num(41.0)));
    assert_eq!(vars.get("second"), Some(&Value::Num(41.0)));
    assert!(matches!(vars.get("h"), Some(Value::Closure(_))));
}

#[test]
fn eval_escaped_external_callback_retains_its_selected_target_after_republication_and_clear() {
    let _serial = runmat_filesystem::provider_override_lock();
    let temporary = tempfile::TempDir::new().expect("selected eval source root");
    let helper = temporary.path().join("selected_eval_helper.m");
    std::fs::write(
        &helper,
        "function y = selected_eval_helper()\n y = 41;\nend\n",
    )
    .expect("old selected target source");
    let root = temporary.path().to_string_lossy().replace('\'', "''");
    let mut session = RunMatSession::with_options(false, false).expect("session");
    let first = crate::execute_text_request_for_testing(
        &mut session,
        &format!(
            "addpath('{root}'); warm = selected_eval_helper(); eval('old_callback = @() selected_eval_helper();'); first = feval(old_callback);"
        ),
    )
    .expect("first selected eval executes");
    assert!(first.error.is_none(), "{first:?}");
    let values = session.get_variables().expect("first workspace");
    assert_eq!(values.get("warm"), Some(&Value::Num(41.0)));
    assert_eq!(values.get("first"), Some(&Value::Num(41.0)));
    assert!(matches!(
        values.get("old_callback"),
        Some(Value::Closure(_))
    ));

    std::fs::write(
        &helper,
        "function y = selected_eval_helper()\n y = 43;\nend\n",
    )
    .expect("revised selected target source");
    let second = crate::execute_text_request_for_testing(
        &mut session,
        "clear selected_eval_helper; fresh = selected_eval_helper(); eval('new_callback = @() selected_eval_helper();'); old_after = feval(old_callback); new_after = feval(new_callback); clear selected_eval_helper; old_after_clear = feval(old_callback); new_after_clear = feval(new_callback);",
    )
    .expect("revised selected eval executes");
    assert!(second.error.is_none(), "{second:?}");
    let values = session
        .get_variables()
        .expect("retained selected workspace");
    for (name, expected) in [
        ("fresh", 43.0),
        ("old_after", 41.0),
        ("new_after", 43.0),
        ("old_after_clear", 41.0),
        ("new_after_clear", 43.0),
    ] {
        assert_eq!(values.get(name), Some(&Value::Num(expected)), "{name}");
    }

    // This late function-only product omits the eval-local sibling's body.
    // Its typed Local call must retain that exact executing definition while
    // the separate unresolved call keeps the selected external revision.
    let mixed = crate::execute_text_request_for_testing(
        &mut session,
        "eval('mixed_callback = @(x) eval_local_sibling(x) + selected_eval_helper(); function y = eval_local_sibling(x); y = x + 1; end'); mixed_first = feval(mixed_callback, 1); clear eval_local_sibling; clear selected_eval_helper; mixed_after = feval(mixed_callback, 1);",
    )
    .expect("mixed local and selected callback executes");
    assert!(mixed.error.is_none(), "{mixed:?}");
    let values = session.get_variables().expect("mixed retained workspace");
    assert_eq!(values.get("mixed_first"), Some(&Value::Num(45.0)));
    assert_eq!(values.get("mixed_after"), Some(&Value::Num(45.0)));

    // The next source is absent from the path at eval entry. Only reaching
    // this closure's creation site may select it, after addpath has executed.
    // The malformed source in an unreachable branch must never be loaded;
    // the parameter named like an eligible source remains a local binding.
    let later = tempfile::TempDir::new().expect("late selected source root");
    let later_helper = later.path().join("issuance_helper.m");
    std::fs::write(
        &later_helper,
        "function y = issuance_helper()\n y = 17;\nend\n",
    )
    .expect("late selected target source");
    std::fs::write(later.path().join("unreachable_helper.m"), "function y = (")
        .expect("unreachable malformed source");
    let later_root = later.path().to_string_lossy().replace('\'', "''");
    let issue = crate::execute_text_request_for_testing(
        &mut session,
        &format!(
            "eval('addpath(''{later_root}''); late = @() issuance_helper(); shadow = @(issuance_helper) issuance_helper + 1; if false; never = @() unreachable_helper(); end;'); shadow_value = feval(shadow, 2);"
        ),
    )
    .expect("late closure source executes");
    assert!(issue.error.is_none(), "{issue:?}");
    std::fs::write(
        &later_helper,
        "function y = issuance_helper()\n y = 19;\nend\n",
    )
    .expect("revised late target source");
    let invoke = crate::execute_text_request_for_testing(
        &mut session,
        "clear issuance_helper; latest = issuance_helper(); late_after = feval(late); shadow_after = feval(shadow, 2);",
    )
    .expect("late selected closure remains executable");
    assert!(invoke.error.is_none(), "{invoke:?}");
    let values = session.get_variables().expect("late retained workspace");
    for (name, expected) in [
        ("shadow_value", 3.0),
        ("latest", 19.0),
        ("late_after", 17.0),
        ("shadow_after", 3.0),
    ] {
        assert_eq!(values.get(name), Some(&Value::Num(expected)), "{name}");
    }

    // The first nested eval assigns this name after the outer eval was
    // compiled but before its closure is created. The handle must capture
    // the assigned function value now, not read the later workspace value.
    let shadowed = crate::execute_text_request_for_testing(
        &mut session,
        "eval('eval(''shadowed_source = @() 29; shadowed_array = [29, 30];''); pinned = @() shadowed_source(); pinned_complex = @() 1 + shadowed_source(); pinned_array = @() shadowed_array(2);'); shadowed_source = @() 31; shadowed_array = [31, 32]; pinned_after = feval(pinned); pinned_complex_after = feval(pinned_complex); pinned_array_after = feval(pinned_array);",
    )
    .expect("assigned-source shadow is issued at closure creation");
    assert!(shadowed.error.is_none(), "{shadowed:?}");
    let values = session
        .get_variables()
        .expect("shadowed retained workspace");
    assert_eq!(values.get("pinned_after"), Some(&Value::Num(29.0)));
    assert_eq!(values.get("pinned_complex_after"), Some(&Value::Num(30.0)));
    assert_eq!(values.get("pinned_array_after"), Some(&Value::Num(30.0)));
    assert!(matches!(values.get("pinned"), Some(Value::Closure(_))));
}

#[path = "dynamic_source_product_tests/super_constructor.rs"]
mod super_constructor;

#[test]
fn dynamic_cache_reader_snapshot_does_not_hold_mutation_access() {
    let _serial = runmat_filesystem::provider_override_lock();
    let temporary = tempfile::TempDir::new().expect("source root");
    std::fs::write(
        temporary.path().join("snapshot_helper.m"),
        "function y = snapshot_helper()\n y = 17;\nend\n",
    )
    .expect("helper source");
    let mut session = RunMatSession::with_options(false, false).expect("session");
    let root = temporary.path().to_string_lossy().replace('\'', "''");
    let first = crate::execute_text_request_for_testing(
        &mut session,
        &format!("addpath('{root}'); first = snapshot_helper();"),
    )
    .expect("dynamic call");
    assert!(first.error.is_none(), "{first:?}");
    let reader = session.dynamic_function_cache.snapshot();
    assert_eq!(reader.len(), 1);

    session
        .dynamic_function_cache
        .clear_named("snapshot_helper");
    assert_eq!(session.cached_source_count(), 0);
    assert_eq!(reader.len(), 1, "a reader retains its exact owned product");
    let second =
        crate::execute_text_request_for_testing(&mut session, "second = snapshot_helper();")
            .expect("rebuild with old reader retained");
    assert!(second.error.is_none(), "{second:?}");
    assert_eq!(
        session.get_variables().unwrap().get("second"),
        Some(&Value::Num(17.0))
    );
    assert_eq!(session.cached_source_count(), 1);
    session.dynamic_function_cache.clear();
    assert_eq!(session.cached_source_count(), 0);
    assert_eq!(reader.len(), 1);
}

fn execute_helper(session: &mut RunMatSession) -> crate::abi::RuntimeDiagnostic {
    let request = crate::abi::ExecutionRequest::for_source(
        crate::abi::SourceInput::Text {
            name: "<dynamic-source-product>".into(),
            text: "helper();".into(),
        },
        session.compat_mode(),
        crate::abi::HostExecutionPolicy::default(),
        session.workspace_handle(),
    );
    let outcome = futures::executor::block_on(session.execute_request(request))
        .result
        .expect("public source invocation returns an outcome");
    outcome
        .diagnostics
        .iter()
        .find(|item| item.severity == crate::abi::DiagnosticSeverity::Error)
        .cloned()
        .expect("the private companion reports its runtime failure")
}

#[test]
fn dynamic_cache_hit_preserves_private_companion_source_attribution() {
    let _serial = runmat_filesystem::provider_override_lock();
    let temporary = tempfile::TempDir::new().expect("isolated source root");
    std::fs::create_dir(temporary.path().join("private")).expect("private source directory");
    std::fs::write(
        temporary.path().join("helper.m"),
        "function y = helper()\n y = secret();\nend\n",
    )
    .expect("dynamic entry source");
    std::fs::write(
        temporary.path().join("private/secret.m"),
        "function y = secret()\n y = 0;\n error('private source witness');\nend\n",
    )
    .expect("private companion source");
    let mut session = RunMatSession::with_options(false, false).expect("interpreter session");
    let root = temporary.path().to_string_lossy().replace('\'', "''");
    let request = crate::abi::ExecutionRequest::for_source(
        crate::abi::SourceInput::Text {
            name: "<dynamic-source-path-setup>".into(),
            text: format!("addpath('{root}');"),
        },
        session.compat_mode(),
        crate::abi::HostExecutionPolicy::default(),
        session.workspace_handle(),
    );
    let setup = futures::executor::block_on(session.execute_request(request))
        .result
        .expect("public path setup");
    assert!(
        setup
            .diagnostics
            .iter()
            .all(|item| item.severity != crate::abi::DiagnosticSeverity::Error),
        "{setup:?}"
    );
    let first = execute_helper(&mut session);
    assert!(
        first.message.contains("private source witness"),
        "{first:?}"
    );
    assert!(
        first
            .callstack
            .iter()
            .any(|frame| frame.contains("private/secret.m")),
        "private source must own the failure: {first:?}"
    );
    let expected_private_path = temporary.path().join("private/secret.m");
    let (admitted, unit_bytes, entrypoint_origin, private_fullpath) = {
        let cache = session.dynamic_function_cache.snapshot();
        assert_eq!(cache.len(), 1, "the entry was dynamically resolved");
        let entry = cache.values().next().expect("resolved helper");
        assert_eq!(
            entry.admitted.envelope().manifest.identity.entrypoint,
            "helper"
        );
        let private_record = entry.source_records.iter().find(|record| {
            record
                .fullpath_name
                .as_deref()
                .is_some_and(|path| std::path::Path::new(path) == expected_private_path.as_path())
        });
        assert!(
            private_record.is_some(),
            "expected captured private source {expected_private_path:?}, records: {:?}",
            entry
                .source_records
                .iter()
                .map(|record| (&record.name, &record.fullpath_name))
                .collect::<Vec<_>>()
        );
        let private_fullpath = private_record
            .and_then(|record| record.fullpath_name.as_deref())
            .expect("captured private source spelling")
            .to_string();
        assert!(
            !entry
                .unit_bytes
                .windows(private_fullpath.len())
                .any(|window| window == private_fullpath.as_bytes()),
            "host display path must not enter the portable envelope"
        );
        (
            Rc::clone(&entry.admitted),
            Arc::clone(&entry.unit_bytes),
            entry.entrypoint_origin,
            private_fullpath,
        )
    };

    let second = execute_helper(&mut session);
    assert_eq!(second.code, first.code);
    assert_eq!(second.message, first.message);
    assert_eq!(second.callstack, first.callstack);
    let cache = session.dynamic_function_cache.snapshot();
    assert_eq!(cache.len(), 1);
    let entry = cache.values().next().expect("same dynamic entry");
    assert!(
        Rc::ptr_eq(&entry.admitted, &admitted),
        "admitted product cache hit"
    );
    assert!(
        Arc::ptr_eq(&entry.unit_bytes, &unit_bytes),
        "canonical envelope cache hit"
    );
    assert!(entry
        .source_records
        .iter()
        .any(|record| { record.fullpath_name.as_deref() == Some(private_fullpath.as_str()) }));
    drop(cache);

    let cached = session.dynamic_function_cache.snapshot();
    let (path, function) = cached
        .iter()
        .next()
        .map(|(path, entry)| (path.clone(), entry.binding.function))
        .expect("cached product");
    drop(cached);
    session.dynamic_function_cache.replace_binding(
        &path,
        runmat_value::BoundFunctionBinding::unavailable(function),
        Vec::new(),
        HashSet::new(),
    );
    let rebound = execute_helper(&mut session);
    assert_eq!(rebound.callstack, first.callstack);
    let rebound_cache = session.dynamic_function_cache.snapshot();
    let rebound_entry = rebound_cache.values().next().expect("rebound product");
    assert!(Rc::ptr_eq(&rebound_entry.admitted, &admitted));
    assert!(rebound_entry
        .source_records
        .iter()
        .any(|record| { record.fullpath_name.as_deref() == Some(private_fullpath.as_str()) }));
    drop(rebound_cache);

    std::fs::write(
        temporary.path().join("private/secret.m"),
        "function y = secret()\n y = 0;\n error('changed private source witness');\nend\n",
    )
    .expect("changed private companion source");
    let changed = execute_helper(&mut session);
    assert!(
        changed.message.contains("changed private source witness"),
        "a companion content revision must invalidate the product: {changed:?}"
    );
    assert!(
        changed
            .callstack
            .iter()
            .any(|frame| frame.contains("private/secret.m")),
        "the rebuilt companion retains its exact source: {changed:?}"
    );
    let cache = session.dynamic_function_cache.snapshot();
    let changed_entry = cache.values().next().expect("rebuilt helper");
    assert!(!Rc::ptr_eq(&changed_entry.admitted, &admitted));
    assert!(!Arc::ptr_eq(&changed_entry.unit_bytes, &unit_bytes));
    assert_ne!(changed_entry.entrypoint_origin, entrypoint_origin);
}

#[test]
fn committed_dynamic_binding_reuses_incoming_session_and_exact_private_source() {
    let _serial = runmat_filesystem::provider_override_lock();
    let temporary = tempfile::TempDir::new().expect("isolated source root");
    std::fs::create_dir(temporary.path().join("private")).expect("private source directory");
    std::fs::write(
        temporary.path().join("stable_helper.m"),
        concat!(
            "function y = stable_helper(fail)\n",
            " y = stable_secret(fail);\n",
            "end\n"
        ),
    )
    .expect("dynamic entry source");
    std::fs::write(
        temporary.path().join("private/stable_secret.m"),
        concat!(
            "function y = stable_secret(fail)\n",
            " y = 17;\n",
            " if fail\n",
            "  error('committed private source witness');\n",
            " end\n",
            "end\n"
        ),
    )
    .expect("private companion source");
    let mut session = RunMatSession::with_options(false, false).expect("interpreter session");
    let root = temporary.path().to_string_lossy().replace('\'', "''");
    let first = crate::execute_text_request_for_testing(
        &mut session,
        &format!("addpath('{root}'); first = stable_helper(false);"),
    )
    .expect("successful dynamic invocation");
    assert!(first.error.is_none(), "first invocation commits: {first:?}");
    assert_eq!(
        session.get_variables().unwrap().get("first"),
        Some(&Value::Num(17.0))
    );
    let binding = session
        .dynamic_function_cache
        .snapshot()
        .values()
        .next()
        .expect("committed dynamic product")
        .binding
        .clone();

    let second = crate::execute_text_request_for_testing(&mut session, "stable_helper(true);")
        .expect("failing dynamic invocation returns an outcome");
    let error = second
        .error
        .as_ref()
        .expect("private callback reports its failure");
    assert!(error.message.contains("committed private source witness"));
    assert!(
        error
            .context
            .call_stack
            .iter()
            .any(|frame| frame.contains("private/stable_secret.m")),
        "warm binding must resolve its retained private source in the incoming session: {error:?}"
    );
    assert_eq!(
        session
            .dynamic_function_cache
            .snapshot()
            .values()
            .next()
            .expect("same cached product")
            .binding,
        binding,
        "a committed warm hit must not rebind the admitted product"
    );
}

#[test]
fn escaped_dynamic_private_callback_exports_after_source_session_closes() {
    let _serial = runmat_filesystem::provider_override_lock();
    let temporary = tempfile::TempDir::new().expect("isolated source root");
    std::fs::create_dir(temporary.path().join("private")).expect("private source directory");
    std::fs::write(
        temporary.path().join("callback_factory.m"),
        "function callback = callback_factory()\n callback = @retained_secret;\nend\n",
    )
    .expect("dynamic callback factory");
    std::fs::write(
        temporary.path().join("private/retained_secret.m"),
        concat!(
            "function value = retained_secret(input)\n",
            " value = input;\n",
            " error('dynamic retained private witness');\n",
            "end\n"
        ),
    )
    .expect("private retained callback");

    let mut source = RunMatSession::with_options(false, false).expect("source session");
    let root = temporary.path().to_string_lossy().replace('\'', "''");
    let setup = crate::execute_text_request_for_testing(
        &mut source,
        &format!(
            "addpath('{root}'); escaped_dynamic = callback_factory(); error('after dynamic publication');"
        ),
    )
    .expect("later error is returned in the execution outcome");
    assert!(
        setup.error.is_some(),
        "the foreground request must fail later"
    );
    let escaped = source
        .get_variables()
        .expect("source workspace")
        .get("escaped_dynamic")
        .cloned()
        .expect("dynamic callback publication survives the later error");
    let source_runtime = source.runtime_context.clone();
    let service = std::rc::Rc::clone(
        source_runtime
            .service_ports()
            .require_callable_artifacts("export a dynamic callback")
            .expect("source callable artifact service"),
    );
    let bundle = futures::executor::block_on(service.export(
        source_runtime,
        vec![escaped],
        runmat_runtime::context::RuntimeCallableArtifactLimits::default(),
    ))
    .expect("export dynamic callback and its private definition");
    drop(source);
    drop(temporary);

    let destination = RunMatSession::with_options(false, false).expect("destination session");
    let destination_runtime = destination.runtime_context.clone();
    let service = std::rc::Rc::clone(
        destination_runtime
            .service_ports()
            .require_callable_artifacts("import a dynamic callback")
            .expect("destination callable artifact service"),
    );
    let prepared = futures::executor::block_on(service.prepare_admit(
        destination_runtime.clone(),
        bundle,
        runmat_runtime::context::RuntimeCallableArtifactLimits::default(),
    ))
    .expect("prepare dynamic callback import");
    let committed = prepared.commit().expect("commit dynamic callback import");
    let Value::BoundFunctionHandle(handle) = committed.values()[0].clone() else {
        panic!("dynamic callback must retain exact bound authority");
    };
    let data = handle.into_data();
    let mut error = futures::executor::block_on(
        runmat_runtime::user_functions::try_call_bound_function_in_context(
            &destination_runtime,
            runmat_value::BoundFunctionBinding {
                function: data.function,
                authority: data.authority,
            },
            &[Value::Num(3.0)],
            1,
        ),
    )
    .expect("selected destination invoker")
    .expect_err("private callback reports its retained failure");
    runmat_runtime::execution::preserve_source_call_stack_in_context(
        &mut error,
        &destination_runtime,
    );
    assert!(
        error
            .context
            .call_stack
            .iter()
            .any(|frame| frame.contains("private/retained_secret.m")),
        "imported callback must retain its original private source: {error:?}"
    );
}

#[test]
fn ordinary_dynamic_function_preserves_unused_companion_parent_behavior() {
    let _serial = runmat_filesystem::provider_override_lock();
    let temporary = tempfile::TempDir::new().expect("source root");
    std::fs::write(
        temporary.path().join("ordinary_math.m"),
        "function y = ordinary_math(x)\ny = x + 1;\nend\n",
    )
    .expect("ordinary function");
    std::fs::write(
        temporary.path().join("UnusedCompanion.m"),
        "classdef UnusedCompanion < AbsentCompanionBase\nend\n",
    )
    .expect("unrelated companion class");
    let mut session = RunMatSession::with_options(false, false).expect("session");
    let root = temporary.path().to_string_lossy().replace('\'', "''");
    let first = crate::execute_text_request_for_testing(
        &mut session,
        &format!("addpath('{root}'); first = ordinary_math(18);"),
    )
    .expect("cold ordinary invocation");
    assert!(first.error.is_none(), "{first:?}");
    assert_eq!(
        session.get_variables().unwrap().get("first"),
        Some(&Value::Num(19.0))
    );
    let companion = runmat_types::ClassIdentity::new("UnusedCompanion")
        .expect("typed companion class identity");
    let cache = session.dynamic_function_cache.snapshot();
    assert!(cache.values().any(|entry| entry
        .admitted
        .registry()
        .functions
        .values()
        .flat_map(|body| body.instructions.iter())
        .any(|instruction| matches!(instruction, runmat_vm::Instr::RegisterClass { name, .. } if name == &companion))),
        "the fixture must actually include the companion declaration");
    let second =
        crate::execute_text_request_for_testing(&mut session, "second = ordinary_math(19);")
            .expect("warm ordinary invocation");
    assert!(second.error.is_none(), "{second:?}");
    assert_eq!(
        session.get_variables().unwrap().get("second"),
        Some(&Value::Num(20.0))
    );
}

#[test]
fn filesystem_run_retains_selected_private_callback_across_republication_and_clear() {
    let _serial = runmat_filesystem::provider_override_lock();
    let temporary = tempfile::TempDir::new().expect("selected filesystem source root");
    std::fs::create_dir(temporary.path().join("private")).expect("private source directory");
    let main = temporary.path().join("main.m");
    let entry = temporary.path().join("entry.m");
    let private = temporary.path().join("private/step.m");
    std::fs::write(
        &entry,
        "callback = make_callback();\nfunction output = make_callback()\n output = @step;\nend\n",
    )
    .expect("selected companion factory");
    std::fs::write(&private, "function y = step(x)\n y = x + 1;\nend\n")
        .expect("selected private callback");
    let entry_path = entry.to_string_lossy().replace('\'', "''");
    std::fs::write(
        &main,
        format!("run('{entry_path}'); old_callback = callback; first = feval(old_callback, 40);\n"),
    )
    .expect("initial source script");

    let mut session = RunMatSession::with_options(false, false).expect("session");
    let execute_main = |session: &mut RunMatSession| {
        let request = crate::abi::ExecutionRequest::for_source(
            crate::abi::SourceInput::Path(main.to_string_lossy().into_owned()),
            session.compat_mode(),
            crate::abi::HostExecutionPolicy::default(),
            session.workspace_handle(),
        );
        futures::executor::block_on(session.execute_request(request))
            .result
            .expect("filesystem request returns an outcome")
    };
    let first = execute_main(&mut session);
    assert!(
        first
            .diagnostics
            .iter()
            .all(|item| item.severity != crate::abi::DiagnosticSeverity::Error),
        "{first:?}"
    );
    assert_eq!(
        session.get_variables().unwrap().get("first"),
        Some(&Value::Num(41.0))
    );

    std::fs::write(&private, "function y = step(x)\n y = x + 3;\nend\n")
        .expect("revised private callback");
    std::fs::write(
        &entry,
        "callback = make_callback();\nfunction output = make_callback()\n output = @step;\n % revised selected factory\nend\n",
    )
    .expect("revised same-path factory");
    std::fs::write(
        &main,
        format!(
            "run('{entry_path}'); new_callback = callback; old_after = feval(old_callback, 40); new_after = feval(new_callback, 40); clear entry; old_after_clear = feval(old_callback, 40); new_after_clear = feval(new_callback, 40);\n"
        ),
    )
    .expect("revised source script");
    let second = execute_main(&mut session);
    assert!(
        second
            .diagnostics
            .iter()
            .all(|item| item.severity != crate::abi::DiagnosticSeverity::Error),
        "{second:?}"
    );
    let values = session.get_variables().expect("retained source workspace");
    for (name, expected) in [
        ("old_after", 41.0),
        ("new_after", 43.0),
        ("old_after_clear", 41.0),
        ("new_after_clear", 43.0),
    ] {
        assert_eq!(values.get(name), Some(&Value::Num(expected)), "{name}");
    }
}
