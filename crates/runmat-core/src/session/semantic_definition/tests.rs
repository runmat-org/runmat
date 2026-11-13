use super::*;
use runmat_hir::FunctionId;
use runmat_vm::{Bytecode, BytecodeSourceRef, FunctionBytecode, FunctionRegistry, Instr};
use std::collections::{HashMap, HashSet};

#[cfg(not(target_arch = "wasm32"))]
#[path = "tests/lexical_parent.rs"]
mod lexical_parent;

thread_local! {
    static ISSUED: (runmat_runtime::context::RuntimeContext, std::cell::RefCell<Vec<runmat_runtime::source_context::RuntimeSourceLease>>) = (
        runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        )),
        std::cell::RefCell::new(Vec::new()),
    );
}

fn runtime_ref(source: usize) -> BytecodeSourceRef {
    ISSUED.with(|(runtime, leases)| {
        let mut leases = leases.borrow_mut();
        while leases.len() <= source {
            let ordinal = leases.len();
            leases.push(
                runmat_runtime::source_context::lease_source_context_in_context(
                    runtime,
                    format!("source-{ordinal}.m"),
                    None,
                    format!("source {ordinal}"),
                )
                .expect("test source owner"),
            );
        }
        BytecodeSourceRef::Runtime(leases[source].source_ref())
    })
}

fn separately_owned_runtime_ref(
    name: &str,
    text: &str,
) -> (
    runmat_runtime::context::RuntimeContext,
    runmat_runtime::source_context::RuntimeSourceLease,
    BytecodeSourceRef,
) {
    let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let lease =
        runmat_runtime::source_context::lease_source_context_in_context(&runtime, name, None, text)
            .expect("separate test source owner");
    let source = BytecodeSourceRef::Runtime(lease.source_ref());
    (runtime, lease, source)
}

fn definition(id: usize, name: &str, callee: usize) -> FunctionBytecode {
    FunctionBytecode {
        function: FunctionId(id),
        display_name: name.into(),
        source_id: Some(runtime_ref(5)),
        instructions: vec![Instr::CreateBoundFunctionHandle(
            FunctionId(callee),
            "callee".into(),
            runmat_types::FunctionSourceRole::Unknown,
        )],
        ..Default::default()
    }
}

#[test]
fn session_source_revision_is_semantic_without_portable_serialization() {
    let mut function = definition(71, "bound_callback", 71);
    let original = super::fingerprint(&function).expect("bound source has a semantic revision");
    assert!(
        serde_json::to_vec(&function).is_err(),
        "runtime authority remains nonportable"
    );
    function.source_id = Some(runtime_ref(6));
    assert_ne!(
        original,
        super::fingerprint(&function).expect("changed source has a semantic revision")
    );
}

fn group(base: usize) -> FunctionRegistry {
    FunctionRegistry::new(HashMap::from([
        (FunctionId(base), definition(base, "first", base + 1)),
        (FunctionId(base + 1), definition(base + 1, "second", base)),
    ]))
}

fn binding_origin(
    label: &str,
    function: runmat_types::ProgramFunctionId,
) -> runmat_execution::FunctionBindingOrigin {
    let source =
        crate::ExecutableSource::new(label, format!("{label}.m"), "function y = f(x); y = x; end");
    let revision = crate::ExecutableRevision::derive(
        &source,
        None,
        crate::program_environment(crate::CompatMode::RunMat),
    );
    runmat_execution::FunctionBindingOrigin::from_revision(&revision.program_revision, function)
        .expect("test binding origin")
}

#[test]
fn finalized_publication_uses_exact_decorated_body_and_registry_owners() {
    let function = FunctionId(41);
    let mut prepared = definition(function.0, "published", function.0);
    prepared.portable_function = runmat_types::ProgramFunctionId(3);
    prepared.binding_origin = Some(binding_origin("published-v1", prepared.portable_function));
    prepared.source_file = Some("/durable/published.m".into());
    let owner = std::sync::Arc::clone(&prepared.definition_owner);
    let durable_source = prepared.source_id.expect("durable source");
    let mut publication = FunctionRegistry::new(HashMap::from([(function, prepared.clone())]));
    publication.names.insert("published_alias".into(), function);

    let mut decorated = prepared;
    let (_invocation_runtime, _invocation_lease, invocation_source) =
        separately_owned_runtime_ref("source-5.m", "source 5");
    decorated.source_id = Some(invocation_source);
    let point = runmat_types::ProgramPointId {
        function: decorated.portable_function,
        block: 0,
        position: 0,
    };
    decorated.regions.push(runmat_vm::BytecodeRegion {
        id: runmat_types::RegionId {
            function: decorated.portable_function,
            ordinal: 0,
        },
        entry: runmat_vm::BytecodeRegionBoundary { point, pc: 0 },
        exits: vec![runmat_vm::BytecodeRegionBoundary { point, pc: 4 }],
    });
    let mut executable = FunctionRegistry::new(HashMap::from([(function, decorated.clone())]));
    executable.names.insert("published_alias".into(), function);

    super::finalize_current_definitions(&mut publication, &HashSet::from([function]), &executable)
        .expect("finalize exact body");

    let finalized = publication.get(function).expect("published body");
    assert_eq!(finalized.regions, decorated.regions);
    assert_eq!(finalized.source_id, Some(durable_source));
    assert_ne!(finalized.source_id, decorated.source_id);
    assert!(std::sync::Arc::ptr_eq(&finalized.definition_owner, &owner));
    assert_eq!(publication.resolve_name("published_alias"), Some(function));
    assert_eq!(
        publication.source_functions[&durable_source],
        vec![function]
    );
}

#[test]
fn finalized_publication_validates_every_owner_before_mutation() {
    let first = FunctionId(51);
    let second = FunctionId(52);
    let mut first_body = definition(first.0, "first-published", first.0);
    first_body.binding_origin = Some(binding_origin("first-v1", first_body.portable_function));
    let mut second_body = definition(second.0, "second-published", second.0);
    second_body.binding_origin = Some(binding_origin("second-v1", second_body.portable_function));
    for (conflict, expected) in [(0, "source content"), (1, "binding origin")] {
        let first_owner = std::sync::Arc::clone(&first_body.definition_owner);
        let before_source = first_body.source_id.expect("source");
        let mut publication = FunctionRegistry::new(HashMap::from([
            (first, first_body.clone()),
            (second, second_body.clone()),
        ]));
        let mut first_decorated = first_body.clone();
        first_decorated.instructions.push(Instr::LoadConst(17.0));
        let mut second_conflict = second_body.clone();
        match conflict {
            0 => second_conflict.source_id = Some(runtime_ref(6)),
            1 => {
                second_conflict.binding_origin = Some(binding_origin(
                    "second-conflict",
                    second_conflict.portable_function,
                ));
            }
            _ => unreachable!(),
        }
        let executable = FunctionRegistry::new(HashMap::from([
            (first, first_decorated),
            (second, second_conflict),
        ]));

        let error = super::finalize_current_definitions(
            &mut publication,
            &HashSet::from([first, second]),
            &executable,
        )
        .expect_err("conflicting owner facts must fail before replacement");
        assert!(error.contains(expected), "{error}");
        assert_eq!(publication.get(first).unwrap().instructions.len(), 1);
        assert!(std::sync::Arc::ptr_eq(
            &publication.get(first).unwrap().definition_owner,
            &first_owner,
        ));
        let mut source_members = publication.source_functions[&before_source].clone();
        source_members.sort_by_key(|function| function.0);
        assert_eq!(source_members, vec![first, second]);
    }
}

#[test]
fn finalized_publication_does_not_publish_execution_only_companions() {
    let public = FunctionId(61);
    let private = FunctionId(62);
    let mut public_body = definition(public.0, "public", public.0);
    public_body.binding_origin = Some(binding_origin("public-v1", public_body.portable_function));
    let mut private_body = definition(private.0, "private", private.0);
    private_body.binding_origin =
        Some(binding_origin("private-v1", private_body.portable_function));
    let mut publication = FunctionRegistry::new(HashMap::from([(public, public_body.clone())]));
    let executable = FunctionRegistry::new(HashMap::from([
        (public, public_body),
        (private, private_body),
    ]));

    super::finalize_current_definitions(
        &mut publication,
        &HashSet::from([public, private]),
        &executable,
    )
    .expect("private companion remains execution-only");
    assert!(publication.get(private).is_none());
    assert!(publication.resolve_name("private").is_none());
}

#[test]
fn complete_mutual_group_reuses_normalized_identities() {
    let incoming = group(1);
    let published = group(100);
    let remap =
        unchanged_closed_product_remap(&Bytecode::empty(), &incoming, &published, &HashSet::new())
            .unwrap();
    assert_eq!(remap.get(&FunctionId(1)), Some(&FunctionId(100)));
    assert_eq!(remap.get(&FunctionId(2)), Some(&FunctionId(101)));
}

#[test]
fn changed_body_source_scope_capture_layout_and_alias_do_not_reuse() {
    let published = group(100);
    for mutation in 0..6 {
        let mut incoming = group(1);
        let function = incoming.functions.get_mut(&FunctionId(1)).unwrap();
        match mutation {
            0 => function.instructions.push(Instr::LoadConst(9.0)),
            1 => function.source_id = Some(runtime_ref(6)),
            2 => function.private_owner_scope = "other_scope".into(),
            3 => function.capture_slots.push(1),
            4 => {
                incoming.names.insert("new_alias".into(), FunctionId(1));
            }
            5 => function.varargin_slot = Some(0),
            _ => unreachable!(),
        }
        assert!(unchanged_closed_product_remap(
            &Bytecode::empty(),
            &incoming,
            &published,
            &HashSet::new()
        )
        .is_none());
    }
}

#[test]
fn removed_same_source_definition_does_not_reuse_subset() {
    let incoming =
        FunctionRegistry::new(HashMap::from([(FunctionId(1), definition(1, "first", 1))]));
    let published = FunctionRegistry::new(HashMap::from([
        (FunctionId(100), definition(100, "first", 100)),
        (FunctionId(101), definition(101, "removed", 101)),
    ]));
    assert!(unchanged_closed_product_remap(
        &Bytecode::empty(),
        &incoming,
        &published,
        &HashSet::new()
    )
    .is_none());
}

#[test]
fn aliases_must_match_in_both_directions() {
    let mut incoming = group(1);
    let mut published = group(100);
    published.names.insert("alias".into(), FunctionId(100));
    assert!(unchanged_closed_product_remap(
        &Bytecode::empty(),
        &incoming,
        &published,
        &HashSet::new()
    )
    .is_none());
    incoming.names.insert("alias".into(), FunctionId(1));
    assert!(unchanged_closed_product_remap(
        &Bytecode::empty(),
        &incoming,
        &published,
        &HashSet::new()
    )
    .is_some());
    incoming.names.insert("alias".into(), FunctionId(2));
    assert!(unchanged_closed_product_remap(
        &Bytecode::empty(),
        &incoming,
        &published,
        &HashSet::new()
    )
    .is_none());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn repeated_source_reuses_remapped_dotted_local_function_during_hot_native_execution() {
    runmat_gc::gc_test_context(|| {
        let mut session = super::super::RunMatSession::with_options(true, false).unwrap();
        session.set_native_tiering_config_for_testing(runmat_jit::tiering::TieringConfig {
            generic_hot_threshold: 1,
            specialized_hot_threshold: 100,
            deterministic: true,
            ..Default::default()
        });
        let setup = "seed_value = seed(1);\nfunction out = seed(x)\nout = x + 1;\nend";
        let outcome = crate::execute_text_request_for_testing(&mut session, setup).unwrap();
        assert!(outcome.error.is_none(), "{:?}", outcome.error);
        let seed = session.function_registry.resolve_name("seed").unwrap();

        let source = "obj = 2; result = obj.bump(3);\nfunction out = bump(receiver, value)\nout = receiver + value;\nend";
        let mut published_bump = None;
        let mut native_calls = 0;
        for _ in 0..5 {
            let outcome = crate::execute_text_request_for_testing(&mut session, source).unwrap();
            assert!(outcome.error.is_none(), "{:?}", outcome.error);
            native_calls += usize::from(outcome.used_jit);
            assert_eq!(
                session.get_variables().unwrap().get("result"),
                Some(&runmat_value::Value::Num(5.0))
            );
            let bump = session.function_registry.resolve_name("bump").unwrap();
            assert_ne!(bump, seed);
            if let Some(previous) = published_bump {
                assert_eq!(
                    bump, previous,
                    "unchanged local definition must retain its session identity"
                );
            }
            published_bump = Some(bump);
        }
        assert!(
            native_calls >= 2,
            "recompiled dotted call must run natively"
        );
    });
}

#[test]
fn unrelated_published_source_does_not_prevent_reuse() {
    let incoming = group(1);
    let mut functions = group(100).into_functions();
    let mut unrelated = definition(200, "unrelated", 200);
    unrelated.source_id = Some(runtime_ref(6));
    functions.insert(FunctionId(200), unrelated);
    let mut published = FunctionRegistry::new(functions);
    published
        .names
        .insert("unrelated_alias".into(), FunctionId(200));
    assert!(unchanged_closed_product_remap(
        &Bytecode::empty(),
        &incoming,
        &published,
        &HashSet::new()
    )
    .is_some());
}

#[test]
fn source_identity_includes_text_name_and_path() {
    let mut pool = crate::source_pool::SourcePool::default();
    let source = pool.intern_with_fullpath("unit", Some("/first/unit.m"), "f; g;");
    assert_eq!(
        source,
        pool.intern_with_fullpath("unit", Some("/first/unit.m"), "f; g;")
    );
    assert_ne!(
        source,
        pool.intern_with_fullpath("unit", Some("/first/unit.m"), "f;")
    );
    assert_ne!(
        source,
        pool.intern_with_fullpath("other", Some("/first/unit.m"), "f; g;")
    );
    assert_ne!(
        source,
        pool.intern_with_fullpath("unit", Some("/other/unit.m"), "f; g;")
    );
}

#[test]
fn publishing_function_captures_original_fullpath_before_source_retirement() {
    let mut session = crate::RunMatSession::with_options(false, false).unwrap();
    session.active_source_name = "old_file".into();
    session.active_source_fullpath_name = Some("/old/old_file.m".into());
    let text = "function y = old_file(x)\ny = x + 1;\nend";
    let mut prepared = session
        .compile_input(text)
        .expect("compile original source");
    let source = crate::ExecutableSource::new("test:old_file", "old_file", text);
    let revision = crate::ExecutableRevision::derive(
        &source,
        None,
        crate::program_environment(session.compat_mode),
    );
    let source_map = session.executable_source_map(&source).expect("source map");
    prepared
        .attach_function_origins(&revision.program_revision, &source_map)
        .expect("publish source facts");
    let id = *prepared
        .current_function_ids
        .iter()
        .next()
        .expect("function");
    let published = prepared
        .function_registry_after_success
        .get(id)
        .expect("published body");
    assert_eq!(published.source_file.as_deref(), Some("/old/old_file.m"));
    let lease = runmat_value::FunctionDefinitionOwner::issue(&published.definition_owner, id.0);
    let source_id = published
        .source_id
        .and_then(BytecodeSourceRef::runtime)
        .expect("published runtime source");
    let mut current = prepared.function_registry_after_success;
    current.remove_source(source_id);
    let retained = current
        .get_owned(id, &lease)
        .expect("retired body retained");
    assert_eq!(retained.source_file.as_deref(), Some("/old/old_file.m"));
}

#[test]
fn function_file_localfunctions_excludes_primary_at_public_call_boundary() {
    runmat_gc::gc_test_context(|| {
        let mut session = crate::RunMatSession::with_options(false, false).unwrap();
        let source = "function handles = primary_file()\nhandles = localfunctions();\nend\nfunction y = child_local(x)\ny = x + 1;\nend";
        let defined = crate::execute_text_request_for_testing(&mut session, source).unwrap();
        assert!(defined.error.is_none(), "{:?}", defined.error);
        let called = crate::execute_text_request_for_testing(
            &mut session,
            "selected_handles = primary_file();",
        )
        .unwrap();
        assert!(called.error.is_none(), "{:?}", called.error);
        let variables = session.get_variables().unwrap();
        let Some(runmat_value::Value::Cell(handles)) = variables.get("selected_handles") else {
            panic!("localfunctions must return a cell");
        };
        assert_eq!(handles.data().len(), 1);
        assert!(matches!(
            &handles.data()[0],
            runmat_value::Value::BoundFunctionHandle(handle)
                if handle.name == "child_local"
                    && handle.source_role == runmat_types::FunctionSourceRole::Local
        ));
    });
}

#[test]
fn repeated_captured_definition_survives_workspace_growth() {
    runmat_gc::gc_test_context(|| {
        let mut session = crate::RunMatSession::with_options(false, false).unwrap();
        let execute = |session: &mut crate::RunMatSession, source: &str| {
            let outcome = crate::execute_text_request_for_testing(session, source).unwrap();
            assert!(outcome.error.is_none(), "{source}: {:?}", outcome.error);
        };
        execute(&mut session, "offset = 1;");
        let source = "f = @(x) x + offset; answer = f(2); assert(answer == offset + 2);";
        execute(&mut session, source);
        let first = session.function_registry.clone();
        execute(&mut session, "saved = f;");
        execute(&mut session, "offset = offset + 1;");
        execute(&mut session, source);
        let second = &session.function_registry;
        for (name, id) in &first.names {
            let original = first.get(*id).unwrap();
            let next_id = second
                .resolve_name(name)
                .expect("definition remains published");
            let mut next = second.get(next_id).unwrap().clone();
            next.rebind_identity(*id);
            let before = super::fingerprint(original).unwrap();
            let after = super::fingerprint(&next).unwrap();
            assert_eq!(
                *id, next_id,
                "unchanged capture definition was retired; semantic fingerprints: {before:?} -> {after:?}"
            );
        }
        execute(&mut session, "assert(saved(2) == 3);");
    });
}

#[test]
fn retained_handle_keeps_old_definition_after_name_republication() {
    runmat_gc::gc_test_context(|| {
        let mut session = crate::RunMatSession::with_options(false, false).unwrap();
        let execute = |session: &mut crate::RunMatSession, source: &str| {
            let outcome = crate::execute_text_request_for_testing(session, source).unwrap();
            assert!(outcome.error.is_none(), "{source}: {:?}", outcome.error);
        };
        execute(&mut session, "function y = retained_f(x)\ny = x + 1;\nend");
        execute(&mut session, "old = @retained_f;");
        execute(
            &mut session,
            "function y = retained_f(x)\ny = x + 100;\nend",
        );
        execute(&mut session, "new = @retained_f;");
        execute(&mut session, "assert(old(2) == 3); assert(new(2) == 102);");
    });
}

#[test]
fn retained_static_helper_does_not_shadow_latest_dynamic_name() {
    runmat_gc::gc_test_context(|| {
        let mut session = crate::RunMatSession::with_options(false, false).unwrap();
        let execute = |session: &mut crate::RunMatSession, source: &str| {
            let outcome = crate::execute_text_request_for_testing(session, source).unwrap();
            assert!(outcome.error.is_none(), "{source}: {:?}", outcome.error);
        };
        execute(
            &mut session,
            "function y = retained_mix(x)\ny = retained_helper(x) + feval('retained_helper', x);\nend\nfunction y = retained_helper(x)\ny = x + 1;\nend",
        );
        execute(&mut session, "old_mix = @retained_mix;");
        execute(
            &mut session,
            "function y = retained_mix(x)\ny = retained_helper(x) + feval('retained_helper', x);\nend\nfunction y = retained_helper(x)\ny = x + 100;\nend",
        );
        execute(&mut session, "new_mix = @retained_mix;");
        execute(
            &mut session,
            "old_value = old_mix(2); new_value = new_mix(2);",
        );
        let old_value = session.logical_workspace_value("old_value").unwrap();
        let new_value = session.logical_workspace_value("new_value").unwrap();
        assert_eq!(
            old_value,
            Some(runmat_value::Value::Num(105.0)),
            "retained static helper plus latest dynamic name; new value: {new_value:?}"
        );
        assert_eq!(
            new_value,
            Some(runmat_value::Value::Num(204.0)),
            "new static helper plus latest dynamic name; old value: {old_value:?}"
        );
    });
}

#[test]
fn entrypoint_text_names_stay_dynamic_after_helper_republication() {
    runmat_gc::gc_test_context(|| {
        let mut session = crate::RunMatSession::with_options(false, false).unwrap();
        let execute = |session: &mut crate::RunMatSession, source: &str| {
            let outcome = crate::execute_text_request_for_testing(session, source).unwrap();
            assert!(outcome.error.is_none(), "{source}: {:?}", outcome.error);
        };
        execute(
            &mut session,
            "function y = entry_helper(x)\ny = x + 1;\nend",
        );
        execute(&mut session, "old_entry_helper = @entry_helper;");
        execute(
            &mut session,
            "function y = entry_helper(x)\ny = x + 100;\nend",
        );
        execute(
            &mut session,
            "target = 'entry_helper'; literal = feval('entry_helper', 2); variable = feval(target, 2); explicit = feval(@entry_helper, 2); mapped = cellfun(@entry_helper, {2}); mapped_total = sum(mapped); old_value = old_entry_helper(2);",
        );
        for name in ["literal", "variable", "explicit", "mapped_total"] {
            assert_eq!(
                session.logical_workspace_value(name).unwrap(),
                Some(runmat_value::Value::Num(102.0)),
                "{name} must select the latest published helper"
            );
        }
        assert_eq!(
            session.logical_workspace_value("old_value").unwrap(),
            Some(runmat_value::Value::Num(3.0)),
            "the earlier explicit handle retains its old definition"
        );
    });
}

#[test]
fn retained_body_creates_handle_for_its_old_helper_after_republication() {
    runmat_gc::gc_test_context(|| {
        let mut session = crate::RunMatSession::with_options(false, false).unwrap();
        let execute = |session: &mut crate::RunMatSession, source: &str| {
            let outcome = crate::execute_text_request_for_testing(session, source).unwrap();
            assert!(outcome.error.is_none(), "{source}: {:?}", outcome.error);
        };
        execute(
            &mut session,
            "function h = handle_factory()\nh = @retained_helper;\nend\nfunction y = retained_helper(x)\ny = x + 1;\nend",
        );
        execute(&mut session, "old_factory = @handle_factory;");
        execute(
            &mut session,
            "function h = handle_factory()\nh = @retained_helper;\nend\nfunction y = retained_helper(x)\ny = x + 100;\nend",
        );
        execute(&mut session, "new_factory = @handle_factory;");
        execute(
            &mut session,
            "old_helper = old_factory(); new_helper = new_factory(); assert(old_helper(2) == 3); assert(new_helper(2) == 102);",
        );
    });
}
