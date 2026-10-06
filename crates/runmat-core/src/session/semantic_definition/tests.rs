use super::*;
use runmat_hir::{FunctionId, SourceId};
use runmat_vm::{Bytecode, BytecodeSourceRef, FunctionBytecode, FunctionRegistry, Instr};
use std::collections::{HashMap, HashSet};

fn definition(id: usize, name: &str, callee: usize) -> FunctionBytecode {
    FunctionBytecode {
        function: FunctionId(id),
        display_name: name.into(),
        source_id: Some(BytecodeSourceRef::Runtime(SourceId(5))),
        instructions: vec![Instr::CreateBoundFunctionHandle(
            FunctionId(callee),
            "callee".into(),
            runmat_types::FunctionSourceRole::Unknown,
        )],
        ..Default::default()
    }
}

fn group(base: usize) -> FunctionRegistry {
    FunctionRegistry::new(HashMap::from([
        (FunctionId(base), definition(base, "first", base + 1)),
        (FunctionId(base + 1), definition(base + 1, "second", base)),
    ]))
}

#[test]
fn complete_mutual_group_reuses_normalized_identities() {
    let incoming = group(1);
    let published = group(100);
    let remap =
        unchanged_unit_remap(&Bytecode::empty(), &incoming, &published, &HashSet::new()).unwrap();
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
            1 => function.source_id = Some(BytecodeSourceRef::Runtime(SourceId(6))),
            2 => function.private_owner_scope = "other_scope".into(),
            3 => function.capture_slots.push(1),
            4 => {
                incoming.names.insert("new_alias".into(), FunctionId(1));
            }
            5 => function.varargin_slot = Some(0),
            _ => unreachable!(),
        }
        assert!(
            unchanged_unit_remap(&Bytecode::empty(), &incoming, &published, &HashSet::new())
                .is_none()
        );
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
    assert!(
        unchanged_unit_remap(&Bytecode::empty(), &incoming, &published, &HashSet::new()).is_none()
    );
}

#[test]
fn aliases_must_match_in_both_directions() {
    let mut incoming = group(1);
    let mut published = group(100);
    published.names.insert("alias".into(), FunctionId(100));
    assert!(
        unchanged_unit_remap(&Bytecode::empty(), &incoming, &published, &HashSet::new()).is_none()
    );
    incoming.names.insert("alias".into(), FunctionId(1));
    assert!(
        unchanged_unit_remap(&Bytecode::empty(), &incoming, &published, &HashSet::new()).is_some()
    );
    incoming.names.insert("alias".into(), FunctionId(2));
    assert!(
        unchanged_unit_remap(&Bytecode::empty(), &incoming, &published, &HashSet::new()).is_none()
    );
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
                session.get_variables().get("result"),
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
    unrelated.source_id = Some(BytecodeSourceRef::Runtime(SourceId(6)));
    functions.insert(FunctionId(200), unrelated);
    let mut published = FunctionRegistry::new(functions);
    published
        .names
        .insert("unrelated_alias".into(), FunctionId(200));
    assert!(
        unchanged_unit_remap(&Bytecode::empty(), &incoming, &published, &HashSet::new()).is_some()
    );
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
        let Some(runmat_value::Value::Cell(handles)) =
            session.get_variables().get("selected_handles")
        else {
            panic!("localfunctions must return a cell");
        };
        assert_eq!(handles.data.len(), 1);
        assert!(matches!(
            &handles.data[0],
            runmat_value::Value::BoundFunctionHandle { name, source_role, .. }
                if name == "child_local"
                    && *source_role == runmat_types::FunctionSourceRole::Local
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
            next.rebind_identity(*id, runmat_types::ProgramFunctionId(id.0 as u32));
            let before = serde_json::to_value(original).unwrap();
            let after = serde_json::to_value(&next).unwrap();
            let differences: Vec<_> = before
                .as_object()
                .unwrap()
                .iter()
                .filter(|(field, value)| after.get(field.as_str()) != Some(*value))
                .map(|(field, _)| field.as_str())
                .collect();
            assert_eq!(
                *id, next_id,
                "unchanged capture definition was retired; differing fields: {differences:?}"
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
            "assert(old_mix(2) == 105); assert(new_mix(2) == 204);",
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
