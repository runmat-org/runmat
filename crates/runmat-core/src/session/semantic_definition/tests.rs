use super::*;
use runmat_hir::{FunctionId, SourceId};
use runmat_vm::{Bytecode, FunctionBytecode, FunctionRegistry, Instr};
use std::collections::{HashMap, HashSet};

fn definition(id: usize, name: &str, callee: usize) -> FunctionBytecode {
    FunctionBytecode {
        function: FunctionId(id),
        display_name: name.into(),
        source_id: Some(SourceId(5)),
        instructions: vec![Instr::CreateBoundFunctionHandle(
            FunctionId(callee),
            "callee".into(),
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
            1 => function.source_id = Some(SourceId(6)),
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

#[test]
fn unrelated_published_source_does_not_prevent_reuse() {
    let incoming = group(1);
    let mut functions = group(100).functions;
    let mut unrelated = definition(200, "unrelated", 200);
    unrelated.source_id = Some(SourceId(6));
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
