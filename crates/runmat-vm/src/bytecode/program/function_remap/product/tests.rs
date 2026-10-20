use super::*;
use crate::bytecode::{BytecodeSourceRef, FunctionRegistry, Instr};
use runmat_types::{FunctionSourceRole, ProgramFunctionId, ProgramSourceId};

fn decorated_registry() -> (Bytecode, FunctionRegistry) {
    let source = Some(BytecodeSourceRef::Program(ProgramSourceId(0)));
    let published = FunctionBytecode {
        function: FunctionId(7),
        portable_function: ProgramFunctionId(7),
        display_name: "stable".into(),
        source_id: source,
        instructions: vec![Instr::CreateExternalBoundFunctionHandle(
            FunctionId(7),
            "stable".into(),
            FunctionSourceRole::Primary,
        )],
        ..Default::default()
    };
    let incoming = FunctionBytecode {
        function: FunctionId(70),
        // Compiler ordinal remains stable while the session ID is fresh.
        portable_function: ProgramFunctionId(7),
        display_name: "stable".into(),
        source_id: source,
        instructions: vec![Instr::CreateExternalBoundFunctionHandle(
            FunctionId(70),
            "stable".into(),
            FunctionSourceRole::Primary,
        )],
        ..Default::default()
    };
    let mut bytecode = Bytecode::empty();
    let published = FunctionRegistry::new(HashMap::from([(published.function, published)]));
    bytecode.function_registry =
        FunctionRegistry::new(HashMap::from([(incoming.function, incoming)]));
    bytecode.bound_functions = bytecode.function_registry.functions.clone();
    bytecode.instructions = vec![Instr::CreateExternalBoundFunctionHandle(
        FunctionId(70),
        "stable".into(),
        FunctionSourceRole::Primary,
    )];
    (bytecode, published)
}

#[test]
fn eval_metadata_remap_changes_only_decorated_owner_not_local_body() {
    let (mut bytecode, _) = decorated_registry();
    bytecode.active_function = Some(FunctionId(70));
    let before = serde_json::to_vec(&bytecode.instructions).unwrap();
    remap_decorated_eval_metadata(
        &mut bytecode,
        &HashMap::from([(FunctionId(70), FunctionId(71))]),
    )
    .expect("one exact local owner");
    assert_eq!(bytecode.active_function, Some(FunctionId(71)));
    assert!(bytecode
        .function_registry
        .get_current(FunctionId(70))
        .is_some());
    assert_eq!(serde_json::to_vec(&bytecode.instructions).unwrap(), before);
    assert!(remap_decorated_eval_metadata(
        &mut bytecode,
        &HashMap::from([
            (FunctionId(70), FunctionId(71)),
            (FunctionId(72), FunctionId(71)),
        ]),
    )
    .is_err());
    assert_eq!(bytecode.active_function, Some(FunctionId(71)));
}

#[test]
fn exact_collision_preserves_published_definition_owner_and_registry_indexes() {
    let (mut bytecode, published) = decorated_registry();
    let prior = Arc::clone(
        &published
            .get_current(FunctionId(7))
            .unwrap()
            .definition_owner,
    );
    remap_decorated_session_definitions(
        &mut bytecode,
        &published,
        &HashMap::from([(FunctionId(70), FunctionId(7))]),
    )
    .unwrap();
    assert!(!bytecode
        .function_registry
        .functions
        .contains_key(&FunctionId(70)));
    let body = bytecode
        .function_registry
        .get_current(FunctionId(7))
        .unwrap();
    assert!(Arc::ptr_eq(&prior, &body.definition_owner));
    assert_eq!(body.portable_function, ProgramFunctionId(7));
    assert_eq!(
        bytecode.function_registry.resolve_name("stable"),
        Some(FunctionId(7))
    );
    assert_eq!(
        bytecode
            .function_registry
            .source_functions
            .get(&BytecodeSourceRef::Program(ProgramSourceId(0)))
            .unwrap(),
        &vec![FunctionId(7)]
    );
    assert_eq!(
        bytecode.bound_functions[&FunctionId(7)].function,
        FunctionId(7)
    );
    assert!(matches!(
        &body.instructions[0],
        Instr::CreateExternalBoundFunctionHandle(FunctionId(7), _, _)
    ));
    assert!(matches!(
        &bytecode.instructions[0],
        Instr::CreateExternalBoundFunctionHandle(FunctionId(7), _, _)
    ));
}

#[test]
fn semantic_mismatch_or_noninjective_map_leaves_whole_product_untouched() {
    let (mut bytecode, published) = decorated_registry();
    bytecode
        .function_registry
        .functions
        .get_mut(&FunctionId(70))
        .unwrap()
        .var_count = 1;
    bytecode.bound_functions = bytecode.function_registry.functions.clone();
    let before = bytecode.function_registry.clone();
    let error = remap_decorated_session_definitions(
        &mut bytecode,
        &published,
        &HashMap::from([(FunctionId(70), FunctionId(7))]),
    )
    .unwrap_err();
    assert!(error.contains("exact semantic identity"));
    assert_eq!(
        bytecode.function_registry.functions.len(),
        before.functions.len()
    );
    assert_eq!(
        bytecode.function_registry.functions[&FunctionId(70)].var_count,
        before.functions[&FunctionId(70)].var_count
    );
    assert_eq!(bytecode.bound_functions[&FunctionId(70)].var_count, 1);
    assert!(remap_decorated_session_definitions(
        &mut bytecode,
        &published,
        &HashMap::from([
            (FunctionId(70), FunctionId(7)),
            (FunctionId(8), FunctionId(7))
        ]),
    )
    .is_err());
}
