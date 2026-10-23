use super::*;
use crate::bytecode::BytecodeDottedInvokeArgument;
use runmat_types::{FunctionSourceRole, MemberName};

#[test]
fn session_binding_rejects_text_and_accepts_explicit_handle() {
    use crate::bytecode::FunctionRegistry;

    let helper = FunctionId(7);
    let mut bytecode = Bytecode::with_instructions(vec![Instr::LoadCharRow("helper".into())], 0);
    bytecode.function_registry = FunctionRegistry::new(HashMap::from([(
        helper,
        FunctionBytecode {
            function: helper,
            display_name: "helper".into(),
            source_role: FunctionSourceRole::Local,
            ..Default::default()
        },
    )]));
    let binding =
        SessionCallableBinding::named(0, helper, "helper".into(), FunctionReferenceKind::Local);
    assert!(remap_session_entrypoint(&mut bytecode, &HashMap::new(), &[binding]).is_err());
    assert!(matches!(bytecode.instructions[0], Instr::LoadCharRow(_)));

    bytecode.instructions[0] = Instr::CreateFunctionHandle("helper".into());
    let binding =
        SessionCallableBinding::named(0, helper, "helper".into(), FunctionReferenceKind::Local);
    remap_session_entrypoint(&mut bytecode, &HashMap::new(), &[binding])
        .expect("explicit source handle may bind");
    assert!(matches!(
        &bytecode.instructions[0],
        Instr::CreateBoundFunctionHandle(function, name, FunctionSourceRole::Local)
            if *function == helper && name == "helper"
    ));
}

#[test]
fn nested_parent_remaps_without_changing_portable_definition_identity() {
    let old_parent = FunctionId(2);
    let new_parent = FunctionId(42);
    let old_nested = FunctionId(3);
    let new_nested = FunctionId(43);
    let mut nested = FunctionBytecode {
        function: old_nested,
        display_name: "inner".into(),
        source_role: FunctionSourceRole::Nested,
        lexical_parent: Some(old_parent),
        portable_function: runmat_types::ProgramFunctionId(3),
        portable_lexical_parent: Some(runmat_types::ProgramFunctionId(2)),
        inherited_storage_slots: vec![crate::bytecode::InheritedStorageSlot {
            binding: runmat_types::BindingId(9),
            slot: 1,
            owner: old_parent,
            portable_owner: runmat_types::ProgramFunctionId(2),
            name: "counter".into(),
            owner_function: "outer".into(),
            kind: crate::bytecode::InheritedStorageKind::Persistent,
        }],
        ..Default::default()
    };
    let fingerprint =
        crate::bytecode::program::semantic_fingerprint::semantic_definition_fingerprint(&nested)
            .expect("original definition fingerprint");
    remap_closed_function(
        &mut nested,
        &HashMap::from([(old_parent, new_parent), (old_nested, new_nested)]),
    )
    .expect("closed nested projection");
    assert_eq!(nested.function, new_nested);
    assert_eq!(nested.lexical_parent, Some(new_parent));
    assert_eq!(nested.inherited_storage_slots[0].owner, new_parent);
    assert_eq!(
        nested.inherited_storage_slots[0].portable_owner,
        runmat_types::ProgramFunctionId(2)
    );
    assert_eq!(
        nested.portable_lexical_parent,
        Some(runmat_types::ProgramFunctionId(2))
    );
    let remapped =
        crate::bytecode::program::semantic_fingerprint::semantic_definition_fingerprint(&nested)
            .expect("remapped definition fingerprint");
    assert_ne!(
        remapped, fingerprint,
        "runtime identity is part of the fingerprint"
    );
    nested.portable_lexical_parent = Some(runmat_types::ProgramFunctionId(77));
    assert_ne!(
        crate::bytecode::program::semantic_fingerprint::semantic_definition_fingerprint(&nested)
            .expect("distinct portable parent fingerprint"),
        remapped,
        "different portable lexical parents must not share a definition identity"
    );
}

#[test]
fn session_publication_projects_nested_parent_and_rejects_unproved_external_eval_parent() {
    let compiler_parent = FunctionId(2);
    let destination_parent = FunctionId(42);
    let compiler_nested = FunctionId(3);
    let destination_nested = FunctionId(43);
    let mut nested = FunctionBytecode {
        function: compiler_nested,
        source_role: FunctionSourceRole::Nested,
        lexical_parent: Some(compiler_parent),
        portable_function: runmat_types::ProgramFunctionId(3),
        portable_lexical_parent: Some(runmat_types::ProgramFunctionId(2)),
        ..Default::default()
    };
    remap_session_function(
        &mut nested,
        destination_nested,
        None,
        &HashMap::from([
            (compiler_parent, destination_parent),
            (compiler_nested, destination_nested),
        ]),
    )
    .expect("Core session publication projects lexical parent");
    assert_eq!(nested.function, destination_nested);
    assert_eq!(nested.lexical_parent, Some(destination_parent));
    assert_eq!(
        nested.portable_lexical_parent,
        Some(runmat_types::ProgramFunctionId(2))
    );

    let mut eval_nested = FunctionBytecode {
        function: compiler_nested,
        source_role: FunctionSourceRole::Nested,
        lexical_parent: Some(destination_parent),
        portable_lexical_parent: Some(runmat_types::ProgramFunctionId(2)),
        ..Default::default()
    };
    assert!(remap_session_function(
        &mut eval_nested,
        destination_nested,
        None,
        // This external parent has a numeric ID but is not an admitted member
        // of the eval unit's remap.
        &HashMap::from([(compiler_nested, destination_nested)]),
    )
    .is_err());
    assert_eq!(eval_nested.function, compiler_nested);
    assert_eq!(eval_nested.lexical_parent, Some(destination_parent));
}

#[test]
fn closed_nested_parent_absence_rejects_without_mutating_definition() {
    let original = FunctionBytecode {
        function: FunctionId(3),
        source_role: FunctionSourceRole::Nested,
        lexical_parent: Some(FunctionId(2)),
        portable_function: runmat_types::ProgramFunctionId(3),
        portable_lexical_parent: Some(runmat_types::ProgramFunctionId(2)),
        ..Default::default()
    };
    let mut closed = original.clone();
    assert!(remap_closed_function(
        &mut closed,
        &HashMap::from([(FunctionId(3), FunctionId(43))]),
    )
    .is_err());
    assert_eq!(closed.function, original.function);
    assert_eq!(closed.lexical_parent, original.lexical_parent);

    let mut published = original.clone();
    assert!(remap_closed_session_function(
        &mut published,
        FunctionId(43),
        &HashMap::from([(FunctionId(3), FunctionId(43))]),
    )
    .is_err());
    assert_eq!(published.function, original.function);
    assert_eq!(published.lexical_parent, original.lexical_parent);
}

#[test]
fn every_static_dependency_shape_uses_the_same_remap() {
    let old = FunctionId(7);
    let new = FunctionId(71);
    let remap = HashMap::from([(old, new)]);
    let identity = || CallableIdentity::ExternalFunction {
        function: old,
        display_name: "helper".into(),
    };
    let dotted = || BytecodeSubscriptStep::DottedInvoke {
        member: MemberName("method".into()),
        arguments: vec![BytecodeDottedInvokeArgument::Colon],
        semantic_fallback: Some(std::sync::Arc::new(identity())),
    };
    let mut instructions = vec![
        Instr::CreateExternalBoundFunctionHandle(old, "helper".into(), FunctionSourceRole::Primary),
        Instr::CallWorkspaceFirstMulti {
            name: "helper".into(),
            identity: identity(),
            fallback_policy: runmat_types::CallableFallbackPolicy::ExternalBoundary,
            bare_identifier: true,
            arg_count: 0,
            out_count: 1,
        },
        Instr::ReadSubscriptPath {
            steps: vec![dotted()],
            selection: runmat_types::SequenceUse::RequireSingle,
            context: runmat_types::ObjectIndexingContext::Expression,
            to_sequence_register: false,
        },
        Instr::CaptureSubscriptPath {
            steps: vec![dotted()],
            sequence_slot: 0,
            context: runmat_types::ObjectIndexingContext::Expression,
        },
        Instr::BeginSubscriptEndReceiver {
            prefix: vec![dotted()],
        },
    ];
    for instruction in &mut instructions {
        let context = RemapContext {
            ids: &remap,
            policy: RemapPolicy::ClosedUnit,
        };
        remap_instruction_with_policy(instruction, &context);
    }
    let mut observed = Vec::new();
    for instruction in &instructions {
        visit_instruction(instruction, |function| observed.push(function));
    }
    assert_eq!(observed, vec![new; 5]);

    let mut body = FunctionBytecode {
        function: old,
        portable_function: runmat_types::ProgramFunctionId(3),
        source_role: FunctionSourceRole::Local,
        instructions,
        ..Default::default()
    };
    remap_closed_function(&mut body, &remap).expect("closed projection");
    assert_eq!(body.function, new);
    assert_eq!(body.portable_function, runmat_types::ProgramFunctionId(3));
}

#[test]
fn eval_local_remap_does_not_capture_parent_external_identity_with_same_numeric_id() {
    let overlapping = FunctionId(7);
    let eval_destination = FunctionId(71);
    let remap = HashMap::from([(overlapping, eval_destination)]);
    let external = || CallableIdentity::ExternalFunction {
        function: overlapping,
        display_name: "parent_helper".into(),
    };
    let mut body = FunctionBytecode {
        // This body belongs to the eval compilation and must move.
        function: overlapping,
        portable_function: runmat_types::ProgramFunctionId(0),
        source_role: FunctionSourceRole::Local,
        instructions: vec![
            // These references were resolved against the selected parent
            // registry. Their numeric collision with the eval body is not
            // identity authority.
            Instr::CreateExternalBoundFunctionHandle(
                overlapping,
                "parent_helper".into(),
                FunctionSourceRole::Primary,
            ),
            Instr::CallFunctionMulti {
                identity: external(),
                fallback_policy: runmat_types::CallableFallbackPolicy::ExternalBoundary,
                arg_count: 0,
                out_count: 1,
            },
        ],
        ..Default::default()
    };

    remap_eval_local_function(&mut body, &remap).expect("eval-local projection");

    assert_eq!(body.function, eval_destination);
    let Instr::CreateExternalBoundFunctionHandle(function, ..) = &body.instructions[0] else {
        panic!("expected parent external handle");
    };
    assert_eq!(*function, overlapping);
    let Instr::CallFunctionMulti { identity, .. } = &body.instructions[1] else {
        panic!("expected parent external call");
    };
    assert_eq!(identity, &external());

    // The same shapes belong to the admitted unit under closed-unit staging
    // and therefore must be remapped there.
    remap_closed_function(&mut body, &remap).expect("closed projection");
    let Instr::CreateExternalBoundFunctionHandle(function, ..) = &body.instructions[0] else {
        panic!("expected admitted external handle");
    };
    assert_eq!(*function, eval_destination);
    let Instr::CallFunctionMulti { identity, .. } = &body.instructions[1] else {
        panic!("expected admitted external call");
    };
    assert_eq!(
        identity,
        &CallableIdentity::ExternalFunction {
            function: eval_destination,
            display_name: "parent_helper".into(),
        }
    );
}
