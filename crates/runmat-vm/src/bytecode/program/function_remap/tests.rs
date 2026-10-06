use super::*;
use crate::bytecode::BytecodeDottedInvokeArgument;
use runmat_types::{FunctionSourceRole, MemberName};

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
        semantic_fallback: Some(identity()),
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
    remap_closed_function(&mut body, &remap);
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

    remap_eval_local_function(&mut body, &remap);

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
    remap_closed_function(&mut body, &remap);
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
