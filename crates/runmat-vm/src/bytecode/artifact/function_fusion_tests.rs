use crate::bytecode::{
    compile::derive_semantic_fusion_instruction_windows, program::FunctionFusionScaffold,
    FunctionRegistry, FusionCandidateGroup, BYTECODE_SCHEMA_VERSION,
    FUNCTION_REGISTRY_SCHEMA_VERSION,
};

use super::{
    decode_interpreter_program_v2, encode_interpreter_program_v2, envelope::encode_payload,
    InterpreterPayloadForm,
};

fn compiled_function(source: &str) -> (FunctionRegistry, runmat_hir::FunctionId) {
    let ast = runmat_parser::parse(source).unwrap();
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).unwrap();
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).unwrap();
    assert!(hir.assembly.entrypoints.is_empty());
    let function = hir.assembly.functions[0].id;
    (
        FunctionRegistry::new(
            crate::bytecode::compile_semantic_function_registry(&hir.assembly, &mir).unwrap(),
        ),
        function,
    )
}
#[test]
fn program_v2_preserves_checked_function_axis_control_bridge() {
    let (registry, function) = compiled_function("function y=f(x)\ny=min(x+x,[],2);\nend\n");
    let body = registry
        .get(function)
        .expect("selected function definition");
    let scaffold = body.fusion_scaffold.as_ref().expect("MIR candidate");
    assert_eq!(scaffold.candidates.len(), 1);
    let candidate = &scaffold.candidates[0];
    assert_eq!(candidate.signal_count, 2);
    assert_eq!(candidate.stmt_end - candidate.stmt_start, 3);
    body.validate_fusion_scaffold()
        .expect("exact selected empty control remains admissible");

    let encoded = encode_interpreter_program_v2(&registry).expect("canonical program V2");
    let decoded = decode_interpreter_program_v2(&encoded).expect("checked program V2");
    let selected = decoded.get(function).expect("decoded selected function");
    selected
        .validate_fusion_scaffold()
        .expect("decoded control still has source authority");
    let executable = selected.execution_bytecode(&decoded);
    let (plan, graph) = executable.prepare_vm_fusion_plan();
    assert!(
        graph.is_some(),
        "ordinary function gets an actual runtime graph"
    );
    assert!(
        plan.is_some_and(|plan| plan.groups.iter().any(|group| {
            group.group.kind == runmat_accelerate::fusion::FusionKind::Reduction
                && group.group.nodes.len() >= 2
        })),
        "ordinary function keeps the producer and reduction in one checked group"
    );

    let mut bad_count = registry.clone();
    bad_count
        .functions
        .get_mut(&function)
        .unwrap()
        .fusion_scaffold
        .as_mut()
        .unwrap()
        .candidates[0]
        .signal_count = 3;
    bad_count
        .functions
        .get_mut(&function)
        .unwrap()
        .fusion_scaffold
        .as_mut()
        .unwrap()
        .signal_count = 3;
    assert!(encode_interpreter_program_v2(&bad_count).is_err());

    let mut bad_span = registry.clone();
    bad_span
        .functions
        .get_mut(&function)
        .unwrap()
        .fusion_scaffold
        .as_mut()
        .unwrap()
        .candidates[0]
        .stmt_start += 1;
    assert!(encode_interpreter_program_v2(&bad_span).is_err());

    let mut bad_control = registry.clone();
    let changed = bad_control.functions.get_mut(&function).unwrap();
    let control_pc = changed
        .instructions
        .iter()
        .position(|instruction| matches!(instruction, crate::Instr::CreateMatrix(0, 0)))
        .expect("compiled [] control");
    changed.instructions[control_pc] = crate::Instr::CreateMatrix(1, 1);
    changed.reseal_fusion_stage_after_owned_remap();
    assert!(changed.validate_fusion_stage_projection().is_ok());
    assert!(
        encode_interpreter_program_v2(&bad_control).is_err(),
        "freshly sealed nonempty control is not a compiler-owned [] bridge"
    );
}

#[test]
fn program_v2_preserves_admitted_function_hint_and_legacy_absence() {
    let (registry, function) =
        compiled_function("function y=f()\npersistent p\nt=p.*2;\np=p+t+1;\ny=p;\nend\n");
    assert!(registry.get(function).unwrap().fusion_scaffold.is_some());
    let encoded = encode_interpreter_program_v2(&registry).unwrap();
    let decoded = decode_interpreter_program_v2(&encoded).unwrap();
    assert!(decoded.get(function).unwrap().fusion_scaffold.is_some());

    let mut legacy = registry.clone();
    legacy.functions.get_mut(&function).unwrap().fusion_scaffold = None;
    let encoded = encode_interpreter_program_v2(&legacy).unwrap();
    let decoded = decode_interpreter_program_v2(&encoded).unwrap();
    assert!(decoded.get(function).unwrap().fusion_scaffold.is_none());
}

#[test]
fn program_v2_rejects_self_consistent_fake_two_signal_hint_in_one_statement() {
    let (mut registry, function) =
        compiled_function("function y=f()\npersistent p\ny=p.*2;\nend\n");
    let body = registry.functions.get_mut(&function).unwrap();
    assert!(body.fusion_scaffold.is_none());
    let selected = body
        .fusion_stage_projection
        .as_ref()
        .expect("selected statement PCs");
    let signal_rows = selected
        .statements
        .iter()
        .filter(|row| {
            (row.pc.start..row.pc.end).any(|pc| {
                crate::bytecode::compile::instr_fusion_signal_kind(&body.instructions[pc]).is_some()
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(signal_rows.len(), 1);
    let row = signal_rows[0];
    let signal_pc = (row.pc.start..row.pc.end)
        .find(|pc| {
            crate::bytecode::compile::instr_fusion_signal_kind(&body.instructions[*pc]).is_some()
        })
        .unwrap();
    let candidate = FusionCandidateGroup {
        id: 0,
        signal_count: 2,
        function,
        block: runmat_mir::BasicBlockId(row.pc.statement.block as usize),
        stmt_start: row.pc.statement.position as usize,
        stmt_end: row.pc.statement.position as usize + 2,
        source_span: body.instr_spans[signal_pc],
    };
    let mut windows = derive_semantic_fusion_instruction_windows(
        &body.instructions,
        &body.instr_spans,
        std::slice::from_ref(&candidate),
    );
    windows.sort_by_key(|window| (window.span.start, window.span.end));
    body.fusion_scaffold = Some(FunctionFusionScaffold {
        signal_count: 2,
        candidates: vec![candidate],
        windows,
        private_layout: None,
    });
    assert!(body.fusion_scaffold.as_ref().unwrap().valid_for(
        function,
        &body.instructions,
        &body.instr_spans
    ));
    assert!(encode_interpreter_program_v2(&registry).is_err());
    let forged = encode_payload(
        InterpreterPayloadForm::ProgramV2,
        b"runmat-interpreter-program-v2\n",
        &[
            ("bytecode", BYTECODE_SCHEMA_VERSION),
            ("function_registry", FUNCTION_REGISTRY_SCHEMA_VERSION),
        ],
        &registry,
    )
    .unwrap();
    assert!(decode_interpreter_program_v2(&forged).is_err());
}
