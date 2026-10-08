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
