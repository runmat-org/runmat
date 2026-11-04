use crate::{Bytecode, Instr};

fn compile_source(source: &str) -> Bytecode {
    compile_source_and_mir(source).0
}

fn compile_source_and_mir(
    source: &str,
) -> (Bytecode, runmat_mir::MirAssembly, runmat_hir::FunctionId) {
    let ast = runmat_parser::parse(source).expect("parse statement");
    let hir =
        runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("lower statement");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower MIR");
    let entrypoint = &hir.assembly.entrypoints[0];
    let bytecode = super::compile(&hir.assembly, &mir, entrypoint.id).expect("compile statement");
    (bytecode, mir, entrypoint.target)
}

#[test]
fn zero_output_call_statement_does_not_discard_a_nonexistent_value() {
    for source in [
        "disp(7);",
        "sin(0);",
        "noop(7); function noop(x); end",
        "external(7);",
        "f = @sin; f(0);",
        "args = {7}; disp(args{:});",
        "args = {7}; noop(args{:}); function noop(x); end",
        "args = {7}; external(args{:});",
    ] {
        let bytecode = compile_source(source);
        assert!(
            bytecode.instructions.iter().any(|instruction| matches!(
                instruction,
                Instr::CallBuiltinMulti(_, _, 0)
                    | Instr::CallSemanticFunctionMulti(_, _, 0)
                    | Instr::CallFevalMulti(_, 0)
                    | Instr::CallFevalExpandMultiOutput(_, 0)
                    | Instr::CallBuiltinExpandMultiOutput(_, _, 0)
                    | Instr::CallSemanticFunctionExpandMultiOutput(_, _, 0)
                    | Instr::CallFunctionExpandMultiOutput { out_count: 0, .. }
                    | Instr::CallWorkspaceFirstExpandMultiOutput { out_count: 0, .. }
                    | Instr::CallFunctionMulti { out_count: 0, .. }
                    | Instr::ReadSubscriptPath {
                        selection: runmat_types::SequenceUse::Discard,
                        context: runmat_types::ObjectIndexingContext::Statement,
                        to_sequence_register: false,
                        ..
                    }
            )),
            "expected a zero-output call: {source}; {:?}",
            bytecode.instructions
        );
        assert!(
            !bytecode
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instr::Pop)),
            "zero-output call has no value to discard: {source}; {:?}",
            bytecode.instructions
        );
    }
}

#[test]
fn value_producing_expression_statement_still_discards_its_value() {
    for source in ["7;", "sin(0)"] {
        let bytecode = compile_source(source);
        assert!(
            bytecode
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instr::Pop)),
            "value-producing statement must discard its result: {source}; {:?}",
            bytecode.instructions
        );
    }
}

#[test]
fn unsuppressed_expression_emits_at_its_own_mir_output_site() {
    let (bytecode, mir, function) = compile_source_and_mir("7");
    let ranges = bytecode
        .statement_pc_ranges()
        .expect("statement PC projection");
    let body = mir.bodies.get(&function).expect("selected MIR body");
    assert_eq!(
        ranges.len(),
        body.blocks
            .iter()
            .map(|block| block.statements.len())
            .sum::<usize>()
    );
    let mut outputs = 0;
    for range in &ranges {
        assert_eq!(range.statement.function.0 as usize, function.0);
        let block = body
            .blocks
            .iter()
            .find(|block| block.id.0 == range.statement.block as usize)
            .expect("owning MIR block");
        let statement = block
            .statements
            .get(range.statement.position as usize)
            .expect("owning MIR statement");
        let instructions = &bytecode.instructions[range.start..range.end];
        if matches!(
            statement.kind,
            runmat_mir::MirStmtKind::StatementOutput(runmat_mir::MirStatementOutput {
                source: runmat_mir::MirStatementOutputSource::Expression,
                ..
            })
        ) {
            outputs += 1;
            assert!(matches!(
                instructions,
                [Instr::EmitStackTop { .. }, Instr::Pop]
            ));
        } else {
            assert!(!instructions
                .iter()
                .any(|instruction| matches!(instruction, Instr::EmitStackTop { .. } | Instr::Pop)));
        }
    }
    assert_eq!(outputs, 1, "one expression-output site owns both effects");
}
