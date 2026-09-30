use crate::{Bytecode, Instr};

fn compile_source(source: &str) -> Bytecode {
    let ast = runmat_parser::parse(source).expect("parse statement");
    let hir =
        runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("lower statement");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower MIR");
    super::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id).expect("compile statement")
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
