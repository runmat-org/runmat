use runmat_hir::{lower, LoweringContext};
use runmat_mir::{analysis::analyze_assembly, lowering::lower_assembly};

fn diagnostics(source: &str) -> Vec<runmat_mir::MirDiagnostic> {
    let ast = runmat_parser::parse(source).expect("operator fixture parses");
    let hir = lower(&ast, &LoweringContext::empty()).expect("operator fixture lowers to HIR");
    let mir = lower_assembly(&hir.assembly).expect("operator fixture lowers to MIR");
    analyze_assembly(&mir).diagnostics
}

#[test]
fn nested_transfer_function_arithmetic_uses_cataloged_operator_contracts() {
    let found = diagnostics(
        "s = tf('s'); \
         plant = 1 / (2*s + 3); \
         velocity = (4 + 5/s) * plant; \
         position = 6 * velocity / s;",
    );
    assert!(found.is_empty(), "{found:#?}");
}

#[test]
fn invalid_transfer_function_power_keeps_typed_method_diagnostic() {
    let found = diagnostics("s = tf('s'); bad = s ^ 0.5;");
    assert!(found
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-TF-METHOD-EXPONENT"));
    assert!(found
        .iter()
        .all(|diagnostic| diagnostic.code != "RM-TYPE-BINARY-OPERAND"));
}

#[test]
fn callable_selected_from_cell_uses_its_typed_output_contract() {
    let found = diagnostics("callbacks = {@() 37}; result = callbacks{1}(); assert(result == 37);");
    assert!(found.is_empty(), "{found:#?}");
}

#[test]
fn dynamic_callable_selection_does_not_become_a_numeric_index_fact() {
    let found =
        diagnostics("function y = invoke(callbacks); y = callbacks{1}(); assert(y == 37); end");
    assert!(found
        .iter()
        .all(|diagnostic| diagnostic.code != "RM-TYPE-BINARY-OPERAND"));
}
