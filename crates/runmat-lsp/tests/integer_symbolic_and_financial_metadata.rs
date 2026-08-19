use lsp_types::Position;
use runmat_builtins::{BuiltinIntegerInputAvailability, BuiltinIntegerOverflowRule};
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const PACKET: [(&str, usize, &str); 2] = [
    ("limit", 2, "syms x; y=limit(x,x,uint64(2));"),
    ("macd", 2, "y=macd(uint16([2 1 1 1;3 2 2 2]));"),
];

#[test]
fn symbolic_and_financial_integer_metadata_is_explicit() {
    for (name, expected_forms, _) in PACKET {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert_eq!(builtin.integer_capabilities.len(), expected_forms, "{name}");
        assert!(builtin.integer_audit.is_none(), "{name}");
        assert!(builtin
            .integer_capabilities
            .iter()
            .flat_map(|capability| capability.inputs)
            .all(|input| input.classes.len() == 8));
    }
    let limit = runmat_builtins::builtin_function_by_name("limit").expect("limit");
    assert_eq!(
        limit.integer_capabilities[0].inputs[0].availability,
        BuiltinIntegerInputAvailability::RunMatOnly
    );
    assert_eq!(
        limit.integer_capabilities[1].inputs[0].availability,
        BuiltinIntegerInputAvailability::Documented
    );
    assert!(limit.integer_capabilities[1]
        .notes
        .contains("[integer-audit-open]"));
    let macd = runmat_builtins::builtin_function_by_name("macd").expect("macd");
    assert!(macd
        .integer_capabilities
        .iter()
        .all(|capability| capability.overflow == BuiltinIntegerOverflowRule::Error));
}

#[test]
fn symbolic_and_financial_signatures_remain_visible_to_lsp() {
    for (name, _, source) in PACKET {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{name}: {source}");
        assert!(analysis.lowering_error.is_none(), "{name}: {source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}
