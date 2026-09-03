use lsp_types::Position;
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const CASES: [(&str, &str, &str); 8] = [
    (
        "bitand",
        "x = bitand(uint8(1), uint8(2));",
        "C = bitand(A, B)",
    ),
    ("bitor", "x = bitor(uint8(1), uint8(2));", "C = bitor(A, B)"),
    (
        "bitxor",
        "x = bitxor(uint8(1), uint8(2));",
        "C = bitxor(A, B)",
    ),
    ("bitcmp", "x = bitcmp(uint8(1));", "C = bitcmp(A)"),
    ("bitget", "x = bitget(uint8(1), 1);", "B = bitget(A, bit)"),
    ("bitset", "x = bitset(uint8(1), 1);", "C = bitset(A, bit)"),
    (
        "bitshift",
        "x = bitshift(uint8(1), 1);",
        "C = bitshift(A, k)",
    ),
    (
        "idivide",
        "x = idivide(uint8(1), uint8(1));",
        "C = idivide(A, B)",
    ),
];

#[test]
fn migrated_catalog_descriptors_drive_signature_help() {
    for (name, source, expected_label) in CASES {
        let entry = runmat_builtins::builtin_catalog_entry_by_name(name)
            .unwrap_or_else(|| panic!("{name} must have a canonical catalog entry"));
        assert!(
            entry
                .descriptor
                .signatures
                .iter()
                .any(|signature| signature.label == expected_label),
            "{name} must own its public signature in the canonical catalog"
        );
        assert!(
            !entry.integer_capabilities.is_empty(),
            "{name} must own its integer contract in the canonical catalog"
        );

        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        assert!(analysis.compile_error.is_none(), "{source}");

        let help = signature_help_at(source, &analysis, &Position::new(0, 4))
            .unwrap_or_else(|| panic!("{name} must expose descriptor-backed signature help"));
        assert!(
            help.signatures
                .iter()
                .any(|signature| signature.label == expected_label),
            "expected {expected_label} for {source}"
        );
    }
}
