use lsp_types::Position;
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const PACKET: [(&str, usize, &str); 9] = [
    ("num2str", 2, "s=num2str(uint64(1));"),
    (
        "onehotdecode",
        4,
        "x=onehotdecode([1;0],uint64([1 2]),1,'uint64');",
    ),
    (
        "onehotencode",
        3,
        "x=onehotencode(uint64([1 2]),1,'ClassNames',uint64([1 2]));",
    ),
    (
        "ordinal",
        3,
        "x=ordinal(uint64([1 2]),{'a','b'},uint64([1 2]));",
    ),
    (
        "removeLongWords",
        1,
        "d=tokenizedDocument('a'); x=removeLongWords(d,uint8(8));",
    ),
    (
        "removeShortWords",
        1,
        "d=tokenizedDocument('a'); x=removeShortWords(d,uint8(2));",
    ),
    (
        "removeWords",
        1,
        "d=tokenizedDocument('a'); x=removeWords(d,uint8(1));",
    ),
    ("regexprep", 2, "x=regexprep('a a','a','x',uint8(2));"),
    (
        "replaceBetween",
        2,
        "x=replaceBetween('abcd',uint8(2),uint8(3),'x');",
    ),
];

#[test]
fn label_and_text_control_metadata_is_class_complete() {
    for (name, expected_forms, _) in PACKET {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert_eq!(builtin.integer_capabilities.len(), expected_forms, "{name}");
        assert!(builtin.integer_audit.is_none(), "{name}");
        for capability in builtin.integer_capabilities {
            assert!(
                !capability.inputs.is_empty() || capability.form.contains("integer_typename"),
                "{name}: {}",
                capability.form
            );
            for input in capability.inputs {
                assert!(
                    input.classes.len() == 8
                        || input.availability
                            == runmat_builtins::BuiltinIntegerInputAvailability::Rejected,
                    "{name}: {}",
                    input.name
                );
            }
        }
    }
}

#[test]
fn label_and_text_control_signatures_remain_visible_to_lsp() {
    for (name, _, source) in PACKET {
        let analysis = analyze_document_with_compat(source, CompatMode::Matlab);
        assert!(analysis.syntax_error.is_none(), "{name}: {source}");
        assert!(analysis.lowering_error.is_none(), "{name}: {source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}
