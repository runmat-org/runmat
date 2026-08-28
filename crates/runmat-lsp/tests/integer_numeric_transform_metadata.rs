use lsp_types::Position;
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const PACKET: [(&str, usize, &str); 8] = [
    ("pchip", 3, "v=pchip(uint8([0 1]),uint8([0 1]),uint8(1));"),
    ("pdf", 2, "p=pdf('Normal',uint8(0),uint8(0),uint8(1));"),
    ("pdist", 2, "d=pdist(uint8([0;1]));"),
    ("pdist2", 3, "d=pdist2(uint8([0;1]),uint8([1;2]));"),
    ("peaks", 2, "[x,y,z]=peaks(uint8([0 1]),uint8([1 2]));"),
    ("periodogram", 2, "p=periodogram(uint8([0;1;0;1]));"),
    ("perms", 1, "p=perms(uint8([1 2 3]));"),
    ("permute", 2, "b=permute(uint8([1 2;3 4]),uint8([2 1]));"),
];

#[test]
fn numeric_transform_metadata_is_class_complete() {
    for (name, expected_forms, _) in PACKET {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert_eq!(builtin.integer_capabilities.len(), expected_forms, "{name}");
        assert!(builtin.integer_audit.is_none(), "{name}");
        for capability in builtin.integer_capabilities {
            assert!(!capability.inputs.is_empty(), "{name}: {}", capability.form);
            for input in capability.inputs {
                assert_eq!(input.classes.len(), 8, "{name}: {}", input.name);
            }
        }
    }
}

#[test]
fn numeric_transform_signatures_remain_visible_to_lsp() {
    for (name, _, source) in PACKET {
        let analysis = analyze_document_with_compat(source, CompatMode::Matlab);
        assert!(analysis.syntax_error.is_none(), "{name}: {source}");
        assert!(analysis.lowering_error.is_none(), "{name}: {source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}

#[test]
fn runmat_only_numeric_transform_forms_publish_compatibility_identifiers() {
    for name in ["pchip", "pdf", "pdist", "pdist2", "peaks", "periodogram"] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(!builtin.extensions.is_empty(), "{name}");
        for extension in builtin.extensions {
            assert_eq!(
                extension.mode,
                runmat_builtins::BuiltinExtensionMode::RunMatOnly,
                "{name}: {}",
                extension.id
            );
            assert!(
                extension.error_identifier.is_some(),
                "{name}: {}",
                extension.id
            );
        }
    }
}
