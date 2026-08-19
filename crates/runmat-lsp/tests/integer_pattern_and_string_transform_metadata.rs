use lsp_types::Position;
use runmat_builtins::{
    BuiltinIntegerAuditKind, BuiltinIntegerInputAvailability, BuiltinIntegerOverflowRule,
};
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const CAPABILITY_NAMES: [&str; 6] = [
    "lettersPattern",
    "wildcardPattern",
    "pad",
    "startsWith",
    "strfind",
    "split",
];

const INAPPLICABLE_NAMES: [&str; 6] = [
    "pattern",
    "regexpPattern",
    "matches",
    "lower",
    "upper",
    "reverse",
];

#[test]
fn pattern_and_string_transform_integer_metadata_is_explicit() {
    for name in CAPABILITY_NAMES {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(!builtin.integer_capabilities.is_empty(), "{name}");
        assert!(builtin.integer_audit.is_none(), "{name}");
        for capability in builtin.integer_capabilities {
            for input in capability.inputs {
                assert_eq!(input.classes.len(), 8, "{name}: {}", input.name);
            }
        }
    }
    for name in INAPPLICABLE_NAMES {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(builtin.integer_capabilities.is_empty(), "{name}");
        assert_eq!(
            builtin.integer_audit.expect("integer audit").kind,
            BuiltinIntegerAuditKind::NotApplicable,
            "{name}"
        );
    }
}

#[test]
fn documented_and_evidence_open_string_controls_are_distinguished() {
    for name in ["lettersPattern", "wildcardPattern", "pad"] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        for capability in builtin.integer_capabilities {
            assert_eq!(
                capability.overflow,
                BuiltinIntegerOverflowRule::Error,
                "{name}"
            );
            for input in capability.inputs {
                assert_eq!(
                    input.availability,
                    BuiltinIntegerInputAvailability::Documented,
                    "{name}: {}",
                    input.name
                );
            }
        }
    }
    for name in ["strfind", "split"] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(builtin.integer_capabilities.iter().all(|capability| {
            capability.overflow == BuiltinIntegerOverflowRule::EvidenceOpen
                && capability.notes.contains("[integer-audit-open]")
                && capability.inputs.iter().all(|input| {
                    input.availability == BuiltinIntegerInputAvailability::RunMatOnly
                        && input.notes.contains("[integer-audit-open]")
                })
        }));
    }
}

#[test]
fn pattern_and_string_transform_signatures_are_visible_to_lsp() {
    for (name, source) in [
        ("lettersPattern", "p = lettersPattern(uint16(2));"),
        ("pattern", "p = pattern('literal');"),
        ("regexpPattern", "p = regexpPattern('[A-Z]+');"),
        (
            "wildcardPattern",
            "p = wildcardPattern(uint16(0),uint16(3));",
        ),
        ("matches", "tf = matches('abc',lettersPattern);"),
        (
            "startsWith",
            "tf = startsWith('RunMat','run','IgnoreCase',uint8(1));",
        ),
        (
            "strfind",
            "k = strfind('mission','s','ForceCellOutput',uint8(1));",
        ),
        ("lower", "s = lower('ABC');"),
        ("upper", "s = upper('abc');"),
        ("pad", "s = pad('x',uint16(3));"),
        ("reverse", "s = reverse('abc');"),
        ("split", "s = split('Mary Butler',' ',uint8(1));"),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}
