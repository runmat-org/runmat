use lsp_types::Position;
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const PACKET: [(&str, usize, &str); 9] = [
    ("pulstran", 3, "y=pulstran([0 1],0,'rectpuls');"),
    ("rectpuls", 2, "y=rectpuls([0 1]);"),
    ("rand", 3, "y=rand(uint8(2));"),
    ("randi", 4, "y=randi(3,uint8(2));"),
    ("randn", 2, "y=randn(uint8(2));"),
    ("random", 2, "y=random('Normal',0,1);"),
    ("randperm", 1, "y=randperm(uint8(2));"),
    ("randsample", 3, "y=randsample(3,2);"),
    ("rng", 3, "rng(uint32(1));"),
];

#[test]
fn random_and_waveform_integer_metadata_is_public_and_precise() {
    for (name, expected_forms, _) in PACKET {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert_eq!(builtin.integer_capabilities.len(), expected_forms, "{name}");
        assert!(builtin.integer_audit.is_none(), "{name}");
        assert!(builtin
            .integer_capabilities
            .iter()
            .all(|capability| !capability.inputs.is_empty()));
    }
}

#[test]
fn random_and_waveform_signatures_remain_visible_to_lsp() {
    for (name, _, source) in PACKET {
        let analysis = analyze_document_with_compat(source, CompatMode::Matlab);
        assert!(analysis.syntax_error.is_none(), "{name}: {source}");
        assert!(
            analysis.lowering_error.is_none(),
            "{name}: {source}: {:?}",
            analysis.lowering_error
        );
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}

#[test]
fn new_runmat_only_forms_publish_stable_compatibility_identifiers() {
    let expected = [
        ("pulstran", 5),
        ("rectpuls", 3),
        ("randn", 2),
        ("random", 2),
        ("randsample", 5),
        ("rng", 2),
    ];
    let mut ids = std::collections::HashSet::new();
    for (name, count) in expected {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert_eq!(builtin.extensions.len(), count, "{name}");
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
            assert!(
                ids.insert(extension.id),
                "duplicate extension {}",
                extension.id
            );
        }
    }
    assert_eq!(ids.len(), 19);
}
