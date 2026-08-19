use lsp_types::Position;
use runmat_builtins::{BuiltinIntegerAuditKind, BuiltinIntegerInputAvailability};
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const CAPABILITY_PACKET: [(&str, usize, &str); 8] = [
    ("stackedplot", 2, "h=stackedplot(uint8([1 2]));"),
    ("stairs", 1, "h=stairs(uint8([1 2]));"),
    (
        "standardizeMissing",
        3,
        "y=standardizeMissing([1 2],uint8(1));",
    ),
    ("statget", 1, "y=statget(statset(),'TolX',uint8(1));"),
    ("statset", 1, "o=statset('MaxIter',uint8(1));"),
    ("stem", 1, "h=stem(uint8([1 2]));"),
    ("step", 1, "y=step(tf(1,[1 1]),uint8(2));"),
    ("stepinfo", 2, "i=stepinfo(uint8([0 1]));"),
];

#[test]
fn plotting_missing_statistics_and_control_integer_metadata_is_explicit() {
    for (name, expected_forms, _) in CAPABILITY_PACKET {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert_eq!(builtin.integer_capabilities.len(), expected_forms, "{name}");
        assert!(builtin.integer_audit.is_none(), "{name}");
        assert!(builtin
            .integer_capabilities
            .iter()
            .flat_map(|capability| capability.inputs)
            .all(|input| input.classes.len() == 8));
    }

    for name in [
        "stackedplot",
        "stairs",
        "standardizeMissing",
        "statget",
        "stem",
    ] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(builtin
            .integer_capabilities
            .iter()
            .flat_map(|capability| capability.inputs)
            .any(|input| input.availability == BuiltinIntegerInputAvailability::Documented));
    }
    for name in ["standardizeMissing", "statset", "step", "stepinfo"] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(builtin
            .integer_capabilities
            .iter()
            .flat_map(|capability| capability.inputs)
            .any(|input| input.availability == BuiltinIntegerInputAvailability::RunMatOnly));
    }

    let stats = runmat_builtins::builtin_function_by_name("stats").expect("registered stats");
    assert!(stats.integer_capabilities.is_empty());
    assert_eq!(
        stats.integer_audit.map(|audit| audit.kind),
        Some(BuiltinIntegerAuditKind::NotApplicable)
    );
}

#[test]
fn plotting_missing_statistics_and_control_signatures_remain_visible_to_lsp() {
    for (name, _, source) in CAPABILITY_PACKET {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{name}: {source}");
        assert!(analysis.lowering_error.is_none(), "{name}: {source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }

    let source = "s=stats(memoize(@sqrt));";
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    let position = Position::new(0, source.find("stats").unwrap() as u32);
    assert!(signature_help_at(source, &analysis, &position).is_some());
}
