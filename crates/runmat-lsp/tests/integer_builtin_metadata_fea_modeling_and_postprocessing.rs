use lsp_types::{Position, SignatureHelp};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinIntegerAuditKind, BuiltinIntegerBackendRule,
    BuiltinIntegerClass,
};
use runmat_lsp::core::analysis::{
    analyze_document_with_compat, completion_at, signature_help_at, CompatMode,
};

const FEA_MODELING_AND_POSTPROCESSING_BUILTINS: [&str; 10] = [
    "fea.compare",
    "fea.domain",
    "fea.field",
    "fea.interface",
    "fea.loadCase",
    "fea.material",
    "fea.materialAssignment",
    "fea.model",
    "fea.plan",
    "fea.plot",
];

const FEA_INTEGER_CONSTRUCTORS: [&str; 4] = [
    "fea.domain",
    "fea.interface",
    "fea.loadCase",
    "fea.material",
];

const FEA_INTEGER_INAPPLICABLE: [&str; 6] = [
    "fea.compare",
    "fea.field",
    "fea.materialAssignment",
    "fea.model",
    "fea.plan",
    "fea.plot",
];

const INTEGER_CLASSES: [BuiltinIntegerClass; 8] = [
    BuiltinIntegerClass::Int8,
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Int32,
    BuiltinIntegerClass::Int64,
    BuiltinIntegerClass::Uint8,
    BuiltinIntegerClass::Uint16,
    BuiltinIntegerClass::Uint32,
    BuiltinIntegerClass::Uint64,
];

fn signature_help(source: &str, lookup: &str) -> SignatureHelp {
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    assert!(analysis.syntax_error.is_none(), "{source}");
    assert!(analysis.lowering_error.is_none(), "{source}");
    assert!(analysis.compile_error.is_none(), "{source}");
    let column = source.find(lookup).expect("builtin call") as u32;
    signature_help_at(source, &analysis, &Position::new(0, column))
        .unwrap_or_else(|| panic!("descriptor-backed signature help for {lookup}"))
}

#[test]
fn fea_modeling_and_postprocessing_descriptors_are_visible_to_signature_help() {
    for (lookup, source) in [
        ("compare", "out = fea.compare(\"base\", \"candidate\");"),
        ("domain", "out = fea.domain(\"electromagnetic\");"),
        ("field", "out = fea.field(\"run\", \"stress\");"),
        (
            "interface",
            "out = fea.interface(\"contact\", \"left\", \"right\");",
        ),
        (
            "loadCase",
            "out = fea.loadCase(\"load\", \"face\", \"pressure\", \"MagnitudePa\", 1);",
        ),
        (
            "material",
            "out = fea.material(\"steel\", \"YoungsModulusPa\", 200, \"PoissonRatio\", 0);",
        ),
        (
            "materialAssignment",
            "out = fea.materialAssignment(\"region\", \"steel\");",
        ),
        ("model", "out = fea.model(\"model\", \"geometry\");"),
        ("plan", "out = fea.plan(\"study.fea\");"),
        ("plot", "out = fea.plot(\"run\", \"stress\");"),
    ] {
        let help = signature_help(source, lookup);
        assert!(!help.signatures.is_empty(), "{lookup} signatures");
        assert!(
            help.signatures
                .iter()
                .any(|signature| signature.label.contains(&format!("fea.{lookup}"))),
            "{lookup} must expose its namespaced signature"
        );
    }
}

#[test]
fn fea_packet_is_public_and_has_settled_integer_metadata() {
    let source = "x = 1;";
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    let completions = completion_at(source, &analysis, &Position::new(0, 0));

    for name in FEA_MODELING_AND_POSTPROCESSING_BUILTINS {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        let descriptor = builtin
            .descriptor
            .unwrap_or_else(|| panic!("{name} descriptor"));
        assert_eq!(
            descriptor.completion_policy,
            BuiltinCompletionPolicy::Public,
            "{name} completion policy"
        );
        assert!(
            !descriptor.signatures.is_empty(),
            "{name} reference-visible signatures"
        );
        assert!(builtin.extensions.is_empty(), "{name} is RunMat-native");
        assert!(
            completions
                .iter()
                .any(|item| item.label.eq_ignore_ascii_case(name)),
            "public completion for {name}"
        );
    }
}

#[test]
fn fea_numeric_constructor_forms_cover_every_integer_class_on_the_host() {
    for name in FEA_INTEGER_CONSTRUCTORS {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(builtin.integer_audit.is_none(), "{name}");
        assert!(
            !builtin.integer_capabilities.is_empty(),
            "{name} integer forms"
        );
        for capability in builtin.integer_capabilities {
            assert_eq!(
                capability.backend,
                BuiltinIntegerBackendRule::HostOnly,
                "{name} {}",
                capability.form
            );
            for input in capability.inputs {
                assert_eq!(
                    input.classes, &INTEGER_CLASSES,
                    "{name} {} {}",
                    capability.form, input.name
                );
            }
        }
    }
}

#[test]
fn fea_non_numeric_roles_have_explicit_not_applicable_audits() {
    for name in FEA_INTEGER_INAPPLICABLE {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(builtin.integer_capabilities.is_empty(), "{name}");
        let audit = builtin
            .integer_audit
            .unwrap_or_else(|| panic!("{name} integer audit"));
        assert_eq!(audit.kind, BuiltinIntegerAuditKind::NotApplicable, "{name}");
        assert_eq!(audit.canonical_builtin, None, "{name}");
        assert!(!audit.notes.is_empty(), "{name} audit rationale");
    }
}
