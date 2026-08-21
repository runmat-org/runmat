use lsp_types::{Position, SignatureHelp};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinIntegerAuditKind, BuiltinIntegerBackendRule,
    BuiltinIntegerClass,
};
use runmat_lsp::core::analysis::{
    analyze_document_with_compat, completion_at, signature_help_at, CompatMode,
};

const PACKET: [&str; 10] = [
    "feedback",
    "fea.results",
    "fea.run",
    "fea.runOptions",
    "fea.step",
    "fea.study",
    "fea.sweep",
    "fea.trends",
    "fea.validate",
    "feof",
];

const FEA_INTEGER_INAPPLICABLE: [&str; 5] = [
    "fea.run",
    "fea.step",
    "fea.study",
    "fea.sweep",
    "fea.validate",
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
fn fea_execution_and_feedback_file_status_signatures_are_exact_and_visible() {
    for (lookup, source, expected) in [
        (
            "feedback",
            "out = feedback(tf([1], [1 1]), 1);",
            "sys = feedback(sys1, sys2)",
        ),
        (
            "results",
            "out = fea.results(\"run\");",
            "results = fea.results(runOrRunId, Name, Value, ...)",
        ),
        (
            "run",
            "out = fea.run(\"study.fea\");",
            "run = fea.run(studyOrSweepOrPath)",
        ),
        (
            "runOptions",
            "out = fea.runOptions(\"modal\");",
            "options = fea.runOptions(solver, Name, Value, ...)",
        ),
        (
            "step",
            "out = fea.step(\"s\", \"modal\");",
            "step = fea.step(id, kind)",
        ),
        (
            "study",
            "out = fea.study(\"study.fea\");",
            "study = fea.study(path)",
        ),
        (
            "sweep",
            "out = fea.sweep(\"s\", {});",
            "sweep = fea.sweep(id, studies, Name, Value, ...)",
        ),
        (
            "trends",
            "out = fea.trends();",
            "trends = fea.trends(Name, Value, ...)",
        ),
        (
            "validate",
            "out = fea.validate(\"study.fea\");",
            "result = fea.validate(studyOrSweepOrPath)",
        ),
        ("feof", "out = feof(1);", "tf = feof(fid)"),
    ] {
        let help = signature_help(source, lookup);
        assert!(
            help.signatures
                .iter()
                .any(|signature| signature.label == expected),
            "{lookup} exact signature"
        );
    }
}

#[test]
fn packet_is_public_complete_and_visible_to_completion() {
    let source = "x = 1;";
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    let completions = completion_at(source, &analysis, &Position::new(0, 0));
    for name in PACKET {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        let descriptor = builtin
            .descriptor
            .unwrap_or_else(|| panic!("{name} descriptor"));
        assert_eq!(
            descriptor.completion_policy,
            BuiltinCompletionPolicy::Public,
            "{name}"
        );
        assert!(!descriptor.signatures.is_empty(), "{name} signatures");
        assert!(
            completions
                .iter()
                .any(|item| item.label.eq_ignore_ascii_case(name)),
            "public completion for {name}"
        );
    }
}

#[test]
fn fea_capability_metadata_matches_exact_and_binary64_contracts() {
    for (name, expected_forms) in [
        ("fea.results", 3usize),
        ("fea.runOptions", 18usize),
        ("fea.trends", 1usize),
    ] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(
            builtin.integer_audit.is_none(),
            "{name} is capability-bearing"
        );
        assert_eq!(
            builtin.integer_capabilities.len(),
            expected_forms,
            "{name} forms"
        );
        assert!(builtin.extensions.is_empty(), "{name} is RunMat-native");
        for capability in builtin.integer_capabilities {
            assert_eq!(
                capability.backend,
                BuiltinIntegerBackendRule::HostOnly,
                "{name} {}",
                capability.form
            );
            assert!(
                !capability.inputs.is_empty(),
                "{name} {} inputs",
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
fn fea_direct_text_and_object_roles_are_explicitly_integer_inapplicable() {
    for name in FEA_INTEGER_INAPPLICABLE {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(builtin.integer_capabilities.is_empty(), "{name}");
        assert!(builtin.extensions.is_empty(), "{name} is RunMat-native");
        let audit = builtin
            .integer_audit
            .unwrap_or_else(|| panic!("{name} integer audit"));
        assert_eq!(audit.kind, BuiltinIntegerAuditKind::NotApplicable, "{name}");
        assert_eq!(audit.canonical_builtin, None, "{name}");
        assert!(!audit.notes.is_empty(), "{name} rationale");
    }
}

#[test]
fn feedback_and_feof_integer_extensions_use_typed_gather_fallback() {
    for (name, expected_forms) in [("feedback", 2usize), ("feof", 1usize)] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(builtin.integer_audit.is_none(), "{name}");
        assert_eq!(
            builtin.integer_capabilities.len(),
            expected_forms,
            "{name} forms"
        );
        assert!(
            !builtin.extensions.is_empty(),
            "{name} independently gates extensions"
        );
        for capability in builtin.integer_capabilities {
            assert_eq!(
                capability.backend,
                BuiltinIntegerBackendRule::GatherFallback,
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
