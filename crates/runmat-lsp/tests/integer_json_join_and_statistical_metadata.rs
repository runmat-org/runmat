use lsp_types::{HoverContents, Position};
use runmat_builtins::{BuiltinCompletionPolicy, BuiltinIntegerAuditKind, BuiltinIntegerClass};
use runmat_lsp::core::analysis::{
    analyze_document_with_compat, completion_at, hover_at, signature_help_at, CompatMode,
};

const PUBLIC_BUILTINS: [&str; 8] = [
    "join",
    "jsonencode",
    "jsondecode",
    "kstest",
    "lasso",
    "lassoglm",
    "geometry.listRegions",
    "geometry.meshes",
];

const CAPABILITY_BUILTINS: [&str; 5] = ["join", "jsonencode", "kstest", "lasso", "lassoglm"];

const NOT_APPLICABLE_BUILTINS: [&str; 3] =
    ["geometry.listRegions", "geometry.meshes", "jsondecode"];

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

#[test]
fn json_join_and_statistical_packet_has_exact_public_names() {
    let source = "x = 1;";
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    let completions = completion_at(source, &analysis, &Position::new(0, 0));

    for name in PUBLIC_BUILTINS {
        let builtin = runmat_builtins::builtin_function_by_name(name)
            .unwrap_or_else(|| panic!("registered builtin {name}"));
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
fn capability_builtins_publish_all_eight_integer_classes() {
    for name in CAPABILITY_BUILTINS {
        let builtin = runmat_builtins::builtin_function_by_name(name)
            .unwrap_or_else(|| panic!("registered builtin {name}"));
        assert!(
            builtin.integer_audit.is_none(),
            "{name} is capability-bearing"
        );
        assert!(
            !builtin.integer_capabilities.is_empty(),
            "{name} integer capabilities"
        );
        for capability in builtin.integer_capabilities {
            assert!(!capability.inputs.is_empty(), "{name} {}", capability.form);
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
fn object_and_text_only_builtins_have_explicit_not_applicable_audits() {
    for name in NOT_APPLICABLE_BUILTINS {
        let builtin = runmat_builtins::builtin_function_by_name(name)
            .unwrap_or_else(|| panic!("registered builtin {name}"));
        assert!(builtin.integer_capabilities.is_empty(), "{name}");
        let audit = builtin
            .integer_audit
            .unwrap_or_else(|| panic!("{name} integer audit"));
        assert_eq!(audit.kind, BuiltinIntegerAuditKind::NotApplicable, "{name}");
        assert_eq!(audit.canonical_builtin, None, "{name}");
        assert!(!audit.notes.is_empty(), "{name} audit rationale");
    }
}

#[test]
fn calls_expose_unqualified_hover_and_qualified_signature_help() {
    for (lookup, source, expected_signature) in [
        ("join", "out = join(\"ab\");", "out = join(str)"),
        (
            "jsonencode",
            "text = jsonencode(uint8(1));",
            "jsonText = jsonencode(value)",
        ),
        (
            "jsondecode",
            "value = jsondecode(\"1\");",
            "value = jsondecode(text)",
        ),
        ("kstest", "h = kstest([0 1]);", "h = kstest(x)"),
        ("lasso", "b = lasso([1; 2], [1; 2]);", "B = lasso(X, y)"),
        (
            "lassoglm",
            "b = lassoglm([1; 2], [1; 2], 'normal');",
            "B = lassoglm(X, Y, distr)",
        ),
        (
            "listRegions",
            "asset = 1; regions = geometry.listRegions(asset);",
            "regions = geometry.listRegions(asset)",
        ),
        (
            "meshes",
            "asset = 1; out = geometry.meshes(asset);",
            "meshes = geometry.meshes(asset)",
        ),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(
            analysis.lowering_error.is_none(),
            "{source}: {:?}",
            analysis.lowering_error
        );

        let position = Position::new(0, source.find(lookup).expect("builtin call") as u32);
        let help = signature_help_at(source, &analysis, &position)
            .unwrap_or_else(|| panic!("descriptor-backed signature help for {lookup}"));
        assert!(
            help.signatures
                .iter()
                .any(|signature| signature.label == expected_signature),
            "expected {expected_signature} for {lookup}, got {:?}",
            help.signatures
                .iter()
                .map(|signature| signature.label.as_str())
                .collect::<Vec<_>>()
        );

        // Qualified namespace calls expose descriptor-backed signature help. Hover over the
        // terminal segment is not currently part of the LSP contract, matching other qualified
        // builtin metadata fixtures.
        if source.contains("geometry.") {
            continue;
        }

        let hover = hover_at(source, &analysis, &position)
            .unwrap_or_else(|| panic!("descriptor-backed hover for {lookup}"));
        let markdown = match hover.contents {
            HoverContents::Markup(markup) => markup.value,
            other => panic!("expected Markdown hover for {lookup}, got {other:?}"),
        };
        assert!(
            markdown.contains(expected_signature),
            "expected descriptor signature in {lookup} hover, got:\n{markdown}"
        );
    }
}
