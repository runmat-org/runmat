use lsp_types::{HoverContents, Position};
use runmat_lsp::core::analysis::{
    analyze_document_with_compat, completion_at, hover_at, signature_help_at, CompatMode,
};

const EXTRACTAFTER_TO_FEA_BOUNDARY_BUILTINS: [&str; 10] = [
    "extractAfter",
    "extractBefore",
    "extractBetween",
    "extractFileText",
    "extractHTMLText",
    "eye",
    "false",
    "fclose",
    "fcontour",
    "fea.boundaryCondition",
];

#[test]
fn extractafter_to_fea_boundary_descriptors_are_visible_to_signature_help_and_unqualified_hover() {
    for (name, source, expected_signature) in [
        (
            "extractAfter",
            "out = extractAfter(\"abc\", 1);",
            "s = extractAfter(text, boundary)",
        ),
        (
            "extractBefore",
            "out = extractBefore(\"abc\", 2);",
            "s = extractBefore(text, boundary)",
        ),
        (
            "extractBetween",
            "out = extractBetween(\"abc\", 1, 2);",
            "newText = extractBetween(str, start, end)",
        ),
        (
            "extractFileText",
            "out = extractFileText(\"notes.txt\");",
            "str = extractFileText(filename)",
        ),
        (
            "extractHTMLText",
            "out = extractHTMLText(\"<p>x</p>\");",
            "str = extractHTMLText(code)",
        ),
        ("eye", "out = eye(2);", "A = eye(n)"),
        ("false", "out = false(2);", "L = false(n)"),
        ("fclose", "status = fclose(3);", "status = fclose(fid)"),
        ("fcontour", "h = fcontour(@(x,y) x+y);", "h = fcontour(f)"),
        (
            "fea.boundaryCondition",
            "bc = fea.boundaryCondition(\"bc\", \"face\", \"fixed\");",
            "bc = fea.boundaryCondition(id, region, kind, Name, Value, ...)",
        ),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        assert!(analysis.compile_error.is_none(), "{source}");

        let lookup = name.rsplit('.').next().expect("builtin name segment");
        let column = source.find(lookup).expect("builtin call") as u32;
        let position = Position::new(0, column);
        let help = signature_help_at(source, &analysis, &position)
            .unwrap_or_else(|| panic!("descriptor-backed signature help for {name}"));
        assert!(
            help.signatures
                .iter()
                .any(|signature| signature.label == expected_signature),
            "expected {expected_signature} for {name}"
        );

        if name.contains('.') {
            continue;
        }

        let hover = hover_at(source, &analysis, &position)
            .unwrap_or_else(|| panic!("descriptor-backed hover for {name}"));
        let markdown = match hover.contents {
            HoverContents::Markup(markup) => markup.value,
            other => panic!("expected Markdown hover for {name}, got {other:?}"),
        };
        assert!(
            markdown.contains(expected_signature),
            "{name} hover: {markdown}"
        );
    }
}

#[test]
fn integer_builtins_are_public_completions_with_settled_metadata() {
    let source = "x = 1;";
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    let completions = completion_at(source, &analysis, &Position::new(0, 0));

    for name in EXTRACTAFTER_TO_FEA_BOUNDARY_BUILTINS {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        let descriptor = builtin
            .descriptor
            .unwrap_or_else(|| panic!("{name} must expose a descriptor"));
        assert_eq!(
            descriptor.completion_policy,
            runmat_builtins::BuiltinCompletionPolicy::Public,
            "{name} completion policy"
        );
        assert!(
            !builtin.integer_capabilities.is_empty() || builtin.integer_audit.is_some(),
            "{name} must expose a settled integer disposition"
        );
        assert!(
            completions
                .iter()
                .any(|item| item.label.eq_ignore_ascii_case(name)),
            "public completion for {name}"
        );
    }
}

#[test]
fn runmat_only_integer_extensions_are_visible_in_builtin_metadata() {
    for (name, extension_ids) in [
        ("extractAfter", &["extractafter-resident-position"][..]),
        ("extractBefore", &["extractbefore-resident-position"][..]),
        (
            "extractBetween",
            &[
                "extractbetween-full-broadcast",
                "extractbetween-resident-position",
            ][..],
        ),
        ("extractFileText", &["extractfiletext-resident-pages"][..]),
        (
            "extractHTMLText",
            &["extracthtmltext-char-matrix", "extracthtmltext-broad-cell"][..],
        ),
        ("eye", &["eye-implicit-prototype", "eye-nd-dimensions"][..]),
        (
            "false",
            &[
                "false-implicit-prototype",
                "false-resident-size-input",
                "false-single-size-input",
            ][..],
        ),
        (
            "fclose",
            &["fclose-integer-fileid", "fclose-resident-fileid"][..],
        ),
        (
            "fcontour",
            &[
                "fcontour-integer-line-color",
                "fcontour-positional-level-spec",
                "fcontour-resident-numeric-input",
            ][..],
        ),
    ] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        for extension_id in extension_ids {
            assert!(
                builtin.extensions.iter().any(|extension| {
                    extension.id == *extension_id
                        && extension.mode == runmat_builtins::BuiltinExtensionMode::RunMatOnly
                }),
                "{name} must expose {extension_id}"
            );
        }
    }

    let boundary = runmat_builtins::builtin_function_by_name("fea.boundaryCondition")
        .expect("registered boundary constructor");
    assert!(boundary.extensions.is_empty());
}
