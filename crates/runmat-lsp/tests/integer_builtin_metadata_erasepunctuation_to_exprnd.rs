use lsp_types::{HoverContents, Position};
use runmat_lsp::core::analysis::{
    analyze_document_with_compat, completion_at, hover_at, signature_help_at, CompatMode,
};

const ERASEPUNCTUATION_TO_EXPRND_BUILTINS: [&str; 10] = [
    "erasePunctuation",
    "eraseURLs",
    "erf",
    "erfcinv",
    "error",
    "errorbar",
    "exist",
    "exp",
    "expm1",
    "exprnd",
];

#[test]
fn erasepunctuation_to_exprnd_descriptors_are_visible_to_signature_help_and_hover() {
    for (name, source, expected_signature) in [
        (
            "erasePunctuation",
            "out = erasePunctuation('run.mat');",
            "newStr = erasePunctuation(str)",
        ),
        (
            "eraseURLs",
            "out = eraseURLs('https://runmat.com');",
            "newStr = eraseURLs(str)",
        ),
        ("erf", "out = erf(1);", "Y = erf(X)"),
        ("erfcinv", "out = erfcinv(1);", "Y = erfcinv(X)"),
        ("error", "error('message');", "error(msg)"),
        (
            "errorbar",
            "h = errorbar([1 2], [1 1]);",
            "h = errorbar(Y, E)",
        ),
        ("exist", "kind = exist('sin');", "typeID = exist(name)"),
        ("exp", "out = exp(1);", "Y = exp(X)"),
        ("expm1", "out = expm1(1);", "Y = expm1(X)"),
        ("exprnd", "out = exprnd(1);", "r = exprnd(mu)"),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        assert!(analysis.compile_error.is_none(), "{source}");

        let column = source.find(name).expect("builtin call") as u32;
        let position = Position::new(0, column);
        let help = signature_help_at(source, &analysis, &position)
            .unwrap_or_else(|| panic!("descriptor-backed signature help for {name}"));
        assert!(
            help.signatures
                .iter()
                .any(|signature| signature.label == expected_signature),
            "expected {expected_signature} for {name}"
        );

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

    for name in ERASEPUNCTUATION_TO_EXPRND_BUILTINS {
        let catalog = runmat_builtins::builtin_catalog_entry_by_name(name);
        let runtime = runmat_builtins::builtin_function_by_name(name);
        let descriptor = catalog
            .map(|entry| entry.descriptor)
            .or_else(|| runtime.and_then(|builtin| builtin.descriptor))
            .unwrap_or_else(|| panic!("{name} must expose a descriptor"));
        assert_eq!(
            descriptor.completion_policy,
            runmat_builtins::BuiltinCompletionPolicy::Public,
            "{name} completion policy"
        );
        let integer_is_settled = catalog.is_some_and(|entry| {
            !entry.integer_capabilities.is_empty() || entry.integer_audit.is_some()
        }) || runtime.is_some_and(|builtin| {
            !builtin.integer_capabilities.is_empty() || builtin.integer_audit.is_some()
        });
        assert!(
            integer_is_settled,
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
fn matlab_mode_keeps_integer_extension_metadata_visible() {
    for (name, extension_id) in [
        ("error", "error-unqualified-identifier"),
        ("erasePunctuation", "erase-punctuation-char-matrix-input"),
        ("erasePunctuation", "erase-punctuation-broad-cell-input"),
        ("eraseURLs", "erase-urls-char-matrix-input"),
        ("eraseURLs", "erase-urls-broad-cell-input"),
        ("exist", "exist-search-type-extension"),
        ("exp", "exp-integer-input"),
        ("expm1", "expm1-integer-input"),
        ("exprnd", "exprnd-integer-mean"),
    ] {
        let catalog = runmat_builtins::builtin_catalog_entry_by_name(name);
        let runtime = runmat_builtins::builtin_function_by_name(name);
        let extensions = catalog
            .map(|entry| entry.extensions)
            .or_else(|| runtime.map(|builtin| builtin.extensions))
            .unwrap_or_else(|| panic!("registered builtin {name}"));
        assert!(
            extensions.iter().any(|extension| {
                extension.id == extension_id
                    && extension.mode == runmat_builtins::BuiltinExtensionMode::RunMatOnly
            }),
            "{name} must expose {extension_id}"
        );
    }

    for name in ["erf", "erfcinv", "errorbar"] {
        let catalog = runmat_builtins::builtin_catalog_entry_by_name(name);
        let runtime = runmat_builtins::builtin_function_by_name(name);
        let extensions = catalog
            .map(|entry| entry.extensions)
            .or_else(|| runtime.map(|builtin| builtin.extensions))
            .unwrap_or_else(|| panic!("registered builtin {name}"));
        assert!(extensions.is_empty(), "{name} has no extension forms");
    }
}
