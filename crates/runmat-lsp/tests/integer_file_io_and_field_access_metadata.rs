use lsp_types::{HoverContents, Position};
use runmat_lsp::core::analysis::{
    analyze_document_with_compat, completion_at, hover_at, signature_help_at, CompatMode,
};

const BUILTINS: [&str; 10] = [
    "fopen",
    "fprintf",
    "fread",
    "frewind",
    "fwrite",
    "full",
    "func2str",
    "functions",
    "getenv",
    "getfield",
];

#[test]
fn file_io_and_field_access_descriptors_are_public_and_visible() {
    for (name, source, expected_signature) in [
        ("fopen", "fid = fopen('x');", "fid = fopen(filename)"),
        (
            "fprintf",
            "n = fprintf('%d', 1);",
            "count = fprintf(formatSpec, A...)",
        ),
        ("fread", "a = fread(3);", "data = fread(fid)"),
        ("frewind", "frewind(3);", "frewind(fid)"),
        ("fwrite", "n = fwrite(3, 1);", "count = fwrite(fid, data)"),
        ("full", "a = full(1);", "A = full(S)"),
        ("func2str", "s = func2str(@sin);", "name = func2str(fh)"),
        ("functions", "s = functions(@sin);", "info = functions(fh)"),
        ("getenv", "s = getenv('PATH');", "value = getenv(NAME)"),
        (
            "getfield",
            "s = struct(); s.x = 1; v = getfield(s, 'x');",
            "value = getfield(S, field)",
        ),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        assert!(analysis.compile_error.is_none(), "{source}");
        let column = source.find(name).expect("builtin call") as u32;
        let position = Position::new(0, column);
        let help = signature_help_at(source, &analysis, &position)
            .unwrap_or_else(|| panic!("signature help for {name}"));
        assert!(help
            .signatures
            .iter()
            .any(|signature| signature.label == expected_signature));
        let hover = hover_at(source, &analysis, &position)
            .unwrap_or_else(|| panic!("descriptor-backed hover for {name}"));
        let markdown = match hover.contents {
            HoverContents::Markup(markup) => markup.value,
            other => panic!("expected Markdown hover for {name}, got {other:?}"),
        };
        assert!(markdown.contains(expected_signature), "{name}: {markdown}");
    }
}

#[test]
fn file_io_and_field_access_integer_dispositions_are_public_completions() {
    let source = "x = 1;";
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    let completions = completion_at(source, &analysis, &Position::new(0, 0));
    for name in BUILTINS {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        let descriptor = builtin.descriptor.expect("public descriptor");
        assert_eq!(
            descriptor.completion_policy,
            runmat_builtins::BuiltinCompletionPolicy::Public
        );
        assert!(!builtin.integer_capabilities.is_empty() || builtin.integer_audit.is_some());
        assert!(completions
            .iter()
            .any(|item| item.label.eq_ignore_ascii_case(name)));
    }
}

#[test]
fn file_io_and_field_access_extensions_are_independently_registered() {
    for (name, extension_ids) in [
        ("fopen", &["fopen-integer-fileid", "fopen-legacy-all"][..]),
        (
            "fprintf",
            &[
                "fprintf-integer-fileid",
                "fprintf-numeric-format",
                "fprintf-stream-label",
            ][..],
        ),
        (
            "fread",
            &[
                "fread-like",
                "fread-integer-fileid",
                "fread-resident-control",
            ][..],
        ),
        (
            "frewind",
            &["frewind-integer-fileid", "frewind-resident-fileid"][..],
        ),
        (
            "fwrite",
            &[
                "fwrite-integer-fileid",
                "fwrite-integer-skip",
                "fwrite-gpu-input",
            ][..],
        ),
        ("full", &["full-integer-sparse"][..]),
        (
            "getenv",
            &["getenv-character-matrix-name", "getenv-cell-string-name"][..],
        ),
        (
            "getfield",
            &["getfield-textual-index", "getfield-indexed-resident-field"][..],
        ),
    ] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        for extension_id in extension_ids {
            assert!(
                builtin.extensions.iter().any(|extension| {
                    extension.id == *extension_id
                        && extension.mode == runmat_builtins::BuiltinExtensionMode::RunMatOnly
                }),
                "{name} missing {extension_id}"
            );
        }
    }
    for name in ["func2str", "functions"] {
        assert!(runmat_builtins::builtin_function_by_name(name)
            .expect("registered builtin")
            .extensions
            .is_empty());
    }
}
