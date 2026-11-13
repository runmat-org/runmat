mod support;
#[path = "support/hover.rs"]
mod hover_support;

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

fn builtin_metadata(
    name: &str,
) -> (
    &'static runmat_builtins::BuiltinDescriptor,
    &'static [runmat_builtins::BuiltinExtensionDescriptor],
    &'static [runmat_builtins::BuiltinIntegerCapabilityDescriptor],
    Option<&'static runmat_builtins::BuiltinIntegerAuditDescriptor>,
) {
    let entry = runmat_builtins::builtin_catalog_entry_by_name(name)
        .unwrap_or_else(|| panic!("canonical builtin {name}"));
    (
        entry.descriptor,
        entry.extensions,
        entry.integer_capabilities,
        entry.integer_audit,
    )
}

#[test]
fn file_io_and_field_access_descriptors_are_public_and_visible() {
    for (name, source) in [
        ("fopen", "fid = fopen('x');"),
        ("fprintf", "n = fprintf('%d', 1);"),
        ("fread", "a = fread(3);"),
        ("frewind", "frewind(3);"),
        ("fwrite", "n = fwrite(3, 1);"),
        ("full", "a = full(1);"),
        ("func2str", "s = func2str(@sin);"),
        ("functions", "s = functions(@sin);"),
        ("getenv", "s = getenv('PATH');"),
        ("getfield", "s = struct(); s.x = 1; v = getfield(s, 'x');"),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        assert!(analysis.compile_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin call") as u32);
        let help =
            signature_help_at(source, &analysis, &position).expect("canonical signature help");
        support::assert_catalog_signatures(name, &help);
        let hover = hover_at(source, &analysis, &position).expect("canonical hover");
        let HoverContents::Markup(markup) = hover.contents else {
            panic!("expected Markdown hover");
        };
        hover_support::assert_catalog_hover(name, &help, &markup.value);
    }
}
#[test]
fn file_io_and_field_access_integer_dispositions_are_public_completions() {
    let source = "x = 1;";
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    let completions = completion_at(source, &analysis, &Position::new(0, 0));
    for name in BUILTINS {
        let (descriptor, _, integer_capabilities, integer_audit) = builtin_metadata(name);
        assert_eq!(
            descriptor.completion_policy,
            runmat_builtins::BuiltinCompletionPolicy::Public
        );
        assert!(!integer_capabilities.is_empty() || integer_audit.is_some());
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
        let (_, extensions, _, _) = builtin_metadata(name);
        for extension_id in extension_ids {
            assert!(
                extensions.iter().any(|extension| {
                    extension.id == *extension_id
                        && extension.mode == runmat_builtins::BuiltinExtensionMode::RunMatOnly
                }),
                "{name} missing {extension_id}"
            );
        }
    }
    for name in ["func2str", "functions"] {
        assert!(builtin_metadata(name).1.is_empty());
    }
}
