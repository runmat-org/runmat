use lsp_types::{HoverContents, Position};
use runmat_lsp::core::analysis::{
    analyze_document_with_compat, hover_at, signature_help_at, CompatMode,
};

#[test]
fn import_and_network_integer_dispositions_are_public_and_complete() {
    for (name, capability_forms) in [
        ("parquetDatastore", 1),
        ("read", 3),
        ("readcell", 2),
        ("readmatrix", 3),
        ("readtable", 3),
        ("readtimetable", 3),
        ("spreadsheetImportOptions", 2),
        ("textscan", 2),
    ] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert_eq!(
            builtin
                .descriptor
                .expect("public descriptor")
                .completion_policy,
            runmat_builtins::BuiltinCompletionPolicy::Public,
            "{name}"
        );
        assert_eq!(
            builtin.integer_capabilities.len(),
            capability_forms,
            "{name}"
        );
        assert!(builtin.integer_audit.is_none(), "{name}");
    }

    let readline = runmat_builtins::builtin_function_by_name("readline").expect("readline");
    assert!(readline.integer_capabilities.is_empty());
    assert_eq!(
        readline.integer_audit.expect("readline integer audit").kind,
        runmat_builtins::BuiltinIntegerAuditKind::NotApplicable
    );
}

#[test]
fn import_extension_metadata_publishes_independent_compatibility_gates() {
    for (name, extension_ids) in [
        (
            "readmatrix",
            &["readmatrix-like-output", "readmatrix-typed-integer-control"][..],
        ),
        ("readcell", &["readcell-typed-integer-control"][..]),
        ("readtable", &["readtable-typed-integer-control"][..]),
        (
            "readtimetable",
            &["readtimetable-typed-integer-control"][..],
        ),
        (
            "spreadsheetImportOptions",
            &["spreadsheetimportoptions-typed-integer-location-control"][..],
        ),
        ("textscan", &["textscan-typed-integer-control"][..]),
    ] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        for extension_id in extension_ids {
            assert!(
                builtin.extensions.iter().any(|extension| {
                    extension.id == *extension_id
                        && extension.mode == runmat_builtins::BuiltinExtensionMode::RunMatOnly
                        && extension.error_identifier.is_some()
                }),
                "{name} missing {extension_id}"
            );
        }
    }
}

#[test]
fn import_signatures_and_reference_docs_are_visible_to_lsp() {
    for (name, source, signature) in [
        (
            "readmatrix",
            "m=readmatrix('x.csv');",
            "M = readmatrix(filename)",
        ),
        (
            "readtable",
            "t=readtable('x.csv');",
            "T = readtable(filename)",
        ),
        (
            "textscan",
            "c=textscan('1','%u8');",
            "C = textscan(textOrFileID, formatSpec)",
        ),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        assert!(analysis.compile_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(
            help.signatures.iter().any(|item| item.label == signature),
            "expected {signature} for {source}"
        );
        let hover = hover_at(source, &analysis, &position).expect("descriptor-backed hover");
        let HoverContents::Markup(markup) = hover.contents else {
            panic!("expected Markdown hover for {name}");
        };
        assert!(markup.value.contains(signature), "{name}: {}", markup.value);
    }
}
