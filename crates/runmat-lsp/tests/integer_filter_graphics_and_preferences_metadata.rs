use lsp_types::{HoverContents, Position};
use runmat_lsp::core::analysis::{
    analyze_document_with_compat, completion_at, hover_at, signature_help_at, CompatMode,
};

const BUILTINS: [&str; 10] = [
    "fspecial",
    "fsurf",
    "gauspuls",
    "gca",
    "genvarname",
    "get",
    "getAttribute",
    "getmethod",
    "getpref",
    "gobjects",
];

#[test]
fn filter_graphics_and_preference_integer_dispositions_are_public() {
    let source = "x = 1;";
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    let completions = completion_at(source, &analysis, &Position::new(0, 0));
    for name in BUILTINS {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert_eq!(
            builtin
                .descriptor
                .expect("public descriptor")
                .completion_policy,
            runmat_builtins::BuiltinCompletionPolicy::Public
        );
        assert!(!builtin.integer_capabilities.is_empty() || builtin.integer_audit.is_some());
        assert!(completions
            .iter()
            .any(|item| item.label.eq_ignore_ascii_case(name)));
    }
}

#[test]
fn filter_graphics_and_preference_extensions_are_independently_registered() {
    for (name, ids) in [
        (
            "fspecial",
            &["fspecial-nondouble-size", "fspecial-resident-output"][..],
        ),
        (
            "fsurf",
            &["fsurf-integer-domain", "fsurf-integer-style-property"][..],
        ),
        (
            "gauspuls",
            &["gauspuls-integer-time", "gauspuls-resident-input"][..],
        ),
        (
            "gca",
            &["gca-figure-argument", "gca-integer-figure-alias"][..],
        ),
        ("get", &["get-integer-handle-alias"][..]),
        ("getmethod", &["getmethod-bound-method-handle"][..]),
        ("getpref", &["getpref-group-query"][..]),
    ] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        for id in ids {
            assert!(builtin
                .extensions
                .iter()
                .any(|extension| extension.id == *id));
        }
    }
}

#[test]
fn descriptor_help_exposes_representative_settled_forms() {
    for (name, source, signature) in [
        ("fspecial", "h = fspecial('average');", "H = fspecial(type)"),
        ("fsurf", "h = fsurf(@(x,y) x+y);", "h = fsurf(f)"),
        ("gauspuls", "y = gauspuls(0);", "Y = gauspuls(T)"),
        ("gca", "ax = gca();", "ax = gca()"),
        (
            "getpref",
            "v = getpref('g','p');",
            "value = getpref(group, pref)",
        ),
        ("gobjects", "h = gobjects(2,3);", "h = gobjects(sz...)"),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(help.signatures.iter().any(|item| item.label == signature));
        let hover = hover_at(source, &analysis, &position).expect("hover");
        let HoverContents::Markup(markup) = hover.contents else {
            panic!("Markdown hover expected");
        };
        assert!(markup.value.contains(signature));
    }
}
