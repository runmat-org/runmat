use lsp_types::{HoverContents, Position};
use runmat_lsp::core::analysis::{
    analyze_document_with_compat, completion_at, hover_at, signature_help_at, CompatMode,
};

const BUILTINS: [&str; 9] = [
    "gradient",
    "gray2rgb",
    "griddedInterpolant",
    "groupcounts",
    "groupsummary",
    "grp2idx",
    "grpstats",
    "gt",
    "hamming",
];

#[test]
fn interpolation_grouping_comparison_and_window_dispositions_are_public() {
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
        assert!(!builtin.integer_capabilities.is_empty(), "{name}");
        assert!(completions
            .iter()
            .any(|item| item.label.eq_ignore_ascii_case(name)));
    }
}

#[test]
fn interpolation_grouping_and_window_extensions_are_independently_registered() {
    for (name, ids) in [
        (
            "gradient",
            &["gradient-integer-data", "gradient-integer-spacing"][..],
        ),
        ("gray2rgb", &["gray2rgb-callable"][..]),
        (
            "griddedInterpolant",
            &[
                "griddedinterpolant-integer-grid",
                "griddedinterpolant-integer-query",
                "griddedinterpolant-integer-values",
            ][..],
        ),
        ("groupcounts", &["groupcounts-resident-input"][..]),
        ("groupsummary", &["groupsummary-resident-input"][..]),
        (
            "grpstats",
            &[
                "grpstats-integer-data",
                "grpstats-integer-selector",
                "grpstats-resident-input",
            ][..],
        ),
        ("hamming", &["hamming-logical-length"][..]),
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
        ("gradient", "g = gradient([1 4 9]);", "G = gradient(F)"),
        ("gray2rgb", "rgb = gray2rgb([0 1]);", "RGB = gray2rgb(I)"),
        (
            "griddedInterpolant",
            "f = griddedInterpolant([1 4 9]);",
            "F = griddedInterpolant(V)",
        ),
        ("gt", "tf = gt(2,1);", "tf = gt(A, B)"),
        (
            "hamming",
            "w = hamming(4, 'single');",
            "w = hamming(n, precision)",
        ),
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
