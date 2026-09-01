use lsp_types::{HoverContents, Position};
use runmat_lsp::core::analysis::{
    analyze_document_with_compat, hover_at, signature_help_at, CompatMode,
};

#[test]
fn migrated_signatures_and_documentation_come_from_the_canonical_catalog() {
    for (name, source, expected) in [
        ("sqrt", "x=sqrt(single(4));", "Y = sqrt(X)"),
        ("realsqrt", "x=realsqrt(single(4));", "Y = realsqrt(X)"),
        ("exp", "x=exp(single(1));", "Y = exp(X)"),
        ("expm1", "x=expm1(single(1));", "Y = expm1(X)"),
        ("intmin", "x=intmin(\"int16\");", "value = intmin(typename)"),
        (
            "intmax",
            "x=intmax(\"uint16\");",
            "value = intmax(typename)",
        ),
        (
            "realmin",
            "x=realmin(\"single\");",
            "value = realmin(typename)",
        ),
        (
            "realmax",
            "x=realmax(\"single\");",
            "value = realmax(typename)",
        ),
        (
            "flintmax",
            "x=flintmax(\"single\");",
            "value = flintmax(typename)",
        ),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(
            help.signatures
                .iter()
                .any(|signature| signature.label == expected),
            "{name}: {:?}",
            help.signatures
        );
        let hover = hover_at(source, &analysis, &position).expect("catalog hover");
        let HoverContents::Markup(markup) = hover.contents else {
            panic!("expected Markdown hover for {name}");
        };
        assert!(markup.value.contains(expected), "{name}: {}", markup.value);
        assert!(
            markup.value.contains("**Examples**"),
            "canonical examples missing from {name} hover: {}",
            markup.value
        );
        assert!(
            markup.value.contains("**GPU execution**"),
            "canonical GPU documentation missing from {name} hover: {}",
            markup.value
        );
    }
}
