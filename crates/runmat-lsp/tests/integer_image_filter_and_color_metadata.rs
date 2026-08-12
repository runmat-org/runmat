use lsp_types::Position;
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const COHORT: [&str; 8] = [
    "imag", "imfilter", "imhist", "imwrite", "ind2rgb", "rgb2gray", "rgb2hsv", "rgb2lab",
];

#[test]
fn image_filter_and_color_integer_metadata_is_public_and_honest() {
    for name in COHORT {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert_eq!(
            builtin
                .descriptor
                .expect("public descriptor")
                .completion_policy,
            runmat_builtins::BuiltinCompletionPolicy::Public,
            "{name}"
        );
        assert!(!builtin.integer_capabilities.is_empty(), "{name}");
    }
    for name in [
        "imag", "imfilter", "imhist", "imwrite", "rgb2gray", "rgb2lab",
    ] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(
            builtin
                .integer_capabilities
                .iter()
                .any(|capability| capability.notes.contains("[integer-audit-open]")),
            "{name} must expose its bounded evidence gap"
        );
    }
}

#[test]
fn image_filter_and_color_signatures_are_visible_to_lsp() {
    for (name, source) in [
        ("imag", "y = imag(uint8(1));"),
        ("imfilter", "y = imfilter(uint8([1 2]),[1 1]);"),
        ("imhist", "n = imhist(uint8([1 2]));"),
        ("imwrite", "imwrite(uint8(1),'a.png');"),
        ("ind2rgb", "y = ind2rgb(uint8(0),[1 0 0]);"),
        ("rgb2gray", "y = rgb2gray(uint8(zeros(1,1,3)));"),
        ("rgb2hsv", "y = rgb2hsv(uint8(zeros(1,1,3)));"),
        (
            "rgb2lab",
            "y = rgb2lab(uint8([1 2 3]),'ColorSpace','srgb');",
        ),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        assert!(analysis.compile_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}
