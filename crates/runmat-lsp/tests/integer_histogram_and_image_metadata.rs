use lsp_types::Position;
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const COHORT: [&str; 11] = [
    "hist",
    "histc",
    "histcounts2",
    "histogram",
    "histogram2",
    "hsv2rgb",
    "im2double",
    "im2uint16",
    "im2uint8",
    "image",
    "imagesc",
];

#[test]
fn histogram_and_image_integer_metadata_is_public() {
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
        assert!(builtin.integer_audit.is_none(), "{name}");
    }
}

#[test]
fn histogram_and_image_signatures_are_visible_to_lsp() {
    for (name, source, expected) in [
        (
            "hist",
            "[n,c] = hist([1 2]);",
            "[counts, centers] = hist(X)",
        ),
        (
            "histc",
            "n = histc([1 2],[1 2]);",
            "bincounts = histc(x, binranges)",
        ),
        (
            "histcounts2",
            "n = histcounts2([1],[2]);",
            "N = histcounts2(X, Y)",
        ),
        ("histogram", "h = histogram([1 2]);", "h = histogram(X)"),
        (
            "histogram2",
            "h = histogram2([1],[2]);",
            "h = histogram2(X, Y)",
        ),
        ("hsv2rgb", "r = hsv2rgb([1 1 1]);", "RGB = hsv2rgb(HSV)"),
        ("im2double", "d = im2double(uint8(1));", "J = im2double(I)"),
        ("im2uint16", "u = im2uint16(uint8(1));", "J = im2uint16(I)"),
        ("im2uint8", "u = im2uint8(uint16(1));", "J = im2uint8(I)"),
        ("image", "h = image([1 2]);", "h = image(C)"),
        (
            "imagesc",
            "h = imagesc([1 2],[0 2]);",
            "h = imagesc(C, clims)",
        ),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        assert!(analysis.compile_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(
            help.signatures
                .iter()
                .any(|signature| signature.label == expected),
            "expected {expected} for {source}"
        );
    }
}
