use lsp_types::Position;
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

#[test]
fn cosine_family_descriptors_are_visible_to_lsp_in_runmat_mode() {
    for (source, expected) in [
        ("y = cos(uint8(0));", "Y = cos(X)"),
        ("y = cosd(uint8(0));", "Y = cosd(X)"),
        ("y = cosh(uint8(0));", "Y = cosh(X)"),
        ("y = cospi(uint8(0));", "Y = cospi(X)"),
        (
            "y = cosineSimilarity(uint8([1 0]));",
            "similarities = cosineSimilarity(M)",
        ),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        assert!(analysis.compile_error.is_none(), "{source}");
        let help = signature_help_at(source, &analysis, &Position::new(0, 4))
            .expect("descriptor-backed signature help");
        assert!(
            help.signatures
                .iter()
                .any(|signature| signature.label == expected),
            "expected {expected} for {source}"
        );
    }
}

#[test]
fn matlab_mode_keeps_integer_extension_metadata_visible_for_runtime_gates() {
    for (name, source, extension_id) in [
        ("cos", "y = cos(uint8(0));", "cos-integer-input"),
        ("cosd", "y = cosd(uint8(0));", "cosd-integer-input"),
        ("cosh", "y = cosh(uint8(0));", "cosh-integer-input"),
        ("cospi", "y = cospi(uint8(0));", "cospi-integer-input"),
        (
            "cosineSimilarity",
            "y = cosineSimilarity(uint8([1 0]));",
            "cosine-similarity-integer-matrix",
        ),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::Matlab);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        assert!(analysis.compile_error.is_none(), "{source}");
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(
            builtin.extensions.iter().any(|extension| {
                extension.id == extension_id
                    && extension.mode == runmat_builtins::BuiltinExtensionMode::RunMatOnly
            }),
            "{name} must expose its RunMat-only integer gate to LSP metadata"
        );
    }
}
