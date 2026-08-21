use lsp_types::Position;
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const BUILTINS: [&str; 5] = ["hann", "heaviside", "hilbert", "hypot", "ifft"];

#[test]
fn signal_and_inverse_fft_integer_dispositions_are_public() {
    for name in BUILTINS {
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
}

#[test]
fn signal_and_inverse_fft_extension_metadata_is_visible_in_matlab_mode() {
    for (name, extension_id) in [
        ("hann", "hann-logical-length"),
        ("heaviside", "heaviside-integer-input"),
        ("hilbert", "hilbert-integer-data"),
        ("hypot", "hypot-integer-input"),
        ("ifft", "ifft-wide-integer-data"),
    ] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(
            builtin.extensions.iter().any(|extension| {
                extension.id == extension_id
                    && extension.mode == runmat_builtins::BuiltinExtensionMode::RunMatOnly
            }),
            "{name} must publish {extension_id}"
        );
    }
}

#[test]
fn signal_and_inverse_fft_signatures_are_available_to_lsp() {
    for (name, source, signature) in [
        ("hann", "w = hann(4, 'single');", "w = hann(n, precision)"),
        ("heaviside", "y = heaviside(1);", "Y = heaviside(X)"),
        ("hilbert", "z = hilbert([1 0]);", "z = hilbert(x)"),
        ("hypot", "r = hypot(3, 4);", "R = hypot(X, Y)"),
        ("ifft", "y = ifft([1 0]);", "Y = ifft(X)"),
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
    }
}
