use lsp_types::Position;
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const CAPABILITY_BUILTINS: [&str; 6] = ["ifft2", "ifftn", "ifftshift", "head", "heatmap", "height"];

#[test]
fn multidimensional_fft_table_and_heatmap_integer_dispositions_are_public() {
    for name in CAPABILITY_BUILTINS {
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
    for name in ["hgload", "hgsave"] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert_eq!(
            builtin.integer_audit.map(|audit| audit.kind),
            Some(runmat_builtins::BuiltinIntegerAuditKind::NotApplicable),
            "{name}"
        );
        assert!(builtin.integer_capabilities.is_empty(), "{name}");
    }
}

#[test]
fn multidimensional_fft_and_heatmap_extension_metadata_is_visible() {
    for (name, extension_id) in [
        ("ifft2", "ifft2-wide-integer-data"),
        ("ifftn", "ifftn-wide-integer-data"),
        ("ifftshift", "ifftshift-multi-dimension-selector"),
        ("heatmap", "heatmap-gpu-cdata"),
        ("head", "head-gpu-row-count"),
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
fn multidimensional_fft_table_and_heatmap_signatures_are_available_to_lsp() {
    for (name, source, signature) in [
        ("ifft2", "y = ifft2([1 0; 0 0]);", "Y = ifft2(X)"),
        ("ifftn", "y = ifftn(ones(2,2,2));", "Y = ifftn(X)"),
        ("ifftshift", "y = ifftshift([1 2]);", "Y = ifftshift(X)"),
        ("head", "h = head([1;2], 1);", "B = head(A, n)"),
        ("heatmap", "h = heatmap([1 2]);", "h = heatmap(CData)"),
        (
            "heatmap",
            "h = heatmap([1 2], 'FontSize', 12);",
            "h = heatmap(CData, Name, Value, ...)",
        ),
        ("height", "n = height([1;2]);", "n = height(A)"),
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
