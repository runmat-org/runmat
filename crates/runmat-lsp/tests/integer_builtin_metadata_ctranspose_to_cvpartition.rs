use lsp_types::Position;
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

#[test]
fn ctranspose_to_cvpartition_descriptors_are_visible_to_lsp() {
    for (source, expected) in [
        ("m = csvread('x.csv');", "M = csvread(filename)"),
        ("csvwrite('x.csv', uint8(1));", "csvwrite(filename, M)"),
        ("b = ctranspose(uint8(1));", "B = ctranspose(A)"),
        ("q = cumtrapz(uint8([1 2]));", "Q = cumtrapz(Y)"),
        (
            "c = cvpartition(6, 'KFold', 3);",
            "c = cvpartition(n, 'KFold', k)",
        ),
        ("w = damp(tf(1, [1 1]));", "wn = damp(sys)"),
        ("r = daspect();", "ratio = daspect()"),
        (
            "row = dataTipTextRow('x', uint8(1));",
            "row = dataTipTextRow(label, value)",
        ),
        ("d = datacursormode();", "dcm = datacursormode()"),
        (
            "y = datasample(uint8([1 2]), 1);",
            "y = datasample(data, k)",
        ),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        assert!(analysis.compile_error.is_none(), "{source}");
        let column = source.find('(').expect("call expression") as u32 - 1;
        let help = signature_help_at(source, &analysis, &Position::new(0, column))
            .unwrap_or_else(|| panic!("descriptor-backed signature help for {source}"));
        assert!(
            help.signatures
                .iter()
                .any(|signature| signature.label == expected),
            "expected {expected} for {source}"
        );
    }
}

#[test]
fn matlab_mode_keeps_signatures_and_extension_metadata_visible_to_lsp_consumers() {
    for (name, source, extension_id) in [
        (
            "csvread",
            "m = csvread('x.csv', 0, 0, uint8([0 0]));",
            "csvread-two-vector-range",
        ),
        (
            "csvwrite",
            "bytes = csvwrite('x.csv', uint8(1));",
            "csvwrite-bytes-written-output",
        ),
        (
            "cumtrapz",
            "q = cumtrapz(uint8([1 2]));",
            "cumtrapz-integer-y",
        ),
        (
            "cvpartition",
            "c = cvpartition(uint8(6), 'KFold', 3);",
            "cvpartition-integer-observation-count",
        ),
        (
            "datasample",
            "y = datasample(uint8([1 2]), 1);",
            "datasample-integer-data",
        ),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::Matlab);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        assert!(analysis.compile_error.is_none(), "{source}");
        let column = source.find('(').expect("call expression") as u32 - 1;
        let help = signature_help_at(source, &analysis, &Position::new(0, column))
            .unwrap_or_else(|| panic!("MATLAB-mode signature help for {source}"));
        assert!(!help.signatures.is_empty(), "{source}");
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(
            builtin.extensions.iter().any(|extension| {
                extension.id == extension_id
                    && extension.mode == runmat_builtins::BuiltinExtensionMode::RunMatOnly
            }),
            "{name} must expose {extension_id} to LSP metadata"
        );
    }
}
