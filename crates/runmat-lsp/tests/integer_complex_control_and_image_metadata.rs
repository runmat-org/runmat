use lsp_types::{Position, SignatureHelp};
use runmat_builtins::{BuiltinCompletionPolicy, BuiltinExtensionMode};
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const PACKET: [&str; 8] = [
    "imag", "imfilter", "imhist", "impulse", "imwrite", "lab2rgb", "rgb2gray", "rgb2lab",
];

#[test]
fn complex_control_and_image_packet_is_public_and_audited() {
    for name in PACKET {
        let builtin = runmat_builtins::builtin_function_by_name(name)
            .unwrap_or_else(|| panic!("registered builtin {name}"));
        let descriptor = builtin
            .descriptor
            .unwrap_or_else(|| panic!("{name} descriptor"));
        assert_eq!(
            descriptor.completion_policy,
            BuiltinCompletionPolicy::Public
        );
        assert!(!descriptor.signatures.is_empty(), "{name} signatures");
        assert!(
            !builtin.integer_capabilities.is_empty(),
            "{name} capabilities"
        );
        assert!(
            builtin.integer_audit.is_none(),
            "{name} is capability-bearing"
        );
        if !matches!(name, "imag" | "imhist" | "rgb2gray") {
            assert!(
                builtin
                    .integer_capabilities
                    .iter()
                    .all(|capability| !capability.notes.contains("[integer-audit-open]")),
                "{name} must be settled"
            );
        }
    }
}

#[test]
fn image_and_control_signatures_are_visible_to_lsp() {
    for (name, source, expected) in [
        ("imag", "y=imag(uint8(1));", "Y = imag(X)"),
        ("imfilter", "y=imfilter(uint8(1),1);", "B = imfilter(A, H)"),
        ("imhist", "y=imhist(uint8(1));", "counts = imhist(I)"),
        (
            "impulse",
            "sys=tf(1,[1 1]); y=impulse(sys);",
            "y = impulse(sys)",
        ),
        (
            "imwrite",
            "imwrite(uint8(1),'x.png');",
            "imwrite(A, filename)",
        ),
        (
            "lab2rgb",
            "y=lab2rgb([70 5 10],'OutputType','uint8');",
            "RGB = lab2rgb(LAB, OutputType=type)",
        ),
        (
            "rgb2gray",
            "y=rgb2gray(uint8(ones(1,1,3)));",
            "I = rgb2gray(RGB)",
        ),
        (
            "rgb2lab",
            "y=rgb2lab(uint8(ones(1,1,3)));",
            "LAB = rgb2lab(RGB, Name=Value)",
        ),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let SignatureHelp { signatures, .. } =
            signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(
            signatures
                .iter()
                .any(|signature| signature.label == expected),
            "{name}: {:?}",
            signatures
                .iter()
                .map(|signature| &signature.label)
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn packet_extensions_are_declarative_and_independently_named() {
    for (name, extension) in [
        ("impulse", "impulse-integer-numeric-role"),
        ("imfilter", "imfilter-valid-output-shape"),
        ("imhist", "imhist-typed-integer-bin-count"),
        ("imwrite", "imwrite-single-gif-tiff"),
        ("lab2rgb", "lab2rgb-explicit-gpu-input"),
        ("rgb2lab", "rgb2lab-explicit-gpu"),
    ] {
        let builtin = runmat_builtins::builtin_function_by_name(name).unwrap();
        assert!(builtin.extensions.iter().any(|record| {
            record.id == extension && record.mode == BuiltinExtensionMode::RunMatOnly
        }));
    }
}
