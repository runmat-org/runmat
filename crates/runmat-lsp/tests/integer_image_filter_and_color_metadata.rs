use lsp_types::Position;
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const COHORT: [&str; 8] = [
    "imag", "imfilter", "imhist", "imwrite", "ind2rgb", "rgb2gray", "rgb2hsv", "rgb2lab",
];

#[test]
fn image_filter_and_color_integer_metadata_is_public_and_honest() {
    for name in COHORT {
        let builtin =
            runmat_builtins::builtin_catalog_entry_by_name(name).expect("canonical builtin");
        assert_eq!(
            builtin.descriptor.completion_policy,
            runmat_builtins::BuiltinCompletionPolicy::Public,
            "{name}"
        );
        assert!(!builtin.integer_capabilities.is_empty(), "{name}");
    }
    {
        let name = "imhist";
        let builtin =
            runmat_builtins::builtin_catalog_entry_by_name(name).expect("canonical builtin");
        assert!(
            builtin
                .integer_capabilities
                .iter()
                .any(|capability| capability.notes.contains("evidence-open")
                    || capability
                        .inputs
                        .iter()
                        .any(|input| input.notes.contains("evidence-open"))),
            "{name} must expose its bounded evidence gap"
        );
    }
    let imag = runmat_builtins::builtin_catalog_entry_by_name("imag").unwrap();
    let [capability] = imag.integer_capabilities else {
        panic!("one exact component projection")
    };
    let [input] = capability.inputs else {
        panic!("one source role")
    };
    assert_eq!(input.classes, &runmat_builtins::ALL_INTEGER_CLASSES);
    assert_eq!(
        input.availability,
        runmat_builtins::BuiltinIntegerInputAvailability::Documented
    );
    assert_eq!(
        input.scalar_double,
        runmat_builtins::BuiltinIntegerScalarDoubleRule::NotApplicable
    );
    assert_eq!(
        capability.computation_domain,
        runmat_builtins::BuiltinIntegerComputationDomain::ExactInteger
    );
    assert_eq!(
        capability.output_class,
        runmat_builtins::BuiltinIntegerOutputClassRule::PreserveInput
    );
    assert_eq!(
        capability.overflow,
        runmat_builtins::BuiltinIntegerOverflowRule::NotApplicable
    );
    assert_eq!(
        capability.backend,
        runmat_builtins::BuiltinIntegerBackendRule::HostAndGpu
    );
    assert_eq!(
        capability.overload,
        runmat_builtins::BuiltinIntegerOverloadKind::ElementwiseShapePreserving
    );
    assert!(capability.notes.contains("exact same-class zeros"));
    assert!(capability
        .notes
        .contains("projects its exact imaginary storage"));
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
