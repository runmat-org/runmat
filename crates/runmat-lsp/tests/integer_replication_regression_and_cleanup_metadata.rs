use lsp_types::Position;
use runmat_builtins::{
    BuiltinIntegerAuditKind, BuiltinIntegerInputAvailability, BuiltinIntegerOverflowRule,
};
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const CAPABILITY_NAMES: [(&str, usize); 12] = [
    ("regress", 2),
    ("repelem", 2),
    ("repmat", 2),
    ("resample", 3),
    ("rescale", 2),
    ("reshape", 2),
    ("rgb2gray", 2),
    ("ribbon", 2),
    ("ridge", 3),
    ("rlocus", 1),
    ("rmfield", 1),
    ("rmmissing", 2),
];

#[test]
fn replication_regression_and_cleanup_integer_metadata_is_complete() {
    for (name, form_count) in CAPABILITY_NAMES {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert_eq!(builtin.integer_capabilities.len(), form_count, "{name}");
        assert!(builtin.integer_audit.is_none(), "{name}");
        for capability in builtin.integer_capabilities {
            for input in capability.inputs {
                assert!(!input.classes.is_empty(), "{name}: {}", input.name);
            }
        }
    }
    for name in ["removeStopWords", "rmpath"] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(builtin.integer_capabilities.is_empty(), "{name}");
        assert_eq!(
            builtin.integer_audit.expect("integer audit").kind,
            BuiltinIntegerAuditKind::NotApplicable,
            "{name}"
        );
    }
}

#[test]
fn cohort_metadata_distinguishes_documented_extensions_rejections_and_saturation() {
    for name in [
        "repelem",
        "repmat",
        "rescale",
        "reshape",
        "ribbon",
        "rmfield",
        "rmmissing",
    ] {
        let builtin = runmat_builtins::builtin_function_by_name(name).unwrap();
        assert!(
            builtin.integer_capabilities.iter().any(|capability| {
                capability
                    .inputs
                    .iter()
                    .any(|input| input.availability == BuiltinIntegerInputAvailability::Documented)
            }),
            "{name}"
        );
    }
    for name in ["regress", "resample", "ridge", "rlocus"] {
        let builtin = runmat_builtins::builtin_function_by_name(name).unwrap();
        assert!(
            builtin.integer_capabilities.iter().any(|capability| {
                capability
                    .inputs
                    .iter()
                    .any(|input| input.availability == BuiltinIntegerInputAvailability::RunMatOnly)
            }),
            "{name}"
        );
    }
    let resample = runmat_builtins::builtin_function_by_name("resample").unwrap();
    assert!(resample.integer_capabilities.iter().any(|capability| {
        capability.inputs.iter().any(|input| {
            input.name == "X" && input.availability == BuiltinIntegerInputAvailability::Rejected
        })
    }));
    let rgb2gray = runmat_builtins::builtin_function_by_name("rgb2gray").unwrap();
    assert_eq!(
        rgb2gray.integer_capabilities[0].overflow,
        BuiltinIntegerOverflowRule::Saturate
    );
    assert!(rgb2gray
        .integer_capabilities
        .iter()
        .all(|capability| capability.overflow != BuiltinIntegerOverflowRule::EvidenceOpen));
}

#[test]
fn cohort_extension_metadata_publishes_stable_gates() {
    for (name, ids) in [
        (
            "regress",
            &["regress-integer-data", "regress-integer-alpha"][..],
        ),
        ("resample", &["resample-integer-options"][..]),
        (
            "ridge",
            &[
                "ridge-integer-data",
                "ridge-integer-k",
                "ridge-integer-scaled",
            ][..],
        ),
        ("rlocus", &["rlocus-integer-gain"][..]),
        ("rmmissing", &["rmmissing-integer-dimension"][..]),
    ] {
        let builtin = runmat_builtins::builtin_function_by_name(name).unwrap();
        for id in ids {
            assert!(
                builtin.extensions.iter().any(|extension| {
                    extension.id == *id && extension.error_identifier.is_some()
                }),
                "{name}: {id}"
            );
        }
    }
}

#[test]
fn cohort_signatures_are_visible_to_lsp() {
    for (name, source) in [
        ("regress", "b=regress(uint8([1;2]),uint8([1 1;1 2]));"),
        ("repelem", "b=repelem(uint8([1 2]),uint8(2));"),
        ("repmat", "b=repmat(uint8(1),uint8(2));"),
        ("resample", "y=resample([1 2],uint8(2),uint8(1));"),
        ("rescale", "r=rescale(uint8([1 2]));"),
        ("reshape", "b=reshape(uint8([1 2]),uint8(2),uint8(1));"),
        ("rgb2gray", "g=rgb2gray(uint8(reshape([1 2 3],1,1,3)));"),
        ("ribbon", "h=ribbon(uint8([1 2]));"),
        ("ridge", "b=ridge(uint8([1;2]),uint8([1;2]),uint8(1));"),
        ("rlocus", "r=rlocus(tf([1],[1 1]),uint8([0 1]));"),
        ("rmfield", "t=rmfield(struct('a',uint8(1)),'a');"),
        ("rmmissing", "r=rmmissing(uint8([1 2]));"),
        ("removeStopWords", "d=removeStopWords(uint8(1));"),
        ("rmpath", "p=rmpath(uint8(1));"),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}
