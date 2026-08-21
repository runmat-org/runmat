use lsp_types::Position;
use runmat_builtins::{
    BuiltinIntegerBackendRule, BuiltinIntegerInputAvailability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule,
};
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const CAPABILITY_NAMES: [&str; 9] = [
    "pagefun",
    "pagemtimes",
    "pagetranspose",
    "pinv",
    "qr",
    "rank",
    "rcond",
    "real",
    "realsqrt",
];

#[test]
fn page_and_decomposition_integer_metadata_is_explicit() {
    for name in CAPABILITY_NAMES {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(!builtin.integer_capabilities.is_empty(), "{name}");
        assert!(builtin.integer_audit.is_none(), "{name}");
    }
}

#[test]
fn floating_decomposition_extensions_declare_checked_runmat_boundaries() {
    for name in ["pagemtimes", "pinv", "qr", "rank", "rcond"] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(!builtin.extensions.is_empty(), "{name}");
        assert!(
            builtin.integer_capabilities.iter().all(|capability| {
                capability.overflow == BuiltinIntegerOverflowRule::Error
                    && capability.inputs.iter().all(|input| {
                        input.classes.len() == 8
                            && input.availability == BuiltinIntegerInputAvailability::RunMatOnly
                    })
            }),
            "{name}"
        );
        assert!(
            builtin.integer_capabilities.iter().any(|capability| {
                capability.backend == BuiltinIntegerBackendRule::GatherFallback
            }),
            "{name} matrix boundary"
        );
    }
}

#[test]
fn exact_page_projection_and_realsqrt_rejection_are_distinguished() {
    let pagefun = runmat_builtins::builtin_function_by_name("pagefun").expect("pagefun");
    assert_eq!(
        pagefun.integer_capabilities[0].output_class,
        BuiltinIntegerOutputClassRule::FunctionSpecific
    );
    assert_eq!(
        pagefun.integer_capabilities[0].overflow,
        BuiltinIntegerOverflowRule::FunctionSpecific
    );

    let transpose =
        runmat_builtins::builtin_function_by_name("pagetranspose").expect("pagetranspose");
    assert_eq!(
        transpose.integer_capabilities[0].output_class,
        BuiltinIntegerOutputClassRule::PreserveInput
    );
    assert_eq!(
        transpose.integer_capabilities[0].inputs[0].availability,
        BuiltinIntegerInputAvailability::Documented
    );
    assert_eq!(transpose.integer_capabilities[0].inputs[0].classes.len(), 8);

    let real = runmat_builtins::builtin_function_by_name("real").expect("real");
    assert_eq!(
        real.integer_capabilities[0].output_class,
        BuiltinIntegerOutputClassRule::PreserveInput
    );
    assert_eq!(
        real.integer_capabilities[0].overflow,
        BuiltinIntegerOverflowRule::NotApplicable
    );

    let realsqrt = runmat_builtins::builtin_function_by_name("realsqrt").expect("realsqrt");
    assert_eq!(
        realsqrt.integer_capabilities[0].inputs[0].availability,
        BuiltinIntegerInputAvailability::Rejected
    );
    assert!(realsqrt.integer_capabilities[0].inputs[0]
        .classes
        .is_empty());
}

#[test]
fn page_and_decomposition_signatures_are_visible_to_lsp() {
    for (name, source) in [
        ("pagefun", "z=pagefun(@mtimes,uint16(1),uint16(1));"),
        ("pagemtimes", "z=pagemtimes(uint16(1),uint16(1));"),
        ("pagetranspose", "z=pagetranspose(uint16([1 2]));"),
        ("pinv", "z=pinv(uint16([1 2;3 4]));"),
        ("qr", "[q,r,p]=qr(uint16([1 2;3 4]));"),
        ("rank", "z=rank(uint16([1 2;3 4]));"),
        ("rcond", "z=rcond(uint16([1 2;3 4]));"),
        ("real", "z=real(uint16([1 2]));"),
        ("realsqrt", "z=realsqrt(uint16([1 2]));"),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}
