use lsp_types::Position;
use runmat_builtins::{
    BuiltinIntegerBackendRule, BuiltinIntegerInputAvailability, BuiltinIntegerOutputClassRule,
};
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const CAPABILITY_NAMES: [&str; 8] = [
    "norm",
    "normalize",
    "normcdf",
    "norminv",
    "normpdf",
    "normrnd",
    "mvnrnd",
    "mnrfit",
];

#[test]
fn normal_distribution_and_regression_integer_metadata_is_explicit() {
    for name in CAPABILITY_NAMES {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(!builtin.integer_capabilities.is_empty(), "{name}");
        assert!(builtin.integer_audit.is_none(), "{name}");
        for capability in builtin.integer_capabilities {
            for input in capability.inputs {
                assert_eq!(input.classes.len(), 8, "{name}: {}", input.name);
            }
        }
    }
}

#[test]
fn normal_family_metadata_exposes_floating_boundaries_and_host_fallbacks() {
    for name in ["normcdf", "norminv", "normpdf", "normrnd", "mvnrnd"] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(
            builtin.integer_capabilities.iter().any(|capability| {
                capability
                    .inputs
                    .iter()
                    .any(|input| input.availability == BuiltinIntegerInputAvailability::RunMatOnly)
            }),
            "{name}"
        );
        assert!(
            builtin.integer_capabilities.iter().any(|capability| {
                capability.backend == BuiltinIntegerBackendRule::GatherFallback
            }),
            "{name}"
        );
    }
    let mnrfit = runmat_builtins::builtin_function_by_name("mnrfit").unwrap();
    assert!(mnrfit
        .integer_capabilities
        .iter()
        .all(|capability| capability.output_class == BuiltinIntegerOutputClassRule::Double));
}

#[test]
fn normal_distribution_and_regression_extensions_are_visible_in_matlab_mode() {
    for (name, extension) in [
        ("norm", "norm-integer-data"),
        ("normalize", "normalize-integer-data"),
        ("normcdf", "normcdf-integer-x"),
        ("norminv", "norminv-integer-p"),
        ("normpdf", "normpdf-integer-x"),
        ("normrnd", "normrnd-integer-mu"),
        ("mvnrnd", "mvnrnd-integer-mu"),
        ("mnrfit", "mnrfit-integer-x"),
    ] {
        let builtin = runmat_builtins::builtin_function_by_name(name).unwrap();
        assert!(
            builtin
                .extensions
                .iter()
                .any(|candidate| candidate.id == extension),
            "{name}"
        );
    }
}

#[test]
fn normal_distribution_and_regression_signatures_are_visible_to_lsp() {
    for (name, source) in [
        ("norm", "n = norm(uint16([3 4]));"),
        ("normalize", "z = normalize(uint16([1 2 3]));"),
        ("normcdf", "p = normcdf(uint16(0));"),
        ("norminv", "x = norminv(uint16(0));"),
        ("normpdf", "p = normpdf(uint16(0));"),
        ("normrnd", "r = normrnd(uint16(0),uint16(1));"),
        ("mvnrnd", "r = mvnrnd(uint16([0 0]),uint16([1 0;0 1]));"),
        ("mnrfit", "B = mnrfit(uint16([0;1]),uint16([1;2]));"),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}
