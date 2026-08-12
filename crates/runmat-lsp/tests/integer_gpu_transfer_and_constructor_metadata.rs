use lsp_types::Position;
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const COHORT: [&str; 8] = [
    "gather",
    "gpuArray",
    "gpuDevice",
    "isgpuarray",
    "inf",
    "nan",
    "ones",
    "rand",
];

#[test]
fn gpu_transfer_and_constructor_integer_metadata_is_public() {
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
    }
}

#[test]
fn gpu_transfer_and_constructor_signatures_are_visible_to_lsp() {
    for (name, source) in [
        ("gather", "g = gpuArray(1); x = gather(g);"),
        ("gpuArray", "g = gpuArray(uint8(1));"),
        ("gpuDevice", "d = gpuDevice(uint8(1));"),
        ("isgpuarray", "g = gpuArray(1); tf = isgpuarray(g);"),
        ("inf", "x = inf(uint8(2));"),
        ("nan", "x = nan(uint8(2));"),
        ("ones", "x = ones(uint8(2));"),
        ("rand", "x = rand(uint8(2));"),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        assert!(analysis.compile_error.is_none(), "{source}");
        let position = Position::new(0, source.rfind(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}
