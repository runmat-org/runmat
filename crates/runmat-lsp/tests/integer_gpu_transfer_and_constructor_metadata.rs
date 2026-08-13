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
        let catalog = runmat_builtins::builtin_catalog_entry_by_name(name);
        let binding = runmat_builtins::builtin_function_by_name(name);
        let descriptor = catalog
            .map(|entry| entry.descriptor)
            .or_else(|| binding.and_then(|binding| binding.descriptor))
            .unwrap_or_else(|| panic!("catalog entry or runtime binding for {name}"));
        let integer_capabilities = catalog.map_or_else(
            || binding.expect("runtime binding").integer_capabilities,
            |entry| entry.integer_capabilities,
        );
        assert_eq!(
            descriptor.completion_policy,
            runmat_builtins::BuiltinCompletionPolicy::Public,
            "{name}"
        );
        assert!(!integer_capabilities.is_empty(), "{name}");
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
