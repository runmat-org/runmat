use lsp_types::Position;
use runmat_builtins::{
    BuiltinIntegerBackendRule, BuiltinIntegerInputAvailability, BuiltinIntegerOutputClassRule,
};
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const CAPABILITY_NAMES: [&str; 8] = [
    "imag",
    "magic",
    "mat2cell",
    "mat2str",
    "meshgrid",
    "native2unicode",
    "ndgrid",
    "nextpow2",
];

#[test]
fn array_construction_and_conversion_integer_metadata_is_explicit() {
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
fn grid_and_conversion_class_and_backend_rules_are_visible() {
    for name in ["meshgrid", "ndgrid"] {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert_eq!(
            builtin.integer_capabilities[0].output_class,
            BuiltinIntegerOutputClassRule::PreserveInput
        );
        assert_eq!(
            builtin.integer_capabilities[0].backend,
            BuiltinIntegerBackendRule::GpuRestricted
        );
    }
    let mat2cell =
        runmat_builtins::builtin_function_by_name("mat2cell").expect("registered builtin");
    assert_eq!(
        mat2cell.integer_capabilities[1].inputs[0].availability,
        BuiltinIntegerInputAvailability::RunMatOnly
    );
    let mat2str = runmat_builtins::builtin_function_by_name("mat2str").expect("registered builtin");
    assert_eq!(
        mat2str.integer_capabilities[1].inputs[0].availability,
        BuiltinIntegerInputAvailability::RunMatOnly
    );
}

#[test]
fn array_construction_and_conversion_signatures_are_visible_to_lsp() {
    for (name, source) in [
        ("imag", "y = imag(uint16(1));"),
        ("magic", "a = magic(uint16(3));"),
        ("mat2cell", "c = mat2cell(uint16([1 2]),1,[1 1]);"),
        ("mat2str", "s = mat2str(uint16([1 2]),'class');"),
        ("meshgrid", "[x,y] = meshgrid(uint16([1 2]));"),
        (
            "native2unicode",
            "text = native2unicode(uint16([104 105]));",
        ),
        ("ndgrid", "[x,y] = ndgrid(uint16([1 2]));"),
        ("nextpow2", "p = nextpow2(uint16([1 3]));"),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}
