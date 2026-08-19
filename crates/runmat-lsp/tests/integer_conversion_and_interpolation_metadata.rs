use lsp_types::Position;
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const CAPABILITY_NAMES: [&str; 10] = [
    "int", "int2str", "int32", "integral", "interp1", "interp1q", "interp2", "intmax", "intmin",
    "inv",
];

#[test]
fn conversion_and_interpolation_integer_metadata_is_explicit() {
    for name in CAPABILITY_NAMES {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(!builtin.integer_capabilities.is_empty(), "{name}");
    }
}

#[test]
fn conversion_and_interpolation_signatures_are_visible_to_lsp() {
    for (name, source) in [
        ("int", "syms x; f=int(x);"),
        ("int2str", "s=int2str(uint64(1));"),
        ("int32", "y=int32(uint64(1));"),
        ("integral", "q=integral(@sin,uint16(0),uint16(1));"),
        (
            "interp1",
            "q=interp1(uint16([1 2]),uint16([3 4]),uint16(1));",
        ),
        (
            "interp1q",
            "q=interp1q(uint16([1 2]),uint16([3 4]),uint16(1));",
        ),
        (
            "interp2",
            "q=interp2(uint16([1 2;3 4]),uint16(1),uint16(1));",
        ),
        ("intmax", "x=intmax('like',uint64(1));"),
        ("intmin", "x=intmin('like',int64(1));"),
        ("inv", "x=inv(uint16([1 0;0 1]));"),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}
