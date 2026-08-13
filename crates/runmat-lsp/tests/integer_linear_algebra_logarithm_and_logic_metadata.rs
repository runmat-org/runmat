use lsp_types::Position;
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const CAPABILITY_NAMES: [&str; 15] = [
    "kron", "ldivide", "le", "length", "linprog", "linsolve", "linspace", "log", "log10", "log1p",
    "log2", "logical", "logspace", "lt", "lu",
];

#[test]
fn linear_algebra_logarithm_and_logic_integer_metadata_is_explicit() {
    for name in CAPABILITY_NAMES {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(!builtin.integer_capabilities.is_empty(), "{name}");
        assert!(builtin.integer_audit.is_none(), "{name}");
    }
}

#[test]
fn linear_algebra_logarithm_and_logic_signatures_are_visible_to_lsp() {
    for (name, source) in [
        ("kron", "y=kron(uint8(1),uint8(2));"),
        ("ldivide", "y=ldivide(int16(2),int16(5));"),
        ("le", "y=le(uint64(1),uint64(2));"),
        ("length", "y=length(uint32([1 2]));"),
        ("linprog", "y=linprog(int8(-1),int8(1),int8(1));"),
        ("linsolve", "y=linsolve(int8(1),int8(1));"),
        ("linspace", "y=linspace(0,1,uint8(3));"),
        ("log", "y=log(uint8(2));"),
        ("log10", "y=log10(uint8(10));"),
        ("log1p", "y=log1p(uint8(1));"),
        ("log2", "y=log2(uint8(2));"),
        ("logical", "y=logical(uint8(1));"),
        ("logspace", "y=logspace(0,1,uint8(3));"),
        ("lt", "y=lt(uint64(1),uint64(2));"),
        ("lu", "[l,u]=lu(int8([1 0;0 1]));"),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}
