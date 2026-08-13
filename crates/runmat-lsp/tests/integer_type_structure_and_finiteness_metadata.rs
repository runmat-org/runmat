use lsp_types::Position;
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const CAPABILITY_NAMES: [&str; 7] = [
    "ipermute",
    "isUnderlyingType",
    "isa",
    "isequal",
    "isequaln",
    "isfinite",
    "isinf",
];

const INAPPLICABLE_NAMES: [&str; 18] = [
    "isStringScalar",
    "isVocabularyWord",
    "iscategorical",
    "iscell",
    "iscellstr",
    "ischar",
    "iscolumn",
    "isdeployed",
    "isdiag",
    "isempty",
    "isenv",
    "isfield",
    "isfile",
    "isfolder",
    "isgraphics",
    "ishandle",
    "ishermitian",
    "isletter",
];

#[test]
fn type_structure_and_finiteness_integer_metadata_is_explicit() {
    for name in CAPABILITY_NAMES {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert_eq!(builtin.integer_capabilities.len(), 1, "{name}");
        assert!(builtin.integer_audit.is_none(), "{name}");
    }
    for name in INAPPLICABLE_NAMES {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(builtin.integer_capabilities.is_empty(), "{name}");
        assert!(builtin.integer_audit.is_some(), "{name}");
    }
}

#[test]
fn type_structure_and_finiteness_signatures_are_visible_to_lsp() {
    for (name, source) in [
        (
            "ipermute",
            "y=ipermute(uint16(reshape(1:8,[2 2 2])),[2 1 3]);",
        ),
        (
            "isUnderlyingType",
            "tf=isUnderlyingType(uint64(1),'uint64');",
        ),
        ("isa", "tf=isa(uint64(1),'integer');"),
        ("isequal", "tf=isequal(uint64(1),uint64(1));"),
        ("isequaln", "tf=isequaln(uint64(1),uint64(1));"),
        ("isfinite", "tf=isfinite(uint64(1));"),
        ("isinf", "tf=isinf(uint64(1));"),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
    }
}
