use lsp_types::Position;
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

const CAPABILITY_NAMES: [&str; 9] = [
    "ipermute",
    "isUnderlyingType",
    "isa",
    "isequal",
    "isequaln",
    "isfinite",
    "isinf",
    // Typed numeric graphics aliases have an explicit canonical integer role.
    "isgraphics",
    "ishandle",
];

const INAPPLICABLE_NAMES: [&str; 16] = [
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
    "ishermitian",
    "isletter",
];

const FOLLOWUP_CAPABILITY_NAMES: [&str; 6] = [
    "ismissing",
    "isnan",
    "isnumeric",
    "isreal",
    "issparse",
    "isstrprop",
];

const FOLLOWUP_INAPPLICABLE_NAMES: [&str; 19] = [
    "islogical",
    "ismatrix",
    "ismethod",
    "isobject",
    "isordinal",
    "ispref",
    "isrow",
    "isscalar",
    "isspace",
    "isstable",
    "isstring",
    "issymmetric",
    "istable",
    "istimetable",
    "istril",
    "istriu",
    "isvalid",
    "isvarname",
    "isvector",
];

#[test]
fn type_structure_and_finiteness_integer_metadata_is_explicit() {
    for name in CAPABILITY_NAMES {
        let metadata = runmat_builtins::builtin_integer_metadata_by_name(name)
            .unwrap_or_else(|| panic!("integer metadata for {name}"));
        assert_eq!(metadata.capabilities.len(), 1, "{name}");
        assert!(metadata.audit.is_none(), "{name}");
    }
    for name in INAPPLICABLE_NAMES {
        let metadata = runmat_builtins::builtin_integer_metadata_by_name(name)
            .unwrap_or_else(|| panic!("integer metadata for {name}"));
        assert!(metadata.capabilities.is_empty(), "{name}");
        assert!(metadata.audit.is_some(), "{name}");
    }
    let ishandle = runmat_builtins::builtin_catalog_entry_by_name("ishandle").unwrap();
    let [capability] = ishandle.integer_capabilities else {
        panic!("one integer-alias role")
    };
    let [input] = capability.inputs else {
        panic!("one handle candidate")
    };
    assert_eq!(input.name, "h");
    assert_eq!(input.classes, &runmat_builtins::ALL_INTEGER_CLASSES);
    assert_eq!(
        input.availability,
        runmat_builtins::BuiltinIntegerInputAvailability::RunMatOnly
    );
    assert_eq!(
        input.scalar_double,
        runmat_builtins::BuiltinIntegerScalarDoubleRule::Allowed
    );
    assert_eq!(
        capability.computation_domain,
        runmat_builtins::BuiltinIntegerComputationDomain::Predicate
    );
    assert_eq!(
        capability.output_class,
        runmat_builtins::BuiltinIntegerOutputClassRule::Logical
    );
    assert_eq!(
        capability.overflow,
        runmat_builtins::BuiltinIntegerOverflowRule::NotApplicable
    );
    assert_eq!(
        capability.backend,
        runmat_builtins::BuiltinIntegerBackendRule::HostOnly
    );
    assert_eq!(
        capability.overload,
        runmat_builtins::BuiltinIntegerOverloadKind::ElementwiseShapePreserving
    );
    assert!(input
        .notes
        .contains("Disabled extensions and nonrepresentable aliases return false"));
    assert!(capability.notes.contains("no gather or provider work"));
    let [extension] = ishandle.extensions else {
        panic!("one integer-alias extension")
    };
    assert_eq!(extension.id, "ishandle-integer-handle-alias");
    assert_eq!(
        extension.mode,
        runmat_builtins::BuiltinExtensionMode::RunMatOnly
    );
    assert_eq!(
        extension.error_identifier,
        Some("RunMat:compatibility:IshandleIntegerHandleExtension")
    );
}

#[test]
fn followup_integer_predicate_metadata_is_explicit() {
    for name in FOLLOWUP_CAPABILITY_NAMES {
        let metadata = runmat_builtins::builtin_integer_metadata_by_name(name)
            .unwrap_or_else(|| panic!("integer metadata for {name}"));
        assert!(!metadata.capabilities.is_empty(), "{name}");
        assert!(metadata.audit.is_none(), "{name}");
    }
    for name in FOLLOWUP_INAPPLICABLE_NAMES {
        let metadata = runmat_builtins::builtin_integer_metadata_by_name(name)
            .unwrap_or_else(|| panic!("integer metadata for {name}"));
        assert!(metadata.capabilities.is_empty(), "{name}");
        assert!(metadata.audit.is_some(), "{name}");
    }
}

#[test]
fn followup_integer_predicate_signatures_are_visible_to_lsp() {
    for (name, source) in [
        ("ismissing", "tf=ismissing(uint64(1));"),
        ("isnan", "tf=isnan(uint64(1));"),
        ("isnumeric", "tf=isnumeric(uint64(1));"),
        ("isreal", "tf=isreal(uint64(1));"),
        ("issparse", "tf=issparse(uint64(1));"),
        ("isstrprop", "tf=isstrprop(uint16(65),'alpha');"),
        ("isvector", "tf=isvector(uint64([1 2]));"),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        assert!(!help.signatures.is_empty(), "{name}");
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
