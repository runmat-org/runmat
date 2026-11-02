mod support;
#[path = "support/hover.rs"]
mod hover_support;

use lsp_types::{HoverContents, Position};
use runmat_lsp::core::analysis::{
    analyze_document_with_compat, completion_at, hover_at, signature_help_at, CompatMode,
};

const ECDF_TO_ERASEBETWEEN_BUILTINS: [&str; 10] = [
    "ecdf",
    "eig",
    "eigs",
    "empty",
    "encode",
    "endsWith",
    "envelope",
    "eq",
    "erase",
    "eraseBetween",
];

const INTEGER_EMPTY_BUILTINS: [&str; 8] = [
    "int8.empty",
    "int16.empty",
    "int32.empty",
    "int64.empty",
    "uint8.empty",
    "uint16.empty",
    "uint32.empty",
    "uint64.empty",
];

#[test]
fn ecdf_to_erasebetween_descriptors_are_visible_to_signature_help_and_hover() {
    for (name, source) in [
        ("ecdf", "[f, x] = ecdf(uint8([1 2]));"),
        ("eig", "d = eig(uint8(1));"),
        ("eigs", "d = eigs(uint8(1));"),
        ("empty", "a = empty(0);"),
        ("encode", "counts = encode(1, 'word');"),
        ("endsWith", "tf = endsWith('runmat', 'mat');"),
        ("envelope", "upper = envelope(uint8([1 2]));"),
        ("eq", "tf = eq(uint8(1), uint8(1));"),
        ("erase", "out = erase('runmat', 'run');"),
        ("eraseBetween", "out = eraseBetween('runmat', 1, 3);"),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        assert!(analysis.compile_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin call") as u32);
        let help =
            signature_help_at(source, &analysis, &position).expect("canonical signature help");
        support::assert_catalog_signatures(name, &help);
        let hover = hover_at(source, &analysis, &position).expect("canonical hover");
        let HoverContents::Markup(markup) = hover.contents else {
            panic!("expected Markdown hover");
        };
        hover_support::assert_catalog_hover(name, &help, &markup.value);
    }
}
#[test]
fn ecdf_to_erasebetween_builtins_are_public_completions_with_integer_metadata() {
    let source = "x = 1;";
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    let completions = completion_at(source, &analysis, &Position::new(0, 0));

    for name in ECDF_TO_ERASEBETWEEN_BUILTINS {
        let builtin =
            runmat_builtins::builtin_catalog_entry_by_name(name).expect("canonical builtin");
        let descriptor = builtin.descriptor;
        assert_eq!(
            descriptor.completion_policy,
            runmat_builtins::BuiltinCompletionPolicy::Public,
            "{name} must remain a public LSP completion"
        );
        assert!(
            !descriptor.signatures.is_empty(),
            "{name} must expose descriptor-backed signatures"
        );
        assert!(
            !builtin.integer_capabilities.is_empty(),
            "{name} must expose its settled integer capability records"
        );

        let completion = completions
            .iter()
            .find(|item| item.label.eq_ignore_ascii_case(name))
            .unwrap_or_else(|| panic!("public completion for {name}"));
        let detail = completion.detail.as_deref().unwrap_or_default();
        assert!(
            detail.contains(&format!("{name}(")),
            "{name} completion detail must be rendered from its descriptor, got {detail:?}"
        );
    }
}

#[test]
fn integer_empty_static_registrations_follow_the_public_completion_policy() {
    let source = "x = 1;";
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    let completions = completion_at(source, &analysis, &Position::new(0, 0));

    for name in INTEGER_EMPTY_BUILTINS {
        let builtin = runmat_builtins::builtin_catalog_entry_by_name(name)
            .unwrap_or_else(|| panic!("canonical static builtin {name}"));
        let descriptor = builtin.descriptor;
        assert_eq!(
            descriptor.completion_policy,
            runmat_builtins::BuiltinCompletionPolicy::Public,
            "{name} inherits the public empty completion policy"
        );
        assert!(
            !builtin.integer_capabilities.is_empty(),
            "{name} must expose the empty integer capability"
        );
        assert!(
            builtin.extensions.iter().any(|extension| {
                extension.id == "empty-resident-size"
                    && extension.mode == runmat_builtins::BuiltinExtensionMode::RunMatOnly
            }),
            "{name} must expose the resident-size extension"
        );

        let completion = completions
            .iter()
            .find(|item| item.label.eq_ignore_ascii_case(name))
            .unwrap_or_else(|| panic!("public completion for {name}"));
        let detail = completion.detail.as_deref().unwrap_or_default();
        assert!(
            detail.contains(&format!("{name}(")),
            "{name} completion detail must be rendered from the shared empty descriptor, got {detail:?}"
        );
    }
}

#[test]
fn matlab_mode_keeps_ecdf_to_erasebetween_extension_metadata_visible() {
    for (name, extension_id) in [
        ("ecdf", "ecdf-integer-y"),
        ("eig", "eig-nonfloating-coefficient"),
        ("eigs", "eigs-nonfloating-matrix"),
        ("empty", "empty-global-call"),
        ("encode", "encode-numeric-force-cell-output"),
        ("endsWith", "endswith-numeric-ignore-case"),
        ("envelope", "envelope-integer-data"),
        ("eraseBetween", "erasebetween-resident-position"),
    ] {
        let source = format!("x = {name}(uint8(1));");
        let analysis = analyze_document_with_compat(&source, CompatMode::Matlab);
        assert!(analysis.syntax_error.is_none(), "{source}");

        let builtin =
            runmat_builtins::builtin_catalog_entry_by_name(name).expect("canonical builtin");
        assert!(
            builtin.extensions.iter().any(|extension| {
                extension.id == extension_id
                    && extension.mode == runmat_builtins::BuiltinExtensionMode::RunMatOnly
            }),
            "{name} must expose {extension_id} to LSP consumers"
        );
    }

    {
        let name = "eq";
        let builtin =
            runmat_builtins::builtin_catalog_entry_by_name(name).expect("canonical builtin");
        assert!(
            builtin.extensions.is_empty(),
            "{name} has settled integer capability metadata but no compatibility extension"
        );
    }
    let erase = runmat_builtins::builtin_catalog_entry_by_name("erase").unwrap();
    let actual = erase
        .extensions
        .iter()
        .map(|extension| {
            assert_eq!(
                extension.mode,
                runmat_builtins::BuiltinExtensionMode::RunMatOnly
            );
            (extension.id, extension.error_identifier)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        actual,
        [
            (
                "erase-char-matrix",
                Some("RunMat:compatibility:EraseCharMatrixExtension")
            ),
            (
                "erase-string-cell",
                Some("RunMat:compatibility:EraseStringCellExtension")
            ),
        ]
    );
    let [capability] = erase.integer_capabilities else {
        panic!("erase has one rejected-input contract")
    };
    assert_eq!(
        capability.backend,
        runmat_builtins::BuiltinIntegerBackendRule::HostOnly
    );
    assert_eq!(
        capability.computation_domain,
        runmat_builtins::BuiltinIntegerComputationDomain::FunctionSpecific
    );
    assert_eq!(
        capability.overload,
        runmat_builtins::BuiltinIntegerOverloadKind::FunctionSpecific
    );
    assert_eq!(
        capability.output_class,
        runmat_builtins::BuiltinIntegerOutputClassRule::NotApplicable
    );
    assert_eq!(
        capability.overflow,
        runmat_builtins::BuiltinIntegerOverflowRule::NotApplicable
    );
    assert_eq!(
        capability
            .inputs
            .iter()
            .map(|input| input.name)
            .collect::<Vec<_>>(),
        ["str", "match"]
    );
    for input in capability.inputs {
        assert_eq!(input.classes, &runmat_builtins::ALL_INTEGER_CLASSES);
        assert_eq!(
            input.availability,
            runmat_builtins::BuiltinIntegerInputAvailability::Rejected
        );
        assert_eq!(
            input.scalar_double,
            runmat_builtins::BuiltinIntegerScalarDoubleRule::Rejected
        );
    }
    assert!(capability.notes.contains("rejected without gathering"));
}
