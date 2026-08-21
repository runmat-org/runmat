use lsp_types::Position;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinIntegerAuditKind, BuiltinIntegerBackendRule,
};
use runmat_lsp::core::analysis::{analyze_document_with_compat, completion_at, CompatMode};

const CAPABILITY_NAMES: [&str; 10] = [
    "hold",
    "line",
    "loglog",
    "plot",
    "plot3",
    "plotmatrix",
    "plotyy",
    "scatter",
    "semilogx",
    "semilogy",
];

fn initialize_lsp_builtin_registry() {
    let source = "x = 1;";
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    let completions = completion_at(source, &analysis, &Position::new(0, 0));
    assert!(!completions.is_empty());
}

#[test]
fn integer_plotting_metadata_is_public_settled_and_backend_explicit() {
    initialize_lsp_builtin_registry();
    for name in CAPABILITY_NAMES {
        let builtin = runmat_builtins::builtin_function_by_name(name)
            .unwrap_or_else(|| panic!("registered builtin {name}"));
        assert_eq!(
            builtin.descriptor.unwrap().completion_policy,
            BuiltinCompletionPolicy::Public
        );
        assert!(!builtin.integer_capabilities.is_empty(), "{name}");
        assert!(builtin.integer_audit.is_none(), "{name}");
        assert!(
            builtin
                .integer_capabilities
                .iter()
                .all(|capability| !capability.notes.contains("[integer-audit-open]")),
            "{name}"
        );
        if name == "hold" {
            assert_eq!(
                builtin.integer_capabilities[0].backend,
                BuiltinIntegerBackendRule::HostOnly
            );
        } else {
            assert_eq!(
                builtin.integer_capabilities[0].backend,
                BuiltinIntegerBackendRule::GatherFallback
            );
        }
    }
}

#[test]
fn legend_is_explicitly_integer_inapplicable() {
    initialize_lsp_builtin_registry();
    let builtin = runmat_builtins::builtin_function_by_name("legend").expect("legend builtin");
    assert!(builtin.integer_capabilities.is_empty());
    assert_eq!(
        builtin.integer_audit.expect("legend integer audit").kind,
        BuiltinIntegerAuditKind::NotApplicable
    );
}
