use runmat_runtime::builtins::common::spec::{
    canonical_fusion_declaration, FusionStructuralOperation,
};

#[test]
fn matrix_product_declaration_joins_exact_compiled_binding_not_child_spec_module() {
    let entry = runmat_builtins::builtin_catalog_entry_by_name("mtimes").unwrap();
    let identity = entry.binding_identity(&entry.bindings[0]);
    let declaration =
        canonical_fusion_declaration(identity).expect("compiled structural authority");
    assert_eq!(declaration.binding, identity);
    assert_eq!(declaration.semantics.unwrap().binding, identity);
    assert_eq!(
        declaration.structural,
        Some(FusionStructuralOperation::MatrixMultiply)
    );
    assert_eq!(declaration.structural.unwrap().input_arity(), 2);
    assert!(declaration.spec.elementwise.is_none());
    assert!(declaration.spec.reduction.is_none());
    let implementations = runmat_builtins::builtin_implementation_provenance();
    let implementation = implementations
        .iter()
        .find(|item| item.name == "mtimes" && item.binding_variant == Some("default"))
        .unwrap();
    assert_eq!(
        implementation.builtin_path,
        "crate::builtins::math::linalg::ops::matrix_arithmetic::mtimes"
    );
    assert!(
        canonical_fusion_declaration(runmat_builtins::BuiltinBindingIdentity {
            variant: "unregistered",
            ..identity
        })
        .is_none()
    );
}
