use super::RegisterSpecAttrArgs;

#[test]
fn old_spec_declarations_have_no_structural_operation() {
    let args: RegisterSpecAttrArgs = syn::parse_quote!(builtin_path = "crate::operation");
    assert!(args.structural.is_none());
    assert!(args.semantics.is_none());
}

#[test]
fn typed_structural_declaration_is_optional_and_preserves_binding_semantics() {
    let args: RegisterSpecAttrArgs = syn::parse_quote!(
        builtin_path = "crate::operation",
        semantics = SEMANTICS,
        structural = FusionStructuralOperation::MatrixMultiply
    );
    assert!(matches!(args.structural, Some(syn::Expr::Path(_))));
    assert!(args.semantics.is_some());
}

#[test]
fn duplicate_and_unknown_structural_attributes_are_rejected() {
    assert!(syn::parse_str::<RegisterSpecAttrArgs>(
        "structural = Operation::A, structural = Operation::B"
    )
    .is_err());
    assert!(syn::parse_str::<RegisterSpecAttrArgs>("structural_kind = Operation::A").is_err());
}
