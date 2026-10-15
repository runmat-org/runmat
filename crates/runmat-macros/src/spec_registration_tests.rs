use super::RegisterSpecAttrArgs;

#[test]
fn old_spec_declarations_have_no_structural_operation() {
    let args: RegisterSpecAttrArgs = syn::parse_quote!(builtin_path = "crate::operation");
    assert!(args.structural.is_none());
    assert!(args.semantics.is_none());
    assert!(args.epilogue.is_none());
    assert!(args.reduction.is_none());
    assert!(args.host_resources.is_none());
}

#[test]
fn typed_reduction_declaration_is_optional_and_unique() {
    let args: RegisterSpecAttrArgs =
        syn::parse_quote!(builtin_path = "crate::operation", reduction = REDUCTION);
    assert!(matches!(args.reduction, Some(syn::Expr::Path(_))));
    assert!(syn::parse_str::<RegisterSpecAttrArgs>("reduction = A, reduction = B").is_err());
}

#[test]
fn physical_host_resource_declaration_is_optional_and_unique() {
    let args: RegisterSpecAttrArgs = syn::parse_quote!(
        builtin_path = "crate::operation",
        host_resources = owner::checked
    );
    assert!(matches!(args.host_resources, Some(syn::Expr::Path(_))));
    assert!(
        syn::parse_str::<RegisterSpecAttrArgs>("host_resources = A, host_resources = B").is_err()
    );
}

#[test]
fn epilogue_declaration_is_typed_optional_and_unique() {
    let args: RegisterSpecAttrArgs = syn::parse_quote!(
        builtin_path = "crate::operation",
        epilogue = FusionEpilogueOperation::Diagonal
    );
    assert!(matches!(args.epilogue, Some(syn::Expr::Path(_))));
    assert!(syn::parse_str::<RegisterSpecAttrArgs>(
        "epilogue = Operation::A, epilogue = Operation::B"
    )
    .is_err());
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
