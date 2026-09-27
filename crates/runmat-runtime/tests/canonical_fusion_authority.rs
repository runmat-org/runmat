//! A single runtime library owns the compiled implementation inventory.
use runmat_builtins::{builtin_catalog_entry_by_name, BuiltinBindingIdentity};
use runmat_runtime::builtins::common::spec::canonical_fusion_declaration;

#[test]
fn selected_binding_cannot_borrow_another_variants_expression() {
    for name in ["single", "double", "power"] {
        let entry = builtin_catalog_entry_by_name(name).expect("canonical entry");
        let identity = entry.binding_identity(&entry.bindings[0]);
        let declaration = canonical_fusion_declaration(identity).expect("selected declaration");
        assert_eq!(declaration.binding, identity);
        let semantics = declaration.semantics.expect("typed semantics");
        assert_eq!(semantics.binding, identity);
        assert_eq!(semantics.revision, 1);
        assert!(canonical_fusion_declaration(BuiltinBindingIdentity {
            variant: "not-the-selected-binding",
            ..identity
        })
        .is_none());
    }
}
