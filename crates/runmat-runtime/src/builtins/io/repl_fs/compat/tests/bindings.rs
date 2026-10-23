#[test]
fn file_attribute_and_rehash_bindings_match_canonical_default_identity() {
    for name in ["fileattrib", "rehash", "winqueryreg"] {
        let catalog = runmat_builtins::builtin_catalog_entry_by_name(name)
            .unwrap_or_else(|| panic!("missing canonical {name} entry"));
        assert_eq!(catalog.bindings, &runmat_builtins::REQUIRED_DEFAULT_BINDING);
        let runtime = crate::builtin::runtime_builtin_bindings_by_name(name);
        assert_eq!(runtime.len(), 1, "{name} must have one runtime binding");
        assert_eq!(
            runtime[0].identity,
            catalog.binding_identity(&catalog.bindings[0])
        );
    }
}
