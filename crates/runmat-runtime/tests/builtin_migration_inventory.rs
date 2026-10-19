use runmat_runtime::builtin::migration_inventory::{
    migration_inventory, migration_inventory_json, MIGRATION_INVENTORY_SCHEMA_VERSION,
};

#[test]
fn production_linkage_emits_a_valid_deterministic_inventory() {
    let first = migration_inventory_json().expect("first compiled migration inventory");
    let second = migration_inventory_json().expect("second compiled migration inventory");
    assert_eq!(first, second);

    let inventory = migration_inventory();
    assert_eq!(inventory.schema_version, MIGRATION_INVENTORY_SCHEMA_VERSION);
    assert_eq!(
        inventory.snapshot.validation.status, "valid",
        "{:#?}",
        inventory.snapshot.validation.errors
    );
    assert!(inventory.snapshot.validation.errors.is_empty());
}

#[test]
fn internal_autodiff_callbacks_have_production_bindings_without_public_discovery() {
    use runmat_builtins::{
        BuiltinCatalogAudience, DEEP_LEARNING_NETWORK_GRAD_CATALOG_ENTRY,
        DEEP_LEARNING_SQUARE_GRAD_CATALOG_ENTRY, REQUIRED_DEFAULT_BINDING,
    };

    let bindings = runmat_runtime::builtin::runtime_builtin_bindings();
    for entry in [
        &DEEP_LEARNING_SQUARE_GRAD_CATALOG_ENTRY,
        &DEEP_LEARNING_NETWORK_GRAD_CATALOG_ENTRY,
    ] {
        assert_eq!(entry.audience, BuiltinCatalogAudience::Fixture);
        assert_eq!(entry.bindings, REQUIRED_DEFAULT_BINDING.as_slice());
        let identity = entry.binding_identity(&entry.bindings[0]);
        assert_eq!(
            bindings
                .iter()
                .filter(|binding| binding.identity == identity)
                .count(),
            1,
            "required production binding for {identity:?}"
        );
        assert!(
            runmat_builtins::public_builtin_catalog_entry_by_name(entry.identity.name).is_none()
        );
    }
}
