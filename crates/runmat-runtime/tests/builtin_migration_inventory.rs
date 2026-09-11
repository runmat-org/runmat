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
