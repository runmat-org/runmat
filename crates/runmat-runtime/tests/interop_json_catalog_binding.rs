//! Production-linked authority checks for the native FFI and JSON cutovers.

use runmat_builtins::{
    builtin_function_by_name, BuiltinCatalogEntry, CALLLIB_CATALOG_ENTRY, JSONDECODE_CATALOG_ENTRY,
    JSONENCODE_CATALOG_ENTRY, LIBFUNCTIONS_CATALOG_ENTRY, LIBISLOADED_CATALOG_ENTRY,
    LIBPOINTER_CATALOG_ENTRY, LIBSTRUCT_CATALOG_ENTRY, LOADLIBRARY_CATALOG_ENTRY,
    SETDATATYPE_CATALOG_ENTRY, UNLOADLIBRARY_CATALOG_ENTRY,
};
use runmat_runtime::builtin::migration_inventory::migration_inventory;

#[test]
fn migrated_identities_have_one_production_binding_and_no_legacy_authority() {
    let inventory = migration_inventory();
    let entries: &[&BuiltinCatalogEntry] = &[
        &CALLLIB_CATALOG_ENTRY,
        &LIBFUNCTIONS_CATALOG_ENTRY,
        &LIBISLOADED_CATALOG_ENTRY,
        &LIBPOINTER_CATALOG_ENTRY,
        &LIBSTRUCT_CATALOG_ENTRY,
        &LOADLIBRARY_CATALOG_ENTRY,
        &SETDATATYPE_CATALOG_ENTRY,
        &UNLOADLIBRARY_CATALOG_ENTRY,
        &JSONDECODE_CATALOG_ENTRY,
        &JSONENCODE_CATALOG_ENTRY,
    ];
    for entry in entries {
        let name = entry.identity.name;
        let bindings = runmat_runtime::builtin::runtime_builtin_bindings_by_name(name);
        assert_eq!(entry.bindings.len(), 1, "{name}");
        assert_eq!(bindings.len(), 1, "{name}");
        assert_eq!(
            bindings[0].identity,
            entry.binding_identity(&entry.bindings[0]),
            "{name}",
        );
        assert_eq!(bindings[0].identity.variant, "default", "{name}");
        assert!(
            builtin_function_by_name(name).is_none(),
            "{name} legacy binding"
        );
        assert!(
            inventory
                .snapshot
                .observed
                .gpu_specs
                .iter()
                .all(|spec| spec.key != name),
            "{name} legacy GPU declaration"
        );
        assert!(
            inventory
                .snapshot
                .observed
                .fusion_specs
                .iter()
                .all(|spec| spec.key != name),
            "{name} legacy fusion declaration"
        );
    }
    assert_eq!(
        inventory.snapshot.validation.status, "valid",
        "{:#?}",
        inventory.snapshot.validation.errors,
    );
}
