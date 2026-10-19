//! Production-linked authority checks for interop and migrated I/O families.

use runmat_builtins::{
    builtin_catalog_entries, builtin_function_by_name, BuiltinCatalogEntry, CALLLIB_CATALOG_ENTRY,
    H5DISP_CATALOG_ENTRY, H5INFO_CATALOG_ENTRY, H5READ_CATALOG_ENTRY, H5WRITEATT_CATALOG_ENTRY,
    H5WRITE_CATALOG_ENTRY, HDF5INFO_CATALOG_ENTRY, HDF5READ_CATALOG_ENTRY, HDF5WRITE_CATALOG_ENTRY,
    JSONDECODE_CATALOG_ENTRY, JSONENCODE_CATALOG_ENTRY, LIBFUNCTIONS_CATALOG_ENTRY,
    LIBISLOADED_CATALOG_ENTRY, LIBPOINTER_CATALOG_ENTRY, LIBSTRUCT_CATALOG_ENTRY,
    LOADLIBRARY_CATALOG_ENTRY, SETDATATYPE_CATALOG_ENTRY, UNLOADLIBRARY_CATALOG_ENTRY,
    WEBOPTIONS_CATALOG_ENTRY,
};
use runmat_builtins::{DELETE_CATALOG_ENTRY, IMPORTDATA_CATALOG_ENTRY, TEXTSCAN_CATALOG_ENTRY};
use runmat_builtins::{
    DIARY_CATALOG_ENTRY, DISPLAY_CATALOG_ENTRY, DISP_CATALOG_ENTRY, FORMAT_CATALOG_ENTRY,
    INPUT_CATALOG_ENTRY,
};
use runmat_builtins::{
    LOAD_CATALOG_ENTRY, MATFILE_CATALOG_ENTRY, MATFILE_SUBSASGN_CATALOG_ENTRY,
    MATFILE_SUBSREF_CATALOG_ENTRY, SAVE_CATALOG_ENTRY,
};
use runmat_builtins::{READSTRUCT_CATALOG_ENTRY, XMLREAD_CATALOG_ENTRY, XMLWRITE_CATALOG_ENTRY};
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
        &WEBOPTIONS_CATALOG_ENTRY,
        &H5DISP_CATALOG_ENTRY,
        &H5INFO_CATALOG_ENTRY,
        &H5READ_CATALOG_ENTRY,
        &H5WRITE_CATALOG_ENTRY,
        &H5WRITEATT_CATALOG_ENTRY,
        &HDF5INFO_CATALOG_ENTRY,
        &HDF5READ_CATALOG_ENTRY,
        &HDF5WRITE_CATALOG_ENTRY,
        &DIARY_CATALOG_ENTRY,
        &DISP_CATALOG_ENTRY,
        &DISPLAY_CATALOG_ENTRY,
        &FORMAT_CATALOG_ENTRY,
        &INPUT_CATALOG_ENTRY,
        &LOAD_CATALOG_ENTRY,
        &SAVE_CATALOG_ENTRY,
        &MATFILE_CATALOG_ENTRY,
        &MATFILE_SUBSREF_CATALOG_ENTRY,
        &MATFILE_SUBSASGN_CATALOG_ENTRY,
        &READSTRUCT_CATALOG_ENTRY,
        &XMLREAD_CATALOG_ENTRY,
        &XMLWRITE_CATALOG_ENTRY,
        &DELETE_CATALOG_ENTRY,
        &IMPORTDATA_CATALOG_ENTRY,
        &TEXTSCAN_CATALOG_ENTRY,
    ];
    // This is an inventory scope, not runtime class or operation dispatch.
    let data_entries: Vec<_> = builtin_catalog_entries()
        .iter()
        .copied()
        .filter(|entry| {
            ["data.", "Dataset.", "DataArray.", "DataTransaction."]
                .iter()
                .any(|prefix| entry.identity.name.starts_with(prefix))
                || entry.identity.name == "commit"
        })
        .collect();
    assert_eq!(data_entries.len(), 45, "complete Data cutover inventory");
    assert!(
        inventory
            .snapshot
            .observed
            .gpu_specs
            .iter()
            .all(|spec| spec.key != "data.*"),
        "obsolete wildcard Data GPU declaration"
    );
    assert!(
        inventory
            .snapshot
            .observed
            .fusion_specs
            .iter()
            .all(|spec| spec.key != "data.*"),
        "obsolete wildcard Data fusion declaration"
    );
    for entry in entries.iter().copied().chain(data_entries) {
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
