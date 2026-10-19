//! Compiled runtime evidence for the L15 color bindings, in one library.
use runmat_builtins::{
    builtin_function_by_name, BuiltinFusionPolicy, GRAY2RGB_CATALOG_ENTRY, HSV2RGB_CATALOG_ENTRY,
    IND2RGB_CATALOG_ENTRY, LAB2RGB_CATALOG_ENTRY, RGB2GRAY_CATALOG_ENTRY, RGB2HSV_CATALOG_ENTRY,
    RGB2LAB_CATALOG_ENTRY,
};
use runmat_runtime::builtin::migration_inventory::migration_inventory;
use runmat_runtime::builtins::common::spec::canonical_fusion_declaration;

#[test]
fn color_calls_have_one_exact_canonical_binding_and_no_legacy_registration() {
    let inventory = migration_inventory();
    for entry in [
        &GRAY2RGB_CATALOG_ENTRY,
        &HSV2RGB_CATALOG_ENTRY,
        &IND2RGB_CATALOG_ENTRY,
        &LAB2RGB_CATALOG_ENTRY,
        &RGB2GRAY_CATALOG_ENTRY,
        &RGB2HSV_CATALOG_ENTRY,
        &RGB2LAB_CATALOG_ENTRY,
    ] {
        let name = entry.identity.name;
        let bindings = runmat_runtime::builtin::runtime_builtin_bindings_by_name(name);
        assert_eq!(entry.bindings.len(), 1, "{name}");
        assert_eq!(bindings.len(), 1, "{name}");
        assert_eq!(
            bindings[0].identity,
            entry.binding_identity(&entry.bindings[0])
        );
        assert_eq!(bindings[0].identity.variant, "default");
        assert!(
            builtin_function_by_name(name).is_none(),
            "{name} legacy function"
        );
        assert_eq!(entry.placement.fusion, BuiltinFusionPolicy::Boundary);
        assert!(canonical_fusion_declaration(bindings[0].identity).is_none());
        assert!(inventory
            .snapshot
            .observed
            .gpu_specs
            .iter()
            .all(|spec| spec.key != name));
        assert!(inventory
            .snapshot
            .observed
            .fusion_specs
            .iter()
            .all(|spec| spec.key != name));
        let errors = runmat_runtime::builtin::validate_runtime_builtin_registry()
            .into_iter()
            .filter(|error| {
                error.identity == name || error.identity.starts_with(&format!("{name}#"))
            })
            .collect::<Vec<_>>();
        assert!(errors.is_empty(), "{name}: {errors:#?}");
    }
}
