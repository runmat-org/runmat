//! One-runtime-library binding evidence for triangulation identities.
use runmat_builtins::{
    builtin_function_by_name, BuiltinFusionPolicy, DELAUNAY_TRI_CATALOG_ENTRY,
    DELAUNAY_TRI_FREE_BOUNDARY_CATALOG_ENTRY, DELAUNAY_TRI_NEAREST_NEIGHBOR_CATALOG_ENTRY,
    DELAUNAY_TRI_POINT_LOCATION_CATALOG_ENTRY, FREE_BOUNDARY_CATALOG_ENTRY,
    NEAREST_NEIGHBOR_CATALOG_ENTRY, POINT_LOCATION_CATALOG_ENTRY,
};
use runmat_runtime::builtin::migration_inventory::migration_inventory;
use runmat_runtime::builtins::common::spec::canonical_fusion_declaration;

#[test]
fn free_boundary_function_and_object_method_have_distinct_canonical_bindings() {
    let inventory = migration_inventory();
    for entry in [
        &DELAUNAY_TRI_CATALOG_ENTRY,
        &FREE_BOUNDARY_CATALOG_ENTRY,
        &DELAUNAY_TRI_FREE_BOUNDARY_CATALOG_ENTRY,
        &NEAREST_NEIGHBOR_CATALOG_ENTRY,
        &DELAUNAY_TRI_NEAREST_NEIGHBOR_CATALOG_ENTRY,
        &POINT_LOCATION_CATALOG_ENTRY,
        &DELAUNAY_TRI_POINT_LOCATION_CATALOG_ENTRY,
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
    }
}
