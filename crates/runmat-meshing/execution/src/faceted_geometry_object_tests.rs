use runmat_execution::value::ValueRefKind;
use runmat_execution_artifact::object::ObjectInventoryLimits;

use crate::tests::MemoryCache;
use crate::{
    import_faceted_geometry_input, import_faceted_geometry_objects, prepare_faceted_geometry_input,
    prepare_faceted_geometry_objects, MeshingArtifactAccess,
};

#[test]
fn faceted_geometry_round_trips_through_shared_input_objects() {
    let (document, solid) = runmat_geometry_fixtures::faceted_tetrahedron();
    let prepared =
        prepare_faceted_geometry_objects(document, solid, ObjectInventoryLimits::default())
            .unwrap();
    let mut cache = MemoryCache::default();
    cache.insert_all(&prepared.objects);

    let imported = import_faceted_geometry_objects(
        &cache,
        prepared.document.clone(),
        prepared.root_reference(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();

    assert_eq!(imported, prepared);
    assert_eq!(imported.objects.len(), 1);
    assert_eq!(
        imported.objects[0].descriptor.namespace,
        runmat_execution_artifact::ObjectNamespace::InputValue
    );
}

#[test]
fn faceted_geometry_rehashes_cache_and_enforces_inventory_limits() {
    let (document, solid) = runmat_geometry_fixtures::faceted_tetrahedron();
    let prepared =
        prepare_faceted_geometry_objects(document, solid, ObjectInventoryLimits::default())
            .unwrap();
    let mut cache = MemoryCache::default();
    cache.insert_all(&prepared.objects);
    cache.replace(prepared.root.digest, b"poisoned".to_vec());
    assert!(import_faceted_geometry_objects(
        &cache,
        prepared.document.clone(),
        prepared.root_reference(),
        ObjectInventoryLimits::default(),
    )
    .is_err());
    assert!(prepare_faceted_geometry_objects(
        prepared.document,
        prepared.solid,
        ObjectInventoryLimits {
            max_objects: 0,
            ..ObjectInventoryLimits::default()
        },
    )
    .is_err());
}

#[test]
fn faceted_geometry_projects_to_one_driver_owned_input() {
    let (document, solid) = runmat_geometry_fixtures::faceted_tetrahedron();
    let objects =
        prepare_faceted_geometry_objects(document, solid, ObjectInventoryLimits::default())
            .unwrap();
    let access = MeshingArtifactAccess {
        authorization_scope: "faceted-geometry-run".into(),
        encryption_context: runmat_execution::Digest::sha256(b"faceted-geometry-context"),
    };
    let prepared =
        prepare_faceted_geometry_input(objects, access.clone(), ObjectInventoryLimits::default())
            .unwrap();
    assert_eq!(prepared.input_objects().len(), 1);
    assert_eq!(prepared.root_input().kind, ValueRefKind::DriverObject);

    let mut cache = MemoryCache::default();
    cache.insert_all(&prepared.geometry_objects().objects);
    let imported = import_faceted_geometry_input(
        &cache,
        prepared.geometry_objects().document.clone(),
        prepared.root_input(),
        access.clone(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(imported, prepared);

    let mut wrong = prepared.root_input().clone();
    wrong.authorization_scope = "wrong-run".into();
    assert!(import_faceted_geometry_input(
        &cache,
        prepared.geometry_objects().document.clone(),
        &wrong,
        access,
        ObjectInventoryLimits::default(),
    )
    .is_err());
}
