mod fixture;

use runmat_execution_artifact::object::ObjectInventoryLimits;

use crate::tests::MemoryCache;
use crate::{import_exact_geometry_objects, prepare_exact_geometry_objects};

#[test]
fn exact_geometry_round_trips_through_shared_input_objects() {
    let (document, topology, evaluators) = fixture::geometry();
    let prepared = prepare_exact_geometry_objects(
        document,
        topology,
        evaluators,
        None,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let mut cache = MemoryCache::default();
    cache.insert_all(&prepared.objects);
    let imported = import_exact_geometry_objects(
        &cache,
        prepared.document.clone(),
        prepared.root_reference(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();

    assert_eq!(imported, prepared);
    assert!(imported.objects.iter().all(|object| {
        object.descriptor.namespace == runmat_execution_artifact::ObjectNamespace::InputValue
            && object
                .descriptor
                .logical_name
                .starts_with("geometry/canonical/")
    }));
}

#[test]
fn exact_geometry_import_rehashes_cache_and_binds_document_root() {
    let (document, topology, evaluators) = fixture::geometry();
    let prepared = prepare_exact_geometry_objects(
        document,
        topology,
        evaluators,
        None,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let mut cache = MemoryCache::default();
    cache.insert_all(&prepared.objects);
    cache.replace(prepared.root.digest, b"poisoned".to_vec());
    assert!(import_exact_geometry_objects(
        &cache,
        prepared.document.clone(),
        prepared.root_reference(),
        ObjectInventoryLimits::default(),
    )
    .is_err());

    let mut wrong_root_document = prepared.document.clone();
    let runmat_geometry_core::GeometryModel::ExactBRep { model } = &mut wrong_root_document.model
    else {
        unreachable!()
    };
    model.artifact.digest = runmat_geometry_core::GeometryDigest::from_bytes([99; 32]);
    assert!(import_exact_geometry_objects(
        &MemoryCache::default(),
        wrong_root_document,
        prepared.root_reference(),
        ObjectInventoryLimits::default(),
    )
    .is_err());
}

#[test]
fn exact_geometry_object_count_and_total_bytes_are_hard_limits() {
    let (document, topology, evaluators) = fixture::geometry();
    assert!(prepare_exact_geometry_objects(
        document,
        topology,
        evaluators,
        None,
        ObjectInventoryLimits {
            max_objects: 2,
            ..ObjectInventoryLimits::default()
        },
    )
    .is_err());

    let (document, topology, evaluators) = fixture::geometry();
    let prepared = prepare_exact_geometry_objects(
        document,
        topology,
        evaluators,
        None,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let mut cache = MemoryCache::default();
    cache.insert_all(&prepared.objects);
    assert!(import_exact_geometry_objects(
        &cache,
        prepared.document.clone(),
        prepared.root_reference(),
        ObjectInventoryLimits {
            max_total_bytes: prepared.root.encoded_length + 1,
            ..ObjectInventoryLimits::default()
        },
    )
    .is_err());
}
