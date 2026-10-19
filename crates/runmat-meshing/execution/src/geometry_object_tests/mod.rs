use runmat_execution_artifact::object::ObjectInventoryLimits;

use crate::tests::MemoryCache;
use runmat_execution::value::ValueRefKind;
use runmat_geometry_core::{
    ExactCurveImplementation, KernelEvaluatorRef, KERNEL_REPRESENTATION_MEDIA_TYPE,
};

use crate::{
    import_exact_geometry_input, import_exact_geometry_objects, prepare_exact_geometry_input,
    prepare_exact_geometry_objects, ArtifactAccess,
};

#[test]
fn exact_geometry_round_trips_through_shared_input_objects() {
    let (document, topology, evaluators) = runmat_geometry_fixtures::exact_circle();
    let prepared = prepare_exact_geometry_objects(
        document,
        topology,
        evaluators,
        None,
        None,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let mut cache = MemoryCache::default();
    cache.insert_all(&prepared.objects);
    let reads = crate::tests::read_buffers::ReadBuffers::new(&cache);
    let imported = import_exact_geometry_objects(
        &reads,
        prepared.document.clone(),
        prepared.root_reference(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();

    reads.assert_retained(&imported.objects);
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
fn kernel_representation_is_a_verified_transfer_object() {
    let (document, topology, mut evaluators) = runmat_geometry_fixtures::exact_circle();
    let representation = b"DBRep_DrawableShape\nexact-circle".to_vec();
    evaluators.curves[0].implementation = ExactCurveImplementation::Kernel {
        reference: KernelEvaluatorRef {
            entity_token: "edge:part-definition:edge-circle".into(),
            representation_digest: *runmat_execution::Digest::sha256(&representation).bytes(),
        },
    };
    let original = representation.clone();
    let original_pointer = original.as_ptr();
    let prepared = prepare_exact_geometry_objects(
        document,
        topology,
        evaluators,
        Some(original),
        None,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let representation_object = prepared
        .objects
        .iter()
        .find(|object| object.descriptor.media_type == KERNEL_REPRESENTATION_MEDIA_TYPE)
        .unwrap();
    assert_eq!(representation_object.bytes, representation);
    assert_eq!(representation_object.bytes.as_ptr(), original_pointer);
    assert_eq!(
        prepared.kernel_representation().unwrap().unwrap().as_ptr(),
        original_pointer
    );

    let mut cache = MemoryCache::default();
    cache.insert_all(&prepared.objects);
    let reads = crate::tests::read_buffers::ReadBuffers::new(&cache);
    let imported = import_exact_geometry_objects(
        &reads,
        prepared.document.clone(),
        prepared.root_reference(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    reads.assert_retained(&imported.objects);
    assert_eq!(imported, prepared);
    let imported_pointer = imported.kernel_representation().unwrap().unwrap().as_ptr();
    let components = imported.into_components().unwrap();
    assert_eq!(
        components.kernel_representation.as_ref().unwrap().as_ptr(),
        imported_pointer
    );
    let mut missing = prepared.clone();
    missing
        .objects
        .retain(|object| object.descriptor.media_type != KERNEL_REPRESENTATION_MEDIA_TYPE);
    assert!(
        missing.kernel_representation().is_err(),
        "missing native representation must not become a portable fallback"
    );

    cache.replace(
        representation_object.descriptor.digest,
        b"poisoned".to_vec(),
    );
    assert!(import_exact_geometry_objects(
        &cache,
        prepared.document.clone(),
        prepared.root_reference(),
        ObjectInventoryLimits::default(),
    )
    .is_err());
}

#[test]
fn exact_geometry_import_rehashes_cache_and_binds_document_root() {
    let (document, topology, evaluators) = runmat_geometry_fixtures::exact_circle();
    let prepared = prepare_exact_geometry_objects(
        document,
        topology,
        evaluators,
        None,
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
    let (document, topology, evaluators) = runmat_geometry_fixtures::exact_circle();
    assert!(prepare_exact_geometry_objects(
        document,
        topology,
        evaluators,
        None,
        None,
        ObjectInventoryLimits {
            max_objects: 2,
            ..ObjectInventoryLimits::default()
        },
    )
    .is_err());

    let (document, topology, evaluators) = runmat_geometry_fixtures::exact_circle();
    let prepared = prepare_exact_geometry_objects(
        document,
        topology,
        evaluators,
        None,
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

#[test]
fn exact_geometry_projects_to_complete_driver_owned_input_inventory() {
    let (document, topology, evaluators) = runmat_geometry_fixtures::exact_circle();
    let objects = prepare_exact_geometry_objects(
        document,
        topology,
        evaluators,
        None,
        None,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let access = access();
    let prepared =
        prepare_exact_geometry_input(objects, access.clone(), ObjectInventoryLimits::default())
            .unwrap();

    assert_eq!(
        prepared.input_objects().len(),
        prepared.geometry_objects().objects.len()
    );
    assert!(prepared
        .input_objects()
        .iter()
        .all(|reference| reference.kind == ValueRefKind::DriverObject));
    assert_eq!(
        prepared.root_input().logical_digest,
        prepared.geometry_objects().root.digest
    );

    let mut cache = MemoryCache::default();
    cache.insert_all(&prepared.geometry_objects().objects);
    let reads = crate::tests::read_buffers::ReadBuffers::new(&cache);
    let imported = import_exact_geometry_input(
        &reads,
        prepared.geometry_objects().document.clone(),
        prepared.root_input(),
        access,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    reads.assert_retained(&imported.geometry_objects().objects);
    assert_eq!(imported, prepared);
}

#[test]
fn exact_geometry_input_rejects_wrong_authority_and_object_kind() {
    let (document, topology, evaluators) = runmat_geometry_fixtures::exact_circle();
    let objects = prepare_exact_geometry_objects(
        document,
        topology,
        evaluators,
        None,
        None,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let access = access();
    let prepared =
        prepare_exact_geometry_input(objects, access.clone(), ObjectInventoryLimits::default())
            .unwrap();
    let mut cache = MemoryCache::default();
    cache.insert_all(&prepared.geometry_objects().objects);

    let mut wrong = prepared.root_input().clone();
    wrong.authorization_scope = "another-run".into();
    assert!(import_exact_geometry_input(
        &cache,
        prepared.geometry_objects().document.clone(),
        &wrong,
        access.clone(),
        ObjectInventoryLimits::default(),
    )
    .is_err());

    wrong = prepared.root_input().clone();
    wrong.kind = ValueRefKind::WorkerObject;
    assert!(import_exact_geometry_input(
        &cache,
        prepared.geometry_objects().document.clone(),
        &wrong,
        access,
        ObjectInventoryLimits::default(),
    )
    .is_err());
}

fn access() -> ArtifactAccess {
    ArtifactAccess {
        authorization_scope: "geometry-run".into(),
        encryption_context: runmat_execution::Digest::sha256(b"geometry-context"),
    }
}
