use runmat_execution::value::ValueRefKind;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_meshing_core::{
    MeshingDomainModel, RegionMaterialAssignment, MESHING_DOMAIN_MODEL_SCHEMA_VERSION,
};

use crate::tests::MemoryCache;
use crate::{
    import_domain_model_input, import_domain_model_objects, prepare_domain_model_input,
    prepare_domain_model_objects, MeshingArtifactAccess,
};

fn model() -> MeshingDomainModel {
    let (_, topology, _) = runmat_geometry_fixtures::exact_tetrahedron();
    MeshingDomainModel {
        schema_version: MESHING_DOMAIN_MODEL_SCHEMA_VERSION,
        region_materials: vec![RegionMaterialAssignment {
            region_id: topology.regions[0].id.clone(),
            material_id: "steel".into(),
        }],
        contact_ids: Vec::new(),
    }
}

fn access() -> MeshingArtifactAccess {
    MeshingArtifactAccess {
        authorization_scope: "domain-model-run".into(),
        encryption_context: runmat_execution::Digest::sha256(b"domain-model-context"),
    }
}

#[test]
fn domain_model_round_trips_through_one_shared_input_object() {
    let prepared = prepare_domain_model_objects(model(), ObjectInventoryLimits::default()).unwrap();
    let mut cache = MemoryCache::default();
    cache.insert_all(&prepared.objects);
    let imported = import_domain_model_objects(
        &cache,
        prepared.root_reference(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(imported, prepared);
    assert_eq!(
        imported.objects[0].descriptor.namespace,
        runmat_execution_artifact::ObjectNamespace::InputValue
    );
}

#[test]
fn domain_model_import_rehashes_cache_and_enforces_limits() {
    let prepared = prepare_domain_model_objects(model(), ObjectInventoryLimits::default()).unwrap();
    let mut cache = MemoryCache::default();
    cache.insert_all(&prepared.objects);
    cache.replace(prepared.root.digest, b"poisoned".to_vec());
    assert!(import_domain_model_objects(
        &cache,
        prepared.root_reference(),
        ObjectInventoryLimits::default(),
    )
    .is_err());
    assert!(prepare_domain_model_objects(
        prepared.model,
        ObjectInventoryLimits {
            max_objects: 0,
            ..ObjectInventoryLimits::default()
        },
    )
    .is_err());
}

#[test]
fn domain_model_input_is_driver_owned_and_authority_bound() {
    let objects = prepare_domain_model_objects(model(), ObjectInventoryLimits::default()).unwrap();
    let access = access();
    let prepared =
        prepare_domain_model_input(objects, access.clone(), ObjectInventoryLimits::default())
            .unwrap();
    assert_eq!(prepared.input_objects().len(), 1);
    assert_eq!(prepared.root_input().kind, ValueRefKind::DriverObject);

    let mut cache = MemoryCache::default();
    cache.insert_all(&prepared.domain_model_objects().objects);
    assert_eq!(
        import_domain_model_input(
            &cache,
            prepared.root_input(),
            access.clone(),
            ObjectInventoryLimits::default(),
        )
        .unwrap(),
        prepared
    );

    let mut wrong = prepared.root_input().clone();
    wrong.authorization_scope = "wrong-run".into();
    assert!(
        import_domain_model_input(&cache, &wrong, access, ObjectInventoryLimits::default(),)
            .is_err()
    );
}
